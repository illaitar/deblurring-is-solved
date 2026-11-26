"""Push a local folder to a Hugging Face repository in small commits (run where you are logged in).

    python tools/hf_push.py LOCAL_DIR OWNER/NAME --type model|dataset [--lfs PATTERN ...] [--message TEXT] [--dry]

Every request to the Hub API stays small, and every request is retried after a 60 s timeout. On the cluster of this
benchmark (26.11) transfers of more than about 20 KB from a Python built with OpenSSL 3.5 stall: its default TLS key
exchange (the post-quantum X25519MLKEM768 hybrid) does not get through the network there. curl and a Python with
OpenSSL 3.0 are fine; with OpenSSL 3.5, point OPENSSL_CONF at a file that limits the groups:

    openssl_conf = c
    [c]
    ssl_conf = s
    [s]
    system_default = d
    [d]
    Groups = X25519:P-256

The Xet client (Rust, its own TLS) is hit too: 85 % of its transfers failed there, about 2.6 MB/s; with
HF_HUB_DISABLE_XET=1 the files go as Git LFS uploads through Python instead (the 6.9 GB of test sets in about a
minute). The files whose paths match a --lfs pattern are declared in .gitattributes (their contents then go as LFS
files like the binary ones), and the files are committed in batches whose preupload and commit requests stay under
MAX_REQUEST bytes, so the push also works without the OpenSSL file. Files already in the repository with the same
content are skipped: the push can be run again after an interruption.

    OPENSSL_CONF=openssl_nopq.cnf HF_HUB_DISABLE_XET=1 python tools/hf_push.py ...    # as run on 26.11
"""
import fnmatch
import hashlib
import sys
from pathlib import Path

MAX_REQUEST = 15000          # bytes of a preupload or commit request body
PRE_COST, LFS_COST = 900, 350  # per file: preupload entry (512-byte sample in base64 and the path), commit pointer
LFS_LINE = "{} filter=lfs diff=lfs merge=lfs -text"


def _client_factory():
    import httpx2
    from huggingface_hub.utils import _http
    return httpx2.Client(event_hooks={"request": [_http.hf_request_event_hook]}, follow_redirects=True,
                         timeout=httpx2.Timeout(60.0))           # fail and retry instead of waiting forever


def _retry(what, fn, tries=6):
    """Some requests from the cluster stall at random (26.11): every call to the Hub is retried after a timeout."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            print(f"  {what}: {type(e).__name__} {str(e)[:100]} (try {k + 1} of {tries})", flush=True)
    sys.exit(f"{what} failed {tries} times")


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha1(p):
    data = Path(p).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _is_lfs(rel, path, patterns):
    if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(Path(rel).name, pat) for pat in patterns):
        return True
    if path.stat().st_size > 10 * 2 ** 20:
        return True
    with open(path, "rb") as f:
        return b"\0" in f.read(8192)                       # binary, as the Hub decides from the sample


def push(local, repo, repo_type, lfs=(), message="upload", dry=False, private=False):
    import huggingface_hub
    from huggingface_hub import CommitOperationAdd, HfApi
    huggingface_hub.set_client_factory(_client_factory)
    api = HfApi()
    local = Path(local)
    files = sorted(p for p in local.rglob("*") if p.is_file() and ".cache" not in p.relative_to(local).parts)
    if dry:
        for p in files:
            print(" ", p.relative_to(local).as_posix(), p.stat().st_size)
        print(len(files), "files")
        return
    _retry("create_repo", lambda: api.create_repo(repo, repo_type=repo_type, private=private, exist_ok=True))
    # .gitattributes with the --lfs patterns
    import tempfile
    ga = _retry(".gitattributes", lambda: api.hf_hub_download(repo, ".gitattributes", repo_type=repo_type,
                                                              cache_dir=tempfile.mkdtemp(prefix="hf_ga_")))
    lines = Path(ga).read_text().splitlines()
    new = [LFS_LINE.format(p) for p in lfs if LFS_LINE.format(p) not in lines]
    if new:
        _retry("commit .gitattributes", lambda: api.create_commit(
            repo, repo_type=repo_type, commit_message="LFS patterns: " + ", ".join(lfs),
            operations=[CommitOperationAdd(".gitattributes", ("\n".join(lines + new) + "\n").encode())]))
    patterns = [ln.split()[0] for ln in lines + new if "filter=lfs" in ln]
    # what is there already
    have = {}
    for e in _retry("list files", lambda: list(api.list_repo_tree(repo, repo_type=repo_type, recursive=True))):
        if getattr(e, "size", None) is not None:
            have[e.path] = e.lfs.sha256 if getattr(e, "lfs", None) else e.blob_id
    todo, skipped = [], 0
    for p in files:
        rel = p.relative_to(local).as_posix()
        big = _is_lfs(rel, p, patterns)
        if rel in have and have[rel] == (_sha256(p) if big else _git_sha1(p)):
            skipped += 1
            continue
        cost = LFS_COST if big else int(p.stat().st_size * 4 / 3) + 400
        if cost > MAX_REQUEST - PRE_COST:
            sys.exit(f"{rel} ({p.stat().st_size} B) is too large to go inline: add a --lfs pattern for it")
        todo.append((rel, p, cost))
    print(f"{len(files)} files: {skipped} already in {repo}, {len(todo)} to push", flush=True)
    batches, cur, pre, body = [], [], 0, 0
    for rel, p, cost in todo:
        if cur and (pre + PRE_COST > MAX_REQUEST or body + cost > MAX_REQUEST):
            batches.append(cur)
            cur, pre, body = [], 0, 0
        cur.append((rel, p))
        pre, body = pre + PRE_COST, body + cost
    if cur:
        batches.append(cur)
    for k, batch in enumerate(batches, 1):
        _retry(f"commit {k}", lambda: api.create_commit(
            repo, repo_type=repo_type, commit_message=f"{message} ({k}/{len(batches)})",
            operations=[CommitOperationAdd(r, str(q)) for r, q in batch]))
        print(f"  commit {k}/{len(batches)}: {len(batch)} files, last {batch[-1][0]}", flush=True)
    print(f"done: {len(todo)} files in {len(batches)} commits", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 2 or "--type" not in a:
        sys.exit(__doc__)
    lfs = [a[i + 1] for i, x in enumerate(a) if x == "--lfs"]
    msg = a[a.index("--message") + 1] if "--message" in a else "upload"
    push(a[0], a[1], a[a.index("--type") + 1], lfs, msg, "--dry" in a, "--private" in a)
