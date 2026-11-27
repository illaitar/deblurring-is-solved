"""Upload the dbis / dbisf weight files to a Hugging Face model repository (run where you are logged in).

    python tools/hf_upload_dbis.py WEIGHTS_DIR OWNER/NAME [dist/dbis-<version>-py3-none-any.whl] [--dry] [--private]

WEIGHTS_DIR holds the files made by `python -m dbis.convert` (weights/dbis/ by default). Every file of dbis.MODELS is
checked against its recorded sha256; the repository gets the files, a JSON card per file (made from dbis.MODELS and
dbis.SOURCES), hf/dbis/README.md as the model card, hf/dbis/LICENSE.md and the license texts of hf/dbis/licenses/.
A wheel of the package (python -m pip wheel . --no-deps -w dist) given after the repository goes up next to the files,
for pip install "dbis @ https://huggingface.co/OWNER/NAME/resolve/main/<wheel>".
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dbis import MODELS, SOURCES  # noqa: E402

args = sys.argv[1:]
wdir, repo = Path(args[0]), args[1]
dry, private = "--dry" in args, "--private" in args
wheels = [Path(a) for a in args[2:] if a.endswith(".whl")]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


ops = []
for name, m in MODELS.items():
    p = wdir / m["file"]
    sha = sha256(p)
    if sha != m["sha256"]:
        sys.exit(f"{p}: sha256 {sha} differs from the recorded {m['sha256']}")
    src = SOURCES[m["source"]]
    card = {"model": name, "file": m["file"], "bytes": p.stat().st_size, "sha256": sha, "store": m["store"],
            "class": f"dbis.{src['arch']}.{m['cls']}", "args": m["args"],
            "source": {k: v for k, v in src.items() if k != "arch"},
            "load": f'import dbis; net = dbis.load("{name}")',
            "code": "https://github.com/illaitar/deblurring-is-solved"}
    ops.append((p, m["file"]))
    ops.append((json.dumps(card, indent=1, ensure_ascii=False).encode("utf-8"), m["file"].rsplit(".", 1)[0] + ".json"))
hf = ROOT / "hf" / "dbis"
ops += [(hf / "README.md", "README.md"), (hf / "LICENSE.md", "LICENSE.md")]
ops += [(p, "licenses/" + p.name) for p in sorted((hf / "licenses").glob("*.txt"))]
ops += [(w, w.name) for w in wheels]
size = sum(len(x) if isinstance(x, bytes) else x.stat().st_size for x, _ in ops)
print(f"{len(ops)} files, {size / 2 ** 20:.0f} MiB -> {repo} (model, {'private' if private else 'public'})")
for x, q in ops:
    print(f"  {q}  {len(x) if isinstance(x, bytes) else x.stat().st_size:,} B")
if dry:
    sys.exit(0)
# one folder with the repository's files (the weights hard-linked or copied), pushed in small commits by
# tools/hf_push.py (large requests to the Hub stall from the benchmark's cluster; the license texts go through Xet)
import shutil  # noqa: E402
import tempfile  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
import hf_push  # noqa: E402
stage = Path(tempfile.mkdtemp(prefix="hf_dbis_", dir=wdir.parent))
for x, q in ops:
    t = stage / q
    t.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(x, bytes):
        t.write_bytes(x)
    else:
        try:
            t.hardlink_to(x)
        except OSError:
            shutil.copy2(x, t)
hf_push.push(stage, repo, "model", lfs=["*.wpk", "*.whl", "licenses/*.txt"], message="dbis and dbisf weight files, cards and terms",
             private=private)
print(f"files at https://huggingface.co/{repo}/resolve/main/<file> (staged links left in {stage})")
