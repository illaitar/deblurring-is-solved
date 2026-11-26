"""Upload the compact test sets to a Hugging Face dataset repository (run where you are logged in).

    python tools/hf_upload_testsets.py PACKED_DIR OWNER/NAME [--dry] [--private]

PACKED_DIR is the output of the packer (one folder per set with blur.mkv, sharp.mkv and frames.tsv per sequence, and
manifest.json). Every file is checked against the sha256 of manifest.json first; hf/testsets/README.md and LICENSE.md
are copied in as the dataset card and terms. The push (tools/hf_push.py) skips files already uploaded: run it again
after an interruption. Read the sets with tools/testsets.py.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
args = sys.argv[1:]
src, repo = Path(args[0]), args[1]
dry, private = "--dry" in args, "--private" in args


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


man = json.loads((src / "manifest.json").read_text())
n, size = 0, 0
for name, m in man.items():
    for rel, f in m["files"].items():
        if sha256(src / rel) != f["sha256"]:
            sys.exit(f"{rel}: sha256 differs from manifest.json")
        n, size = n + 1, size + f["bytes"]
    print(f"{name}: {m['frames']} pairs, {len(m['sequences'])} folders, {m['bytes'] / 1e9:.2f} GB")
print(f"{n} files, {size / 1e9:.2f} GB checked -> datasets/{repo} ({'private' if private else 'public'})")
for f in ("README.md", "LICENSE.md"):
    shutil.copy2(ROOT / "hf" / "testsets" / f, src / f)
if dry:
    sys.exit(0)
sys.path.insert(0, str(ROOT / "tools"))
import hf_push  # noqa: E402  (small commits: large requests to the Hub stall from the benchmark's cluster)
hf_push.push(src, repo, "dataset", lfs=["*.mkv", "*.tsv", "manifest.json"], message="compact test sets",
             private=private)
print(f"done: https://huggingface.co/datasets/{repo}")
