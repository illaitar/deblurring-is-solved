"""Upload the companion data of the site to a Hugging Face dataset repository (run where you are logged in).

    huggingface-cli login                          # once, with a token that can write
    python tools/hf_upload.py OWNER/NAME [--private] [--skip-img SET,...] [--dry]

Uploads hf/README.md and hf/LICENSE.md, results/frames/{frames,methods,scores}.parquet and bench.sqlite (if present)
to data/, and results/frames/img/ (viewer bundles and index.json) to img/. --skip-img leaves out the viewer frames of
the named sets (e.g. bsd2,dvd, whose authors state no licence for redistribution). Prints the --img-base to pass to
tools/build_site.py afterwards.
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
args = sys.argv[1:]
repo = args[0]
private = "--private" in args
dry = "--dry" in args
skip = set(args[args.index("--skip-img") + 1].split(",")) if "--skip-img" in args else set()
fr = ROOT / "results" / "frames"
ops = [(ROOT / "hf" / "README.md", "README.md"), (ROOT / "hf" / "LICENSE.md", "LICENSE.md")]
ops += [(fr / f, f"data/{f}") for f in ("frames.parquet", "methods.parquet", "bench.sqlite") if (fr / f).exists()]
ops += [(p, f"data/{p.name}") for p in sorted(fr.glob("scores_*.parquet"))]     # 26.11: one scores file per test set
img = fr / "img"
for p in sorted(img.rglob("*")):
    if p.is_file() and not (p.parent.name in skip and p.suffix == ".bin"):
        ops.append((p, "img/" + p.relative_to(img).as_posix()))
total = sum(p.stat().st_size for p, _ in ops)
print(f"{len(ops)} files, {total / 2 ** 20:.0f} MB -> datasets/{repo} ({'private' if private else 'public'})")
if dry:
    for p, q in ops[:10]:
        print(" ", q)
    sys.exit(0)
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
from huggingface_hub import CommitOperationAdd  # noqa: E402
api.create_commit(repo_id=repo, repo_type="dataset", commit_message="frames, per-frame scores and viewer bundles",
                  operations=[CommitOperationAdd(path_in_repo=q, path_or_fileobj=str(p)) for p, q in ops])
print("done; build the site with:")
print(f"  python tools/build_site.py --img-base https://huggingface.co/datasets/{repo}/resolve/main/")
