"""Upload the dbis / dbisf weight files to Hugging Face, one model repository per model (run where you are logged in).

    python tools/hf_upload_dbis.py WEIGHTS_DIR [MODEL ...] [--dry] [--private] [--index]

WEIGHTS_DIR holds the files made by `python -m dbis.convert` (weights/dbis/ by default). Every model of dbis.MODELS
with a recorded sha256 (all, or the MODELs named) goes to dbis.repo(name) = Illaitar/<model>: its file, checked against
the sha256, a JSON card (from dbis.MODELS and dbis.SOURCES), a model card README.md (the method, the file with its
numbers from results/dbis/<source>.json, how to load it), LICENSE.md and the license texts of hf/dbis/licenses/ of its
source. Files of the repository that are not among these are deleted in a commit of their own (the history keeps them).
tools/hf_collection_dbis.py then puts the repositories into the collection dbis.COLLECTION. --index rewrites only the
card of the former all-in-one repository Illaitar/dbis to point to the collection (its files stay).
"""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from dbis import COLLECTION, HF_OWNER, MODELS, SOURCES, repo  # noqa: E402

PIP = 'pip install "deblurring-is-solved @ git+https://github.com/illaitar/deblurring-is-solved"'
CODE = "https://github.com/illaitar/deblurring-is-solved"
SITE = "https://illaitar.github.io/deblurring-is-solved/"
LIC = ROOT / "hf" / "dbis" / "licenses"
# per source: the Hub license id, the license texts in hf/dbis/licenses/, and the terms row of LICENSE.md
TERMS = {
    "NAFNet-64": ("mit", ["NAFNet-LICENSE.txt"],
                  "`NAFNet-GoPro-width64.pth` of [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) "
                  "(commit 2b4af71) | MIT, Copyright (c) 2022 megvii-model (`licenses/NAFNet-LICENSE.txt`)"),
    "BSSTNet": ("other", ["RAFT-LICENSE.txt"],
                "`BSST_gopro.pth` of [huicongzhang/BSSTNet](https://github.com/huicongzhang/BSSTNet) (commit c35a43c) "
                "| MIT, as its README states (\"This project is open sourced under MIT license\"); the repository has "
                "no LICENSE file |\n| the same file | `raft-things.pth` of [princeton-vl/RAFT](https://github.com/"
                "princeton-vl/RAFT) | BSD 3-Clause, Copyright (c) 2020, princeton-vl (`licenses/RAFT-LICENSE.txt`)"),
    "DSTNetPlus-L": ("apache-2.0", ["DSTNet-plus-LICENSE.txt"],
                     "`DSTNetPlus_L_gopro.pth` of [sunny2109/DSTNet-plus](https://github.com/sunny2109/DSTNet-plus) "
                     "(release v0.1.0) | Apache License 2.0 (`licenses/DSTNet-plus-LICENSE.txt`)"),
    "Turtle": ("mit", ["Turtle-LICENSE.txt"],
               "`GoPro_Deblur.pth` of [Ascend-Research/Turtle](https://github.com/Ascend-Research/Turtle) (commit "
               "7094f42) | MIT, Copyright (c) 2024 Huawei Technologies Co., Ltd. (`licenses/Turtle-LICENSE.txt`)"),
    "Concertormer": ("mit", ["Concertormer-LICENSE.txt"],
                     "`Concertormer.pth` (CSA_models.zip, models/GoPro/) of [setsunil/Concertormer](https://github.com/"
                     "setsunil/Concertormer) (commit 94269c5) | MIT, Copyright (c) 2022 megvii-model "
                     "(`licenses/Concertormer-LICENSE.txt`)"),
    "Concertormer-lite": ("mit", ["Concertormer-LICENSE.txt"],
                          "`Concertormer_lite.pth` (CSA_models.zip, models/GoPro/) of [setsunil/Concertormer]("
                          "https://github.com/setsunil/Concertormer) (commit 94269c5) | MIT, Copyright (c) 2022 "
                          "megvii-model (`licenses/Concertormer-LICENSE.txt`)"),
}
VIDEO = {"BSSTNet", "DSTNetPlus-L", "Turtle"}
CHANGES = ("Changes made to the source checkpoint (also stated for the Apache License 2.0, section 4): tensors renamed "
           "to the names of our code, tensors folded into the next layers where that is exact up to float rounding, "
           "fixed tensors that the code rebuilds dropped, and in dbisf files the weights rounded per output channel "
           "and entropy coded. The conversion is `python -m dbis.convert` of " + CODE + "; it checks the md5 of every "
           "source checkpoint (listed in the file's JSON card).")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _ms(x):
    return f"{x:.1f} ms" if x < 1000 else f"{x / 1000:.2f} s"


def card(name, size):
    m = MODELS[name]
    source = m["source"]
    src, (lic, texts, row) = SOURCES[source], TERMS[source]
    fast = name.startswith("dbisf-")
    res = ROOT / "results" / "dbis" / f"{source}.json"
    site = json.loads(res.read_text(encoding="utf-8"))["site"] if res.exists() else None
    tags = ["deblurring", "motion-deblurring"] + (["video-deblurring"] if source in VIDEO else []) + \
        ["image-restoration", "dbis"]
    head = ["---", f"license: {lic}"] + (["license_name: mit-and-bsd-3-clause", "license_link: LICENSE.md"]
                                         if lic == "other" else []) + \
        ["library_name: pytorch", "pipeline_tag: image-to-image", "tags:"] + [f"- {t}" for t in tags] + ["---", ""]
    what = ("our fast and light version: a shorter run time and a smaller file, at most 0.05 dB below the authors' mean "
            "PSNR on every test set of the benchmark" if fast else
            "our code and weight file, the same outputs as the authors' code")
    sib = [n for n, x in MODELS.items() if x["source"] == source and n != name and x["sha256"]]
    lines = [f"# {name}", "",
             f"{src['method']} ({src['paper']}, {src['venue']}) in plain PyTorch, {what}. Made from the authors' GoPro "
             f"checkpoint by `python -m dbis.convert` of [deblurring is solved?]({CODE}) ([benchmark site]({SITE})); "
             f"the authors' code: {src['code']}.", ""]
    if sib:
        lines.append("The same method " + ("with the authors' outputs" if fast else "faster and lighter") + ": "
                     + ", ".join(f"[{HF_OWNER}/{s}](https://huggingface.co/{HF_OWNER}/{s})" for s in sib) + ". ")
    lines += [f"Every model: the [dbis collection]({COLLECTION}).", ""]
    run = (f"python -m dbis.run {name} blurry_frames/ sharp_frames/   # a folder of frames (one clip) or a video file"
           if source in VIDEO else f"python -m dbis.run {name} blurry.png sharp.png   # an image, a folder of frames or a video")
    lines += ["## Use", "", "```bash", PIP.replace("solved @", "solved[bsst] @") if source == "BSSTNet" else PIP, run,
              "```", "", "```python", "import dbis",
              f'net = dbis.load("{name}")   # fetches the file from this repository on first use and checks its sha256',
              "out = net.restore(frames)   # frames [T, 3, H, W], [B, T, 3, H, W] or [3, H, W]", "```", "", "## File", ""]
    kd = "dbisf" if fast else "dbis"
    if site:
        lines += ["| File | Size | GoPro PSNR, dB | Time per 720p frame |", "|---|---:|---:|---:|",
                  f"| `{m['file']}` | {size / 1e6:.0f} MB | {site['gopro_psnr'][kd]:.4f} | {_ms(site['latency_ms'][kd])} |",
                  f"| the authors' code and checkpoint | {site['weights_bytes']['authors'] / 1e6:.0f} MB | "
                  f"{site['gopro_psnr']['authors']:.4f} | {_ms(site['latency_ms']['authors'])} |", "",
                  "PSNR: the 1,111 GoPro test frames, 8-bit outputs. Time: the benchmark's latency harness, one A100, "
                  "batch 1 ([DBIS.md](" + CODE + "/blob/main/DBIS.md) has every number and option)."]
    else:
        lines += ["| File | Size |", "|---|---:|", f"| `{m['file']}` | {size / 1e6:.0f} MB |", "",
                  "The outputs equal those of the authors' code on the GoPro test frames we compared."]
    lines += ["", "`" + Path(m["file"]).stem + ".json` here has the file's sha256, the source checkpoint and its md5, and "
              "the network's arguments. " + ("`.wpk` is the repository's wpack format (`dbis/wpack.py`, needs zstandard)."
                                             if fast else "`.safetensors`, float32."), "", "## Terms", "",
              "The file keeps the terms of the checkpoint it is made from: see `LICENSE.md`"
              + (" and `licenses/`" if texts else "") + ". Cite the method's paper when you use its weights.", ""]
    terms = ["# Terms of the weight file", "", "| File | Source checkpoint | Terms |", "|---|---|---|",
             f"| `{m['file']}` | " + row + " |", "", CHANGES, ""]
    return "\n".join(head + lines), "\n".join(terms), texts


def index_card():
    return "\n".join(["---", "license: other", "license_name: per-file-terms-of-the-authors", "license_link: LICENSE.md",
                      "library_name: pytorch", "pipeline_tag: image-to-image", "---", "",
                      "# dbis: one repository per model now", "",
                      f"The weight files of dbis and dbisf ([deblurring is solved?]({CODE})) are hosted in one "
                      f"repository per model, named as the model, each with the terms of its source and its card: the "
                      f"[dbis collection]({COLLECTION}). `dbis.load` fetches from there. The files here are the same and "
                      "stay for older versions of the package.", ""])


def prune(api, r, keep):
    """Delete the files of repository r that are not in keep (.gitattributes stays), in one commit."""
    import hf_push
    from huggingface_hub import CommitOperationDelete
    have = hf_push._retry(f"list {r}", lambda: api.list_repo_files(r))
    gone = [f for f in have if f not in keep and f != ".gitattributes"]
    if gone:
        hf_push._retry(f"prune {r}", lambda: api.create_commit(
            r, [CommitOperationDelete(path_in_repo=f) for f in gone], commit_message="files of other models moved to "
            "their own repositories"))
    print(f"  {r}: {len(gone)} files of other models deleted ({', '.join(gone) or 'none'})")


args = sys.argv[1:]
dry, private = "--dry" in args, "--private" in args
pos = [a for a in args if not a.startswith("--")]
if "--index" in args:
    stage = Path(tempfile.mkdtemp(prefix="hf_dbis_index_"))
    (stage / "README.md").write_text(index_card(), encoding="utf-8")
    print((stage / "README.md").read_text(encoding="utf-8"))
    if not dry:
        import hf_push
        hf_push.push(stage, f"{HF_OWNER}/dbis", "model", message="point to the repositories of the models")
    sys.exit(0)
wdir, only = Path(pos[0]), set(pos[1:])
api = None
for name, m in MODELS.items():
    if not m["sha256"] or (only and name not in only):
        continue
    p = wdir / m["file"]
    sha = sha256(p)
    if sha != m["sha256"]:
        sys.exit(f"{p}: sha256 {sha} differs from the recorded {m['sha256']}")
    src = SOURCES[m["source"]]
    c = {"model": name, "file": m["file"], "bytes": p.stat().st_size, "sha256": sha, "store": m["store"],
         "class": f"dbis.{src['arch']}.{m['cls']}", "args": m["args"],
         "source": {k: v for k, v in src.items() if k != "arch"}, "load": f'import dbis; net = dbis.load("{name}")',
         "code": CODE}
    readme, terms, texts = card(name, p.stat().st_size)
    ops = [(p, m["file"]), (json.dumps(c, indent=1, ensure_ascii=False).encode("utf-8"), Path(m["file"]).stem + ".json"),
           (readme.encode("utf-8"), "README.md"), (terms.encode("utf-8"), "LICENSE.md")]
    ops += [(LIC / t, "licenses/" + t) for t in texts]
    r = repo(name)
    size = sum(len(x) if isinstance(x, bytes) else x.stat().st_size for x, _ in ops)
    print(f"{r}: {len(ops)} files, {size / 2 ** 20:.0f} MiB ({'private' if private else 'public'})")
    for x, q in ops:
        print(f"  {q}  {len(x) if isinstance(x, bytes) else x.stat().st_size:,} B")
    if dry:
        print(readme)
        continue
    # one folder with the repository's files (the weights hard-linked or copied), pushed in small commits by
    # tools/hf_push.py (large requests to the Hub stall from the benchmark's cluster; the license texts go through Xet)
    import huggingface_hub
    from huggingface_hub import HfApi

    import hf_push
    stage = Path(tempfile.mkdtemp(prefix=f"hf_{name}_", dir=wdir.parent))
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
    hf_push.push(stage, r, "model", lfs=["*.wpk", "licenses/*.txt"], message=f"{name}: weight file, cards and terms",
                 private=private)
    if api is None:
        huggingface_hub.set_client_factory(hf_push._client_factory)
        api = HfApi()
    prune(api, r, {q for _, q in ops})
    print(f"file at {repo(name)}: https://huggingface.co/{r}/resolve/main/{m['file']} (staged links left in {stage})")
