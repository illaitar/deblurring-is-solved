"""Make the dbis / dbisf weight files from the authors' checkpoint.

    python -m dbis.convert SOURCE CHECKPOINT [EXTRA ...] [--out DIR] [--force]
    python -m dbis.convert NAFNet-64 NAFNet-GoPro-width64.pth
    python -m dbis.convert BSSTNet BSST_gopro.pth raft-things.pth

The checkpoint (and the further checkpoints a source lists under "extra", e.g. the RAFT weights of BSSTNet) is read with
weights_only=True (no code in it runs) and checked against the md5 of the released file (--force converts anyway). Its
model tensors are folded by the fold() of dbis/<arch>.py, loaded strictly into every model made from this source, and
written:
  .safetensors  float32 (dbis)
  .wpk          wpack qc48: every weight tensor of 2+ dimensions rounded per output channel to std(channel) / 48,
                entropy coded (dbisf); q128 (one step std / 128 per tensor) is the earlier store, still readable
A JSON card is written next to each file. Its sha256 is compared with the one in dbis/__init__.py; a different BLAS
can change the last bit of a folded bias, so a mismatch is reported in the card and on the screen, not an error.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from . import MODELS, SOURCES, WEIGHTS, build


def digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    import importlib

    import torch
    ap = argparse.ArgumentParser(prog="python -m dbis.convert", description=__doc__.split("\n\n")[0])
    ap.add_argument("source", choices=sorted(SOURCES))
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("extra", type=Path, nargs="*", help="the further checkpoints of the source (SOURCES[...]['extra'])")
    ap.add_argument("--out", type=Path, default=WEIGHTS)
    ap.add_argument("--force", action="store_true", help="convert even if the md5 differs from the released file")
    a = ap.parse_args(argv)
    src = SOURCES[a.source]
    extra = src.get("extra", [])
    if len(a.extra) != len(extra):
        sys.exit(f"{a.source} is made from " + " ".join([src["file"]] + [e["file"] for e in extra]))
    md5s = []
    for path, s in zip([a.checkpoint, *a.extra], [src, *extra]):
        md5s.append(digest(path, "md5"))
        if md5s[-1] != s["md5"] and not a.force:
            sys.exit(f"{path}: md5 {md5s[-1]}, the released {s['file']} has {s['md5']} (use --force to convert anyway)")
    md5 = md5s[0]
    t0 = time.time()
    sds = [torch.load(path, map_location="cpu", weights_only=True) for path in [a.checkpoint, *a.extra]]
    sds = [d if s.get("group") is None else d[s["group"]] for d, s in zip(sds, [src, *extra])]
    sd = importlib.import_module(f".{src['arch']}", __package__).fold(*sds)
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {"source": src["file"], "source_md5": src["md5"], "code": src["code"], "license": src["license"]}
    if extra:
        meta["extra"] = [{"source": e["file"], "source_md5": e["md5"], "code": e["code"], "license": e["license"]}
                         for e in extra]
    for name, m in MODELS.items():
        if m["source"] != a.source:
            continue
        build(name, "cpu").load_state_dict(sd, strict=True)
        path = a.out / m["file"]
        if m["store"] == "float32":
            from safetensors.torch import save_file
            # one metadata key: safetensors keeps metadata in a hash map, so several keys come out in a random order
            save_file(sd, str(path), metadata={"dbis": json.dumps({**meta, "model": name}, sort_keys=True)})
        elif m["store"] in ("q128", "qc48"):
            from . import wpack   # zstandard is needed only here (dbisf files)
            tier, s = ("q", 128) if m["store"] == "q128" else ("qc", 48)
            wpack.pack({k: v.numpy() for k, v in sd.items()}, str(path), tier, s, threads=8, meta={**meta, "model": name})
        else:
            raise ValueError(m["store"])
        sha = digest(path, "sha256")
        card = {"model": name, "file": m["file"], "bytes": path.stat().st_size, "sha256": sha,
                "sha256_recorded": m["sha256"], "matches_recorded": sha == m["sha256"], "store": m["store"],
                "class": f"dbis.{src['arch']}.{m['cls']}", "args": m["args"],
                "source": {**src, "md5_checked": md5,
                           **({"extra": [{**e, "md5_checked": x} for e, x in zip(extra, md5s[1:])]} if extra else {})},
                "converted": time.strftime("%Y-%m-%d %H:%M"), "tool": "python -m dbis.convert"}
        path.with_suffix(".json").write_text(json.dumps(card, indent=1), encoding="utf-8")
        print(f"{name}: {path} ({path.stat().st_size / 2 ** 20:.1f} MiB), sha256 "
              + ("as recorded" if card["matches_recorded"] else
                 f"{sha} (none recorded yet)" if m["sha256"] is None else f"{sha} differs from the recorded {m['sha256']}"))
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
