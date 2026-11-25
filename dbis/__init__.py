"""dbis and dbisf: methods of the benchmark re-implemented in this repository (see DBIS.md).

    dbis-<method>    our code and weight file, the same outputs as the authors' code (largest difference of an output
                     value about 1e-5, the same mean PSNR to the fourth decimal on the test set)
    dbisf-<method>   our fast and light version: shorter run time and a smaller weight file, at most 0.05 dB below the
                     authors' mean PSNR on every test set of the benchmark

The weight files are made locally from the authors' checkpoint:

    python -m dbis.convert NAFNet-64 NAFNet-GoPro-width64.pth     # -> weights/dbis/dbis-NAFNet-64.safetensors, dbisf-NAFNet-64.wpk
    python -m dbis.convert BSSTNet BSST_gopro.pth raft-things.pth # -> dbis-BSSTNet.safetensors, dbisf-BSSTNet.wpk (BSST + RAFT)

    import dbis
    net = dbis.load("dbisf-NAFNet-64")                            # eval mode, on the GPU
    out = net.restore(frames)                                     # the same call for every model (dbis/restore.py):
    out = net.restore(frames, tile=512, precision="fp16")         # frames [T, 3, H, W], [B, T, 3, H, W] or [3, H, W]

Dependencies: torch and numpy (BSSTNet: torchvision too); safetensors for dbis files, zstandard for dbisf files (the
wpack codec, dbis/wpack.py).
"""
import importlib
from pathlib import Path

WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "dbis"

# the authors' checkpoints the models are made from
SOURCES = {
    "NAFNet-64": {
        "method": "NAFNet", "paper": "Simple Baselines for Image Restoration", "venue": "ECCV 2022",
        "code": "https://github.com/megvii-research/NAFNet", "commit": "2b4af71ebe098a92a75910c233a3965a3e93ede4",
        "license": "MIT, Copyright (c) 2022 megvii-model",
        "url": "https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view",
        "file": "NAFNet-GoPro-width64.pth", "md5": "7b7f519ce4203d701dc026ff0c3fd6e0", "group": "params",
        "trained_on": "GoPro", "arch": "nafnet",
    },
    "BSSTNet": {
        "method": "BSSTNet", "paper": "Blur-aware Spatio-temporal Sparse Transformer for Video Deblurring",
        "venue": "CVPR 2024",
        "code": "https://github.com/huicongzhang/BSSTNet", "commit": "c35a43c669243ffc9566946491eac139dcd2680c",
        "license": "MIT (stated in the README of the repository)",
        "url": "https://drive.google.com/drive/folders/19v8wsg8aWayaVhNBmnj2vk4LrvmdViW8",
        "file": "BSST_gopro.pth", "md5": "e9f71df3bd8a678919a89db16b01dcfa", "group": "params",
        "trained_on": "GoPro", "arch": "bsst",
        # further checkpoints, given after the first one to dbis.convert in this order (group None: the whole file)
        "extra": [{
            "file": "raft-things.pth", "md5": "55b58de5d9022eb37893916d246e14a3", "group": None,
            "code": "https://github.com/princeton-vl/RAFT", "license": "BSD-3-Clause, Copyright (c) 2020, princeton-vl",
            "url": "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip (models/raft-things.pth; the "
                   "authors' code loads it from model_zoos/, see the BSSTNet folder above)",
        }],
    },
}

# every model: its source, the class and arguments in dbis/<arch>.py, the weight file and how it is written
MODELS = {
    "dbis-NAFNet-64": {"source": "NAFNet-64", "cls": "NAFNet", "args": {"tlc": "exact"},
                       "file": "dbis-NAFNet-64.safetensors", "store": "float32",
                       "sha256": "731f4a9d79703fd1f38e258582b67e8818f51fe0176acc64ac8acba0fef4fd61"},
    "dbisf-NAFNet-64": {"source": "NAFNet-64", "cls": "NAFNet", "args": {"tlc": "fast"},
                        "file": "dbisf-NAFNet-64.wpk", "store": "qc48",
                        "sha256": "a62c8f19a5467009f68054db8990608594717bb55b10d6b721ee88b685e36914"},
    # patch_batch=1 gives the authors' outputs bit for bit (dbis). dbisf: 14 patches per call (2 BSST calls per 720p
    # window), 24-frame windows (net.restore(clip) cuts a clip into them), flows reused between overlapping windows, TF32
    # and fused attention, failed patches redone; net.compile_blocks() compiles the heavy blocks (1.3x)
    "dbis-BSSTNet": {"source": "BSSTNet", "cls": "BSSTNet", "args": {"patch_batch": 1},
                     "file": "dbis-BSSTNet.safetensors", "store": "float32",
                     "sha256": "cc35ec20c837463413060557cc8671c614ce9e2a8402465daff56c82c45a928a"},
    "dbisf-BSSTNet": {"source": "BSSTNet", "cls": "BSSTNet",
                      "args": {"patch_batch": 14, "stage_batch": 4, "nan_fallback": True, "window": 24, "flow_cache": True,
                               "tf32": True, "sdpa": True},
                      "file": "dbisf-BSSTNet.wpk", "store": "qc48",
                      "sha256": "ac0db58978e14d60746a9ec89d272beb13922f2629f5eccf225c94d312de4fa8"},
}


def _module(name):
    return importlib.import_module(f".{SOURCES[MODELS[name]['source']]['arch']}", __name__)


def build(name, device="meta"):
    """The network of a model without weights (on the meta device by default: no memory, no random init)."""
    import torch
    m = MODELS[name]
    with torch.device(device):
        return getattr(_module(name), m["cls"])(**m["args"])


def load(name, weights_dir=None, device="cuda"):
    """The network of a model with its weights, in eval mode, on `device`."""
    import numpy as np
    import torch
    m = MODELS[name]
    path = Path(weights_dir or WEIGHTS) / m["file"]
    if not path.exists():
        src = SOURCES[m["source"]]
        files = [src] + src.get("extra", [])
        raise FileNotFoundError(f"{path} is missing: download " + ", ".join(f"{s['file']} ({s['url']})" for s in files)
                                + f" and run python -m dbis.convert {m['source']} " + " ".join(s["file"] for s in files))
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(path), device=str(device))
    else:
        from . import wpack
        sd = {k: torch.from_numpy(v if v.flags.writeable else np.array(v)).to(device)
              for k, v in wpack.unpack(str(path), threads=8).items()}
    net = build(name)
    net.load_state_dict(sd, strict=True, assign=True)
    return net.eval()
