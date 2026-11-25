"""dbis and dbisf: methods of the benchmark re-implemented in this repository (see DBIS.md).

    dbis-<method>    our code and weight file, the same outputs as the authors' code (largest difference of an output
                     value about 1e-5, the same mean PSNR to the fourth decimal on the test set)
    dbisf-<method>   our fast and light version: shorter run time and a smaller weight file, within 0.01 dB of the
                     authors' mean PSNR on every test set measured so far

The weight files are made locally from the authors' checkpoint:

    python -m dbis.convert NAFNet-64 NAFNet-GoPro-width64.pth     # -> weights/dbis/dbis-NAFNet-64.safetensors, dbisf-NAFNet-64.wpk

    import dbis
    net = dbis.load("dbisf-NAFNet-64")                            # eval mode, on the GPU
    y = net(x)                                                    # x: [B, 3, H, W] in [0, 1]

Dependencies: torch and numpy; safetensors for dbis files, zstandard for dbisf files (the wpack codec, dbis/wpack.py).
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
}

# every model: its source, the class and arguments in dbis/<arch>.py, the weight file and how it is written
MODELS = {
    "dbis-NAFNet-64": {"source": "NAFNet-64", "cls": "NAFNet", "args": {"tlc": "exact"},
                       "file": "dbis-NAFNet-64.safetensors", "store": "float32",
                       "sha256": "731f4a9d79703fd1f38e258582b67e8818f51fe0176acc64ac8acba0fef4fd61"},
    "dbisf-NAFNet-64": {"source": "NAFNet-64", "cls": "NAFNet", "args": {"tlc": "fast"},
                        "file": "dbisf-NAFNet-64.wpk", "store": "q128",
                        "sha256": "a64e13e8997331b4a1875ac96a607d79271f3fd4396b2eef209068ba933c095e"},
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
        raise FileNotFoundError(f"{path} is missing: download {src['file']} ({src['url']}) and run "
                                f"python -m dbis.convert {m['source']} {src['file']}")
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
