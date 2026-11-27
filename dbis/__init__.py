"""dbis and dbisf: methods of the benchmark re-implemented in this repository (see DBIS.md).

    dbis-<method>    our code and weight file, the same outputs as the authors' code (largest difference of an output
                     value about 1e-5, the same mean PSNR to the fourth decimal on the test set)
    dbisf-<method>   our fast and light version: shorter run time and a smaller weight file, at most 0.05 dB below the
                     authors' mean PSNR on every test set of the benchmark

The weight files are hosted at https://huggingface.co/Illaitar/dbis: dbis.load fetches a missing file into WEIGHTS on
first use and checks its sha256 (dbis.download). WEIGHTS is $DBIS_WEIGHTS if set, else weights/dbis/ of the repository
checkout, else ~/.cache/dbis (the package installed with pip); $HF_ENDPOINT selects a mirror of Hugging Face. The files
can also be made locally from the authors' checkpoint:

    python -m dbis.convert NAFNet-64 NAFNet-GoPro-width64.pth     # -> weights/dbis/dbis-NAFNet-64.safetensors, dbisf-NAFNet-64.wpk
    python -m dbis.convert BSSTNet BSST_gopro.pth raft-things.pth # -> dbis-BSSTNet.safetensors, dbisf-BSSTNet.wpk (BSST + RAFT)
    python -m dbis.convert DSTNetPlus-L DSTNetPlus_L_gopro.pth    # -> dbis-DSTNetPlus-L.safetensors, dbisf-DSTNetPlus-L.wpk
    python -m dbis.convert Turtle GoPro_Deblur.pth                # -> dbis-Turtle.safetensors, dbisf-Turtle.wpk

    import dbis
    net = dbis.load("dbisf-NAFNet-64")                            # eval mode, on the GPU
    out = net.restore(frames)                                     # the same call for every model (dbis/restore.py):
    out = net.restore(frames, tile=512, precision="fp16")         # frames [T, 3, H, W], [B, T, 3, H, W] or [3, H, W]
    enc = dbis.load_encoder("dbisf-NAFNet-64")                    # the encoder alone: enc(x) -> list of feature maps,
    print(enc.feature_info)                                       # with their channels and reduction

Every method has the same layout, dbis/<method>/: net.py (the full network and fold), encoder.py (its encoder, which
the full network subclasses, so the encoder's weights keep their names), blocks/ (the building blocks).

Dependencies: torch and numpy (BSSTNet: torchvision too); safetensors for dbis files, zstandard for dbisf files (the
wpack codec, dbis/wpack.py).
"""
import importlib
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
WEIGHTS = Path(os.environ.get("DBIS_WEIGHTS") or (_REPO / "weights" / "dbis" if (_REPO / "registry").is_dir()
                                                  else Path.home() / ".cache" / "dbis"))
HF_REPO = "Illaitar/dbis"                                      # the hosted weight files (tools/hf_upload_dbis.py)
URL = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/") + f"/{HF_REPO}/resolve/main/"

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
    "DSTNetPlus-L": {
        "method": "DSTNet+ L", "paper": "Learning Efficient Deep Discriminative Spatial and Temporal Networks for Video "
                                         "Deblurring", "venue": "TPAMI 2025",
        "code": "https://github.com/sunny2109/DSTNet-plus", "commit": "54363c15d8b924aa1ae56b8f835c1f0289954e95",
        "license": "Apache License 2.0 (LICENSE.txt of the repository)",
        "url": "https://github.com/sunny2109/DSTNet-plus/releases/tag/v0.1.0 (also https://huggingface.co/Meloo/DSTNetPlus)",
        "file": "DSTNetPlus_L_gopro.pth", "md5": "d7da9b176f72cd2c10c6c146fe481e65", "group": "params",
        "trained_on": "GoPro", "arch": "dstnetplus",
    },
    "Turtle": {
        "method": "Turtle", "paper": "Learning Truncated Causal History Model for Video Restoration",
        "venue": "NeurIPS 2024",
        "code": "https://github.com/Ascend-Research/Turtle", "commit": "7094f4221b64ad0962b4f27ff1b76d788836e804",
        "license": "MIT, Copyright (c) 2024 Huawei Technologies Co., Ltd.",
        "url": "https://github.com/Ascend-Research/Turtle (README: trained models, GoPro_Deblur.pth)",
        "file": "GoPro_Deblur.pth", "md5": "736df72180a69242b495b2463042b29d", "group": "params",
        "trained_on": "GoPro", "arch": "turtle",
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
    # the authors' code as the benchmark runs it: float32, clips of 50 frames, 10 frames at a time through the
    # full-resolution layers (DSTNetPlus defaults)
    "dbis-DSTNetPlus-L": {"source": "DSTNetPlus-L", "cls": "DSTNetPlus", "args": {},
                          "file": "dbis-DSTNetPlus-L.safetensors", "store": "float32",
                          "sha256": "378a5fca3c72b17467ab5d9a39d7c43291b513891f9faa44f86aeb489a5bc3cd"},
    # dbisf: float16 autocast (the authors' CuPy kernel had none), the dynamic conv with plain addcmul, weights rounded per
    # channel to std / 128 (std / 48 costs 0.03 dB on GoPro here); clips stay 50 frames (27 already costs 0.04-0.07 dB).
    # 26.11: the encoder folded into two convs, the propagation in channels_last, the dynamic conv autotuned when compiled
    "dbisf-DSTNetPlus-L": {"source": "DSTNetPlus-L", "cls": "DSTNetPlus",
                           "args": {"precision": "fp16", "exact_dynconv": False, "fold_encoder": True,
                                    "prop_channels_last": True, "autotune": True},
                           "file": "dbisf-DSTNetPlus-L.wpk", "store": "qc128",
                           "sha256": "42a6934bce12fd88d30a1fe33de480070abb7d8311333a96f321f1de1a32cf1b"},
    # the authors' tiled inference as the benchmark runs it: tiles of 320 px with 192 px overlap, one tile per call,
    # each with its own history, float32
    "dbis-Turtle": {"source": "Turtle", "cls": "Turtle", "args": {},
                    "file": "dbis-Turtle.safetensors", "store": "float32",
                    "sha256": "3d80bf02e2f4ff6e949ca4db309fc39a2e6bb51143dc8ee4deb9317153a7b61d"},
    # dbisf: float16 autocast, all tiles of a frame in one call, SAB's two masks applied together, 15 tiles with 80 px
    # overlap at 1280x720 instead of 45 with 192, weights rounded per channel to std / 128 (std / 48: -0.05 dB on GoPro)
    "dbisf-Turtle": {"source": "Turtle", "cls": "Turtle",
                     "args": {"fast": True, "precision": "fp16", "batch": 15, "tile_overlap": 80},
                     "file": "dbisf-Turtle.wpk", "store": "qc128",
                     "sha256": "30c8bea074fa9fb53300fee15c8d16e920eff6c4b3148932e72294c5cb7371b6"},
}


def _module(name):
    return importlib.import_module(f".{SOURCES[MODELS[name]['source']]['arch']}", __name__)


def build(name, device="meta"):
    """The network of a model without weights (on the meta device by default: no memory, no random init)."""
    import torch
    m = MODELS[name]
    with torch.device(device):
        return getattr(_module(name), m["cls"])(**m["args"])


def download(name, weights_dir=None):
    """Fetch a model's weight file from URL into `weights_dir` (weights/dbis/ by default) and check its sha256 against
    the recorded one; returns its path. The file is written under a .part name and renamed only once it checks out."""
    import hashlib
    import urllib.request
    m = MODELS[name]
    d = Path(weights_dir or WEIGHTS)
    d.mkdir(parents=True, exist_ok=True)
    path = d / m["file"]
    part = path.with_name(path.name + ".part")
    h = hashlib.sha256()
    print(f"dbis: fetching {URL}{m['file']} -> {path}", flush=True)
    with urllib.request.urlopen(URL + m["file"], timeout=60) as r, open(part, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 22), b""):
            h.update(chunk)
            f.write(chunk)
    if h.hexdigest() != m["sha256"]:
        part.unlink()
        raise OSError(f"{URL}{m['file']}: sha256 {h.hexdigest()} differs from the recorded {m['sha256']}")
    part.replace(path)
    return path


def _weights(name, weights_dir, device, fetch=True):
    """The state dict of a model's weight file on `device` (fetched with download() if missing and fetch is True)."""
    import numpy as np
    import torch
    m = MODELS[name]
    path = Path(weights_dir or WEIGHTS) / m["file"]
    if not path.exists() and fetch:
        path = download(name, weights_dir)
    if not path.exists():
        src = SOURCES[m["source"]]
        files = [src] + src.get("extra", [])
        raise FileNotFoundError(f"{path} is missing: dbis.download('{name}') fetches it, or download "
                                + ", ".join(f"{s['file']} ({s['url']})" for s in files)
                                + f" and run python -m dbis.convert {m['source']} " + " ".join(s["file"] for s in files))
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(path), device=str(device))
    else:
        from . import wpack
        sd = {k: torch.from_numpy(v if v.flags.writeable else np.array(v)).to(device)
              for k, v in wpack.unpack(str(path), threads=8).items()}
    return sd


def load(name, weights_dir=None, device="cuda", fetch=True):
    """The network of a model with its weights, in eval mode, on `device` (a missing weight file is fetched from URL
    unless fetch is False)."""
    net = build(name)
    net.load_state_dict(_weights(name, weights_dir, device, fetch), strict=True, assign=True)
    return net.eval()


def load_encoder(name, weights_dir=None, device="cuda", fetch=True):
    """The encoder of a model alone (dbis/<method>/encoder.py) with its weights from the model's weight file, in eval
    mode, on `device`: enc(x) -> a list of feature maps, enc.feature_info their channels and reduction."""
    import torch
    m, mod = MODELS[name], _module(name)
    with torch.device("meta"):
        enc = getattr(mod, mod.ENCODER)(**m["args"])
    pre, keys = mod.ENCODER_PREFIX, set(enc.state_dict())
    sd = {k[len(pre):]: v for k, v in _weights(name, weights_dir, device, fetch).items()
          if k.startswith(pre) and k[len(pre):] in keys}
    enc.load_state_dict(sd, strict=True, assign=True)
    return enc.eval()
