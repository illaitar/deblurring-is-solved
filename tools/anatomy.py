"""The anatomy of the dbis networks for the site's section "Anatomy of the networks": every block of a network, in the
order its forward pass runs them, with its parameters, its FLOPs per output frame and the scale it works at.

    python tools/anatomy.py results/dbis/anatomy.json [MODEL ...] [--frames 8] [--real DIR]

A cell is a block (the classes of net.train_checkpoint), a part the network keeps fixed (net.train_frozen, e.g.
BSSTNet's RAFT, as one cell) or a layer with weights of its own outside both (the convolutions that change the scale,
the first and last layers). FLOPs: torch.utils.flop_counter on the meta device (no memory, no time; FLOPs = 2 x
multiply-adds), on one 1280x720 frame, the frame padded as the network pads it. Video networks: clip networks on a clip
of --frames frames, divided by the number of frames; Turtle on its second frame, with the history of the first. The
scale of a cell: the largest output height of the network's cells over the cell's output height, as a power of two.
BSSTNet's sparse attention picks its tokens from the blur of the frames, so its count needs real frames and weights:
--real DIR (a folder of blurry 1280x720 frames, e.g. a GoPro test sequence) runs it on the GPU with dbis.load. Models
not named keep their entry of OUT.
"""
import contextlib
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import dbis  # noqa: E402

H, W = 720, 1280
_autocast = torch.autocast                               # autocast has no meta device: the precision does not change FLOPs
torch.autocast = lambda device_type, *a, **k: (contextlib.nullcontext() if device_type == "meta"
                                               else _autocast(device_type, *a, **k))
DEFAULT = ["dbis-NAFNet-64", "dbis-Concertormer", "dbis-Concertormer-lite", "dbis-Turtle", "dbis-DSTNetPlus-L",
           "dbis-BSSTNet"]


REAL = {"dbis-BSSTNet"}                                    # FLOPs depend on the frames (blur-aware sparse attention)
# the cells of a network: the classes of net.train_checkpoint, unless named here (DSTNet+ checkpoints whole stacks)
BLOCKS = {"dbis-DSTNetPlus-L": ("ResidualBlockNoBN", "ResBlock", "PAGF")}


def clip_of(folder, frames, device):
    import numpy as np
    from PIL import Image
    files = sorted(Path(folder).glob("*.png"))[:frames]
    return torch.stack([torch.from_numpy(np.array(Image.open(f).convert("RGB"))).permute(2, 0, 1) for f in files]
                       )[None].to(device).float().div(255)


def runner(net, name, frames, real=None):
    """A call of the network's forward pass (on meta tensors, or real frames), and the number of output frames."""
    if real is not None:
        clip = clip_of(real, frames, "cuda")
        return lambda: net(clip), clip.shape[1]
    x = torch.zeros(1, 3, H, W, device="meta")
    if net.kind == "image":
        if hasattr(net, "prepare"):
            net.prepare(H, W, torch.device("meta"))
        return lambda: net(x), 1
    if hasattr(net, "step"):                               # a causal network with a history (Turtle)
        mod = sys.modules[type(net).__module__]
        if hasattr(mod, "local_mask"):
            mod.local_mask(-(-H // 32) * 2, -(-W // 32) * 2, 4, x.device)
        _, k, v = net(x)                                 # the first frame fills the history
        return lambda: net(x, k, v), 1
    clip = torch.zeros(1, frames, 3, H, W, device="meta")
    return lambda: net(clip), frames


def anatomy(name, frames, real=None):
    real = real if name in REAL else None
    if name in REAL and real is None:
        raise SystemExit(f"{name}: its FLOPs depend on the frames; give --real DIR")
    net = dbis.load(name) if real else dbis.build(name, "meta").eval()
    blocks = set(BLOCKS.get(name) or getattr(net, "train_checkpoint", ()))
    frozen = tuple(f.rstrip(".") for f in getattr(net, "train_frozen", ()))     # module names ("raft")
    mods = dict(net.named_modules())
    whole = {n for n, m in mods.items() if n and (type(m).__name__ in blocks or n in frozen)}
    inside = {k for n in whole for k in mods if k.startswith(n + ".")}
    cells = [n for n, m in mods.items() if n and n not in inside and (
        n in whole or any(True for _ in m.parameters(recurse=False)))]
    order, height = {}, {}

    def pre(n):
        def f(m, a):                                     # returns None: a value would replace the input
            order.setdefault(n, len(order))
        return f

    def post(n):
        def f(m, a, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if torch.is_tensor(t) and t.dim() >= 4:          # images (B, C, H, W) or clips (B, T, C, H, W), not tokens
                height[n] = max(height.get(n, 0), t.shape[-2])
        return f

    fwd, n_out = runner(net, name, frames, real)
    hooks = [mods[n].register_forward_pre_hook(pre(n)) for n in cells] + \
            [mods[n].register_forward_hook(post(n)) for n in cells]
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fwd()
    for h in hooks:
        h.remove()
    counts = fc.get_flop_counts()
    root = type(net).__name__
    flops = {k[len(root) + 1:]: sum(v.values()) for k, v in counts.items() if k.startswith(root + ".")}
    total = sum(counts.get(root, counts.get("Global", {})).values()) / n_out
    ref = max(height.values())
    out, seen, level = [], set(), 0
    for n in sorted((c for c in cells if c in order), key=order.get):
        m, h = mods[n], height.get(n)
        # no image output (tokens, a pooled vector): the level of the cell before
        level = max(0, round(math.log2(ref / h))) if h and h * 32 >= ref else level
        ps = [p for p in m.parameters() if id(p) not in seen]       # a module shared by two cells counts once
        seen |= {id(p) for p in ps}
        block = 1 if (type(m).__name__ in blocks or n in frozen) else 0
        group = ".".join(n.split(".")[:2])
        cell = [n, type(m).__name__ if n not in frozen else f"{type(m).__name__} (fixed)", block, level, None,
                sum(p.numel() for p in ps), flops.get(n, 0) / n_out / 1e9, group]
        if not block and out and not out[-1][2] and out[-1][3] == level and out[-1][-1] == group:
            # layers in a row at one scale, in one part of the network: one cell
            prev = out[-1]
            prev[0] = prev[0].split(" .. ")[0] + " .. " + n
            prev[1] = "layers"
            prev[5] += cell[5]
            prev[6] += cell[6]
            continue
        out.append(cell)
    for c in out:
        c[6] = round(c[6], 3)
        del c[7]
    src = dbis.SOURCES[dbis.MODELS[name]["source"]]
    cellp = sum(c[5] for c in out)
    allp = sum(p.numel() for p in net.parameters())
    print(f"{name}: {allp / 1e6:.2f} M params ({cellp / allp:.1%} in the cells), {total / 1e9:.0f} GFLOPs per frame "
          f"({sum(c[6] for c in out) * 1e9 / max(1, total):.1%} in the cells), {len(out)} cells, "
          f"{sum(c[2] for c in out)} blocks, levels {sorted({c[3] for c in out})}", flush=True)
    return {"model": name, "source": dbis.MODELS[name]["source"], "method": src["method"], "venue": src["venue"],
            "video": net.kind != "image", "params": allp, "gflops": round(total / 1e9, 1),
            "frames": n_out, "cells": out}


if __name__ == "__main__":
    args = sys.argv[1:]
    frames = int(args[args.index("--frames") + 1]) if "--frames" in args else 8
    real = args[args.index("--real") + 1] if "--real" in args else None
    pos = [a for i, a in enumerate(args)
           if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--frames", "--real"))]
    names = pos[1:] or DEFAULT
    out = Path(pos[0])
    have = {m["model"]: m for m in json.loads(out.read_text(encoding="utf-8"))["models"]} if out.exists() else {}
    for n in names:
        have[n] = anatomy(n, frames, real)
    res = {"frame": [H, W], "cell": ["name", "type", "block", "level", "channels", "params", "gflops"],
           "models": [have[n] for n in DEFAULT + sorted(set(have) - set(DEFAULT)) if n in have]}
    out.write_text(json.dumps(res, separators=(",", ":")) + "\n", encoding="utf-8")
