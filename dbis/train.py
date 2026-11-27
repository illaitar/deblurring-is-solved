"""dbis.train: one training script for every model of dbis, run with a plan fitted to the machine.

    python -m dbis.train dbisf-NAFNet-64 --plan --data DIR --out runs/naf      # once: the settings for this machine
    python -m dbis.train dbisf-NAFNet-64 --data DIR --out runs/naf              # train with them (runs/naf/plan.json)
    torchrun --nproc-per-node 8 -m dbis.train dbisf-NAFNet-64 --data DIR --out runs/naf
    python -m dbis.train dbisf-NAFNet-64 --sweep                                # our table of settings (train_configs.json)

Data: dbis/data.py. The layout of DIR is found by itself (GoPro, REDS, BSD, DVD, RealBlur, GS-Blur, MIORe as
downloaded); the frames of one folder are one sequence; --list / --exclude keep or drop frames and sequences by name.

Recipe: RECIPES[source] is the authors' GoPro recipe of the model (crop, frames per clip, global batch, steps, AdamW,
warmup, lr multipliers, loss, gradient clip; DBIS.md, "Training"). The same for all models: a cosine decay to 1e-7, one
set of augmentations (a random crop at the same place in every frame and both frames of a pair, then one of the 8 flips
and transposes; --reverse also plays clips backwards), and the loss from dbis/losses.py (--loss changes it).
--init released (default) fine-tunes the released weights (FINETUNE of the steps and lr), --init scratch starts anew.

Plan (--plan, once per machine and model; written to OUT/plan.json, which training follows):
  precision   bfloat16 where the GPU has it, else float16 with a loss scaler (--precision overrides)
  batch       the largest batch per GPU that fits, found with real training steps (forward, backward, AdamW step),
              starting from our measured table; with and without activation checkpointing (the blocks of
              net.train_checkpoint recomputed in the backward pass), keeping the one with more samples per second
  layout      image models: channels_last is timed and kept if it is faster
  compile     torch.compile of the training step (forward and backward) is timed and kept if it is faster (NAFNet-64:
              3.45x, 79 s to compile); --no-compile-probe skips it
  data        every frame reader of data.py is timed on DIR, and the data workers per GPU are set so that loading keeps
              up with the GPU; when decoding cannot keep up and --cache DIR is given, the frames are decoded once into
              that cache and read from it
--echo K reuses every clip read for K batches (new crops and flips each time) when reading cannot keep up;
--reader ram holds every frame in memory; --reader gpu holds them on the GPU and cuts the batches there.
The global batch stays the recipe's (gradient accumulation), so a plan changes the speed, not the training.

Options that change the training (off by default; DBIS.md, "Training", has what each did on NAFNet-64):
  --ema D1,D2   averages of the weights (OUT/ema_<D>.pt; --export-ema exports the first)
  --noise-every N, --adapt-batch MIN,MAX   the gradient noise scale (McCandlish et al. 2018) measured every N steps,
                and the batch raised with it: the noise scale grows during training, and a batch far above it spends
                samples that no step needs
  --shuffle perm, --rho K   random reshuffling of the clips; RHO-LOSS selection of K x batch candidates

Sweep (--sweep): the same batch search on our A100 for the recipe's crop and frames and for smaller ones, in bfloat16,
float16 and float32, with and without checkpointing: fixed and per-sample memory and the time per sample, written to
train_configs.json. A plan starts from it.

Output in --out: plan.json, last.pt (a stopped run resumes from it), log.tsv, best.pt (with --val), and at the end the
weight files of the model in the stores of dbis.MODELS, which dbis.load(name, weights_dir=OUT) reads.
"""
import argparse
import contextlib
import functools
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import MODELS, build, load
from .data import Reader, available, best_reader, build_cache, cache_ready, find_pairs, names_in, select
from .losses import make as make_loss

CONFIGS = Path(__file__).with_name("train_configs.json")

# The authors' GoPro recipes (their training configs at the commits of dbis.SOURCES): crop, frames per clip, global
# batch, steps, AdamW (lr, betas, weight decay), warmup steps, lr multipliers by parameter name, loss, gradient clip.
# Cosine decay to 1e-7 in all four. Not in every original: bfloat16 (NAFNet and DSTNet+ trained in float32, BSSTNet and
# Turtle in float16 with a scaler) and the one set of augmentations (Turtle kept the frame unchanged half of the time).
RECIPES = {
    # NAFNet options/train/GoPro/NAFNet-width64.yml: 8 per GPU x 8 GPUs, 400k, PSNR loss, clip 0.01 (basicsr default)
    "NAFNet-64": {"crop": 256, "frames": 1, "batch": 64, "steps": 400_000, "lr": 1e-3, "betas": (0.9, 0.9),
                  "weight_decay": 1e-3, "warmup": 0, "lr_mult": {}, "loss": "psnr_naf", "clip": 0.01},
    # BSSTNet options/train/BSST/gopro_BSST.yml + Video_BSST_model.py: global batch 8 (the paper), L1 over the 24
    # frames, RAFT frozen (train_frozen), per-group lr of the flow completion, offsets and soft split / composition
    "BSSTNet": {"crop": 256, "frames": 24, "batch": 8, "steps": 600_000, "lr": 4e-4, "betas": (0.9, 0.99),
                "weight_decay": 0.0, "warmup": 1000,
                "lr_mult": {"blur_motion_refine": 0.1, "conv_offset": 0.25, "softsplit": 0.5, "softcomp": 0.5},
                "loss": "l1_mean", "clip": 0.01},
    # DSTNet-plus options/train/DSTNetPlus/DSTPlus_large_GoPro.yml: 1 per GPU x 8, Charbonnier + 0.1 x |rfft2 difference|
    "DSTNetPlus-L": {"crop": 256, "frames": 15, "batch": 8, "steps": 600_000, "lr": 2e-4, "betas": (0.9, 0.999),
                     "weight_decay": 1e-4, "warmup": 0, "lr_mult": {}, "loss": "charb_e_1e-12+0.1*fft_cmod_rfft", "clip": 0.01},
    # Turtle options/Turtle_Deblur_Gopro.yml + video_restoration_model.py: 2 per GPU x 8, L1 per frame, no clip; the
    # cache carried through the 5-frame clip without detaching (train_forward)
    "Turtle": {"crop": 192, "frames": 5, "batch": 16, "steps": 200_000, "lr": 4e-4, "betas": (0.9, 0.99),
               "weight_decay": 0.0, "warmup": 0, "lr_mult": {}, "loss": "l1_mean", "clip": None},
    # Concertormer options/train/GoPro/GoPro.yml: progressive training (steps, crop, global batch = per GPU x 8 GPUs),
    # AdamW, TrueCosineAnnealingLR over all 600k, MSL1FFTLoss on the four outputs (targets resized bilinearly)
    "Concertormer": {"stages": [(200_000, 128, 64), (300_000, 256, 16), (100_000, 320, 8)], "frames": 1,
                     "lr": 1e-3, "betas": (0.9, 0.9), "weight_decay": 1e-3, "warmup": 0, "lr_mult": {},
                     "loss": "l1_mean+0.1*fft_cmod_rfft", "multiscale": True, "clip": 0.01},
    # options/train/GoPro/GoPro_lite.yml
    "Concertormer-lite": {"stages": [(400_000, 256, 32), (100_000, 320, 16), (100_000, 384, 8)], "frames": 1,
                          "lr": 1e-3, "betas": (0.9, 0.9), "weight_decay": 1e-3, "warmup": 0, "lr_mult": {},
                          "loss": "l1_mean+0.1*fft_cmod_rfft", "multiscale": True, "clip": 0.01},
}
for _r in RECIPES.values():                    # a single-stage recipe as one stage
    _r.setdefault("stages", [(_r.get("steps"), _r.get("crop"), _r.get("batch"))])
    _r["steps"], _r["crop"], _r["batch"] = sum(s[0] for s in _r["stages"]), _r["stages"][-1][1], _r["stages"][-1][2]
LR_MIN = 1e-7
FINETUNE = {"steps": 0.1, "lr": 0.1}       # a fine-tune from released weights: 10 % of the steps at 10 % of the lr
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


# ---------------------------------------------------------------------------------------------------- samples
class Clips(torch.utils.data.Dataset):
    """Sample i: a random clip of `frames` consecutive pairs of one sequence (one pair for image models), cropped and
    augmented, as two uint8 tensors (T, 3, crop, crop). Every sample is a function of (seed, i) alone.

    echo k > 1 (data echoing): every clip read feeds k batches, each time with a new crop and flips. Slot j of the m-th
    batch a data worker loads takes the clip of slot j of the worker's group m // k, so the frames a worker decoded are
    in its memory for its next k batches, and one batch never holds two crops of one clip. The layout follows the
    DataLoader: batch q of the stream (from `origin`, in samples of this rank) goes to worker q % workers."""

    def __init__(self, seqs, frames, crop, reader, aug=True, reverse=False, seed=0, echo=1, batch=1, workers=0,
                 rank=0, world=1, origin=0):
        self.items = [(v, i) for v in seqs.values() for i in range(len(v) - frames + 1)]
        if not self.items:
            raise ValueError(f"no sequence has {frames} frames")
        self.frames, self.crop, self.reader, self.aug, self.seed = frames, crop, reader, aug, seed
        self.reverse = reverse and frames > 1
        self.echo, self.batch, self.workers = max(1, echo), batch, max(1, workers)
        self.rank, self.world, self.origin = rank, world, origin
        self.held = (None, {})

    def __len__(self):
        return len(self.items)

    def clip(self, i):
        """The pairs of sample i, read (echo 1) or taken from the clips this worker holds (echo k)."""
        if self.echo == 1:
            return None
        q, j = divmod((i - self.rank) // self.world - self.origin, self.batch)
        w, m = q % self.workers, q // self.workers
        group = (w, m // self.echo)
        pick = random.Random(f"{self.seed} {self.rank} {group[0]} {group[1]} {j}")
        v, s = self.items[pick.randrange(len(self.items))]
        if self.held[0] != group:
            self.held = (group, {})
        if j not in self.held[1]:
            self.held[1][j] = [(self.reader(b), self.reader(g)) for b, g in v[s:s + self.frames]]
        return self.held[1][j]

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)
        pairs = self.clip(i)
        if pairs is None:
            v, s = self.items[rng.randrange(len(self.items))]
            pairs = ((self.reader(b), self.reader(g)) for b, g in v[s:s + self.frames])
        c, blur, sharp = self.crop, [], []
        for k, (x, y) in enumerate(pairs):
            if k == 0:
                h, w = x.shape[:2]
                if h < c or w < c:
                    raise ValueError(f"a frame of {h} x {w} is smaller than the crop {c}")
                r0, c0 = rng.randrange(h - c + 1), rng.randrange(w - c + 1)
            blur.append(x[r0:r0 + c, c0:c0 + c])
            sharp.append(y[r0:r0 + c, c0:c0 + c])
        blur, sharp = np.stack(blur), np.stack(sharp)
        if self.aug:
            f = [rng.random() < 0.5 for _ in range(4)]

            def aug(a):
                a = a[:, :, ::-1] if f[0] else a
                a = a[:, ::-1] if f[1] else a
                a = a.transpose(0, 2, 1, 3) if f[2] else a
                return a[::-1] if f[3] and self.reverse else a
            blur, sharp = aug(blur), aug(sharp)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2)   # noqa: E731
        return t(blur), t(sharp)


class GpuFrames:
    """--reader gpu: every frame of the training set in GPU memory (uint8) and the batches cut there, with no data
    workers. The same sampling as Clips (a random clip, one crop at the same place in all its frames and both frames of
    a pair, one of the 8 flips and transposes, clips backwards with reverse), drawn from a generator seeded by (seed,
    rank, batch): the crop offsets, flips and transposes are folded into one gather index.
    shuffle "perm" (random reshuffling): the clips come from a new permutation of all clips every epoch instead of
    independent draws. add_reference() keeps a reference restoration of every frame (for --rho)."""

    def __init__(self, seqs, frames, reader, device, jobs=16):
        from concurrent.futures import ThreadPoolExecutor
        paths = [pr for v in seqs.values() for pr in v]
        first = np.asarray(reader(paths[0][0]))
        self.blur = torch.empty((len(paths), *first.shape), dtype=torch.uint8, device=device)
        self.sharp = torch.empty_like(self.blur)
        with ThreadPoolExecutor(jobs) as ex:
            for i, (x, y) in enumerate(ex.map(lambda p: (np.asarray(reader(p[0])), np.asarray(reader(p[1]))), paths)):
                if x.shape != first.shape or y.shape != first.shape:
                    raise SystemExit(f"--reader gpu needs frames of one size: {paths[i][0]} is {x.shape[:2]}, "
                                     f"the first {first.shape[:2]}")
                self.blur[i].copy_(torch.from_numpy(np.ascontiguousarray(x)))
                self.sharp[i].copy_(torch.from_numpy(np.ascontiguousarray(y)))
        starts, k = [], 0
        for v in seqs.values():
            starts += range(k, k + len(v) - frames + 1)
            k += len(v)
        if not starts:
            raise ValueError(f"no sequence has {frames} frames")
        self.starts, self.frames = torch.tensor(starts, device=device), frames
        self.gib = 2 * self.blur.numel() / 2 ** 30
        self.ref, self.perms = None, {}

    @torch.no_grad()
    def add_reference(self, net):
        """ref[i]: net's restoration of blurry frame i (whole frame, uint8), so that the loss of the reference on any
        crop is a lookup."""
        self.ref = torch.empty_like(self.blur)
        for i in range(len(self.blur)):
            self.ref[i] = net.restore(self.blur[i].permute(2, 0, 1), out="uint8").permute(1, 2, 0)
        self.gib += self.ref.numel() / 2 ** 30

    def cut(self, n, b, crop, aug=True, reverse=False, seed=0, rank=0, shuffle="random", pos=0, world=1):
        """Batch n of this rank: b clips (T, 3, crop, crop) of blurry, sharp (and reference) frames, uint8 on the GPU.
        pos: samples of this rank before this batch (for shuffle perm)."""
        dev, t_ = self.blur.device, self.frames
        h, w = self.blur.shape[1:3]
        if h < crop or w < crop:
            raise ValueError(f"frames of {h} x {w} are smaller than the crop {crop}")
        ar, at = torch.arange(crop, device=dev), torch.arange(t_, device=dev)
        g = torch.Generator(device=dev)
        g.manual_seed(((seed * 1_000_003 + rank) * 1_000_003 + n) % 2 ** 63)
        s = self.starts[torch.randint(len(self.starts), (b,), device=dev, generator=g)]
        if shuffle == "perm":                         # positions in the stream of all ranks -> epoch, place in it
            q = (pos + torch.arange(b, device=dev)) * world + rank
            m = len(self.starts)
            ep, k = q // m, q % m
            need = set(ep.tolist())
            self.perms = {e: p for e, p in self.perms.items() if e in need}
            pick = torch.empty(b, dtype=torch.long, device=dev)
            for e in need:
                if e not in self.perms:                # one permutation per epoch, the same on every rank
                    pg = torch.Generator(device=dev)
                    pg.manual_seed((seed * 1_000_003 + e) % 2 ** 63)
                    self.perms[e] = torch.randperm(m, device=dev, generator=pg)
                pick[ep == e] = self.perms[e][k[ep == e]]
            s = self.starts[pick]
        r0 = torch.randint(h - crop + 1, (b, 1), device=dev, generator=g)
        c0 = torch.randint(w - crop + 1, (b, 1), device=dev, generator=g)
        f = torch.rand(b, 4, device=dev, generator=g) < 0.5
        if not aug:
            f.zero_()
        if not (reverse and t_ > 1):
            f[:, 3] = False
        fr = s[:, None] + torch.where(f[:, 3:4], t_ - 1 - at, at)              # (b, T)
        rows = r0 + torch.where(f[:, 1:2], crop - 1 - ar, ar)                  # (b, crop): flip of H
        cols = c0 + torch.where(f[:, 0:1], crop - 1 - ar, ar)                  # flip of W
        tr = f[:, 2].view(b, 1, 1)                                             # then the transpose
        gr = torch.where(tr, rows[:, None, :], rows[:, :, None])
        gc = torch.where(tr, cols[:, :, None], cols[:, None, :])
        idx = (fr[:, :, None, None], gr[:, None], gc[:, None])                 # -> (b, T, crop, crop, 3)
        out = [a[idx].permute(0, 1, 4, 2, 3).contiguous() for a in (self.blur, self.sharp)]
        if self.ref is not None:
            out.append(self.ref[idx].permute(0, 1, 4, 2, 3).contiguous())
        return out

    def batches(self, crop, b, aug=True, reverse=False, seed=0, rank=0, start=0, shuffle="random", world=1):
        n = start
        while True:
            yield self.cut(n, b, crop, aug, reverse, seed, rank, shuffle, n * b, world)[:2]
            n += 1


class Stream(torch.utils.data.Sampler):
    """Sample indices of this rank: start * world + rank, then every world-th (resumable, disjoint over ranks)."""

    def __init__(self, start, rank, world):
        self.start, self.rank, self.world = start, rank, world

    def __iter__(self):
        i = self.start * self.world + self.rank
        while True:
            yield i
            i += self.world


# ---------------------------------------------------------------------------------------------------- the model
def freeze(net):
    """No gradients for the parts a model keeps fixed (net.train_frozen: module name prefixes); returns them."""
    frozen = tuple(getattr(net, "train_frozen", ()))
    for n, p in net.named_parameters():
        if n.startswith(frozen):
            p.requires_grad_(False)
    return frozen


def _ckpt(fwd, *a, **k):
    if torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(fwd, *a, use_reentrant=False, **k)
    return fwd(*a, **k)


def checkpointing(net, on):
    """Activation checkpointing of the blocks named in net.train_checkpoint (recomputed in the backward pass)."""
    names = set(getattr(net, "train_checkpoint", ()))
    for m in net.modules():
        if type(m).__name__ in names:
            if on and "forward" not in m.__dict__:
                m.forward = functools.partial(_ckpt, type(m).forward.__get__(m))
            elif not on and "forward" in m.__dict__:
                del m.forward
    return net


def forward(net, x, multi=False):
    """x (B, T, 3, H, W) -> prediction of the same shape (image models see (B, 3, H, W)); multi: the list of every
    output of a multi-output network (net.forward_all, coarse to fine), each (B, 1, 3, h, w)."""
    if multi:
        return [o[:, None] for o in net.forward_all(x[:, 0])]
    if net.kind == "image":
        return net.train_forward(x[:, 0])[:, None]
    return net.train_forward(x)


def loss_over(loss_of, pred, target):
    """The loss of one output, or the sum over the outputs of a multi-output network with the target resized
    bilinearly to each (the authors' MSL1FFTLoss of Concertormer)."""
    if not isinstance(pred, (list, tuple)):
        return loss_of(pred.float(), target)
    tot = 0.0
    for o in pred:
        y = target
        if o.shape[-2:] != target.shape[-2:]:
            y = F.interpolate(target.flatten(0, 1), size=o.shape[-2:], mode="bilinear").view(*target.shape[:3], *o.shape[-2:])
        tot = tot + loss_of(o.float(), y)
    return tot


class Step(torch.nn.Module):
    """forward() of the model as a module, so that DistributedDataParallel wraps (and synchronises) it."""

    def __init__(self, net, multi=False):
        super().__init__()
        self.net, self.multi = net, multi

    def forward(self, x):
        return forward(self.net, x, self.multi)


def optimizer(net, r, lr):
    groups = {}
    for n, p in net.named_parameters():
        if p.requires_grad:
            groups.setdefault(next((v for k, v in r["lr_mult"].items() if k in n), 1.0), []).append(p)
    fused = all(p.is_cuda for v in groups.values() for p in v)       # one kernel for the whole update (3.5 % on NAFNet)
    return torch.optim.AdamW([{"params": v, "mult": m} for m, v in groups.items()], lr=lr, betas=r["betas"],
                             weight_decay=r["weight_decay"], fused=fused)


# ---------------------------------------------------------------------------------------------------- measuring
def trial(net, opt, b, crop, frames, precision, device, steps=3, fn=None, cl=False):
    """Peak reserved memory (GiB) and seconds per step of `steps` training steps at batch b (after one warm-up step);
    None if it runs out of memory."""
    fn = fn if fn is not None else (lambda x: forward(net, x))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    try:
        for i in range(steps + 1):
            x = torch.rand(b, frames, 3, crop, crop, device=device)
            if cl and frames == 1:                                  # channels_last input (image models)
                x = x[:, 0].contiguous(memory_format=torch.channels_last).unsqueeze(1)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.autocast("cuda", DTYPES[precision] or torch.float32, enabled=precision != "fp32"):
                p = fn(x)
            loss = (p.float() - x).abs().mean()
            loss.backward()
            if not all(torch.isfinite(q.grad).all() for q in net.parameters() if q.grad is not None):
                raise FloatingPointError("non-finite gradients")      # a setting that trains wrongly is no option
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            if i:
                times.append(time.perf_counter() - t0)
            del x, p
        return torch.cuda.max_memory_reserved(device) / 2 ** 30, float(np.median(times))
    except (torch.cuda.OutOfMemoryError, FloatingPointError):
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return None


def table_guess(source, crop, frames, precision, ckpt, free_gib):
    """The batch our table predicts for this free memory (None when the setting was not measured)."""
    if not CONFIGS.exists():
        return None
    for e in json.loads(CONFIGS.read_text(encoding="utf-8")).get(source, []):
        if (e["crop"], e["frames"], e["precision"], e["checkpoint"]) == (crop, frames, precision, ckpt):
            return max(1, int((0.93 * free_gib - e["fixed_gib"]) // max(e["per_sample_gib"], 1e-3)))
    return None


def search_batch(net, r, source, crop, frames, precision, ckpt, device, cap):
    """The largest batch up to cap that runs, by trials (from the table's guess, then doubling / bisecting), and the
    successful trials {batch: (GiB, s per step)}."""
    checkpointing(net, ckpt)
    opt = optimizer(net, r, 1e-9)
    free = torch.cuda.mem_get_info(device)[0] / 2 ** 30
    b = max(1, min(cap, table_guess(source, crop, frames, precision, ckpt, free) or 1))
    seen, lo, hi = {}, 0, cap + 1              # lo runs, hi does not
    while True:
        if b not in seen:
            seen[b] = trial(net, opt, b, crop, frames, precision, device)
        if seen[b]:
            lo = max(lo, b)
        else:
            hi = min(hi, b)
        if hi - lo <= 1:
            break
        b = min(cap, lo * 2) if hi > cap and lo else (lo + hi) // 2
        if b in seen and (b == lo or b == hi):
            break
    del opt
    checkpointing(net, False)
    torch.cuda.empty_cache()
    return lo, {k: v for k, v in seen.items() if v}


def sweep(name, device, dest=None):
    """Our table: for the recipe's crop and frames and smaller ones, every precision, with and without checkpointing,
    the batch search up to 64, and a fit of the peak memory = fixed + per_sample * batch."""
    source = MODELS[name]["source"]
    r = RECIPES[source]
    net = build(name, device)
    freeze(net)
    net.train()
    crops = sorted({r["crop"]} | {c for c in (128, 192, 256) if c <= r["crop"]})
    crops = [c for c in crops if c in getattr(net, "train_crops", crops)]
    frames = sorted({r["frames"], max(2, r["frames"] // 2)}) if r["frames"] > 1 else [1]
    out = []
    for crop in crops:
        for fr in frames:
            for precision in ("bf16", "fp16", "fp32"):
                for ckpt in (False, True):
                    b, seen = search_batch(net, r, source, crop, fr, precision, ckpt, device, 64)
                    if not b:
                        print(f"{name} crop {crop} frames {fr} {precision} ckpt {ckpt}: does not fit", flush=True)
                        continue
                    bs = sorted(seen)
                    per = (seen[bs[-1]][0] - seen[bs[0]][0]) / (bs[-1] - bs[0]) if len(bs) > 1 else seen[b][0]
                    fixed = seen[bs[0]][0] - per * bs[0] if len(bs) > 1 else 0.0
                    e = {"crop": crop, "frames": fr, "precision": precision, "checkpoint": ckpt, "max_batch": b,
                         "fixed_gib": round(max(fixed, 0.0), 3), "per_sample_gib": round(per, 3),
                         "ms_per_sample": round(1000 * seen[b][1] / b, 2),
                         "trials": {str(k): [round(v[0], 2), round(1000 * v[1], 1)] for k, v in sorted(seen.items())}}
                    out.append(e)
                    print(f"{name} {json.dumps(e)}", flush=True)
    dest = Path(dest or CONFIGS)
    table = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}
    table[source] = out
    table.setdefault("_measured_on", {})[source] = {
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "date": time.strftime("%Y-%m-%d"),
        "memory_gib": round(torch.cuda.get_device_properties(device).total_memory / 2 ** 30, 1)}
    dest.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def stage_fit(net, r, source, crop, frames, want, precision, device, gpus):
    """Batch per GPU, checkpointing, samples per second and peak GiB for one stage (crop, global batch)."""
    per_gpu, found = -(-want // gpus), {}
    for ckpt in (False, True):
        b, seen = search_batch(net, r, source, crop, frames, precision, ckpt, device, per_gpu)
        if b:
            found[ckpt] = (b, b / seen[b][1], seen[b][0])
            print(f"plan: crop {crop}, checkpointing {ckpt}: batch {b} per GPU, {found[ckpt][1]:.2f} samples/s, "
                  f"{found[ckpt][2]:.1f} GiB", flush=True)
        if b >= per_gpu:
            break                                                  # it all fits: checkpointing cannot be faster
    if not found:
        raise SystemExit(f"even one sample of crop {crop}, {frames} frame(s) does not fit this GPU; "
                         "give a smaller --crop or --frames")
    ckpt = max(found, key=lambda k: found[k][1])
    b, sps, gib = found[ckpt]
    accum = -(-want // (b * gpus))
    return {"crop": crop, "global_batch": want, "batch_per_gpu": -(-want // (gpus * accum)), "accumulation": accum,
            "checkpoint": ckpt, "samples_per_s": round(sps * gpus, 2), "peak_gib": round(gib, 2)}


def make_plan(a, name, r, stages, frames, device, seqs, out):
    """The settings for this machine (see the module's docstring), written to OUT/plan.json."""
    source = MODELS[name]["source"]
    gpus = max(1, torch.cuda.device_count())
    precision = a.precision or ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
    net = build(name, device)
    freeze(net)
    net.train()
    fits = [dict(stage_fit(net, r, source, crop, frames, want, precision, device, gpus), steps=n)
            for n, crop, want in stages]
    f = fits[-1]                                                   # the probes run on the largest stage
    checkpointing(net, f["checkpoint"])
    if hasattr(net, "prepare"):
        net.prepare(f["crop"], f["crop"], device)
    base = f["batch_per_gpu"] * gpus / f["samples_per_s"]           # seconds per step as found
    cl, speedup = False, 1.0
    if net.kind == "image":                                        # channels_last (NAFNet: 6 % faster in eager)
        net.to(memory_format=torch.channels_last)
        res = trial(net, optimizer(net, r, 1e-9), f["batch_per_gpu"], f["crop"], frames, precision, device, cl=True)
        cl = bool(res) and res[1] < 0.97 * base
        speedup = base / res[1] if cl else 1.0
        print(f"plan: channels_last {'kept' if cl else 'not faster'}", flush=True)
        if not cl:
            net.to(memory_format=torch.contiguous_format)
    compiled, compile_s = False, None
    if not a.no_compile_probe:                                     # torch.compile of the step (NAFNet: 3.45x)
        for layout in ((True, False) if cl else (False,)):        # compile + bf16 + channels_last gave NaN gradients
            if not layout and cl:                                  # in NAFNet's bias grads (torch 2.10): retry without
                net.to(memory_format=torch.contiguous_format)
            torch._dynamo.reset()
            t0 = time.time()
            res = trial(net, optimizer(net, r, 1e-9), f["batch_per_gpu"], f["crop"], frames, precision, device, steps=5,
                        fn=torch.compile(Step(net)), cl=layout)
            compile_s = round(time.time() - t0)
            if res and base / res[1] > 1.05 * (speedup if layout == cl else 1.0):
                compiled, speedup, cl = True, base / res[1], layout
                break
            print(f"plan: torch.compile with channels_last {layout}: {'not faster' if res else 'failed (out of memory or non-finite gradients)'}",
                  flush=True)
        print(f"plan: torch.compile {'kept' if compiled else 'not used'} (channels_last {cl}; {compile_s} s to compile and time)",
              flush=True)
    for g in fits:                                                 # the data loading must keep up with the faster steps
        g["samples_per_s"] = round(g["samples_per_s"] * speedup, 2)
    checkpointing(net, False)
    del net
    torch.cuda.empty_cache()
    # data: pairs per second of each reader at the largest crop, workers that keep up with the fastest stage, the
    # cache when decoding cannot
    need = max(f["samples_per_s"] for f in fits) / gpus * frames * 1.3     # pairs per second per GPU, with a margin
    crop = max(f["crop"] for f in fits)
    speed, best = best_reader(a.data, seqs, crop, a.cache)
    cpus = max(1, (os.cpu_count() or 2) // gpus - 1)
    reader, workers = best, min(cpus, max(1, math.ceil(need / speed[best])))
    if speed[best] * cpus < need and a.cache and best != "cache":
        n_frames = 2 * sum(len(v) for v in seqs.values())
        est = n_frames * Reader("pil")(next(iter(seqs.values()))[0][0]).nbytes
        Path(a.cache).mkdir(parents=True, exist_ok=True)
        if est < 0.8 * shutil.disk_usage(a.cache).free:
            print(f"plan: decoding gives {speed[best] * cpus:.0f} of the {need:.0f} pairs/s needed: caching "
                  f"{n_frames} frames ({est / 2 ** 30:.0f} GiB) into {a.cache}", flush=True)
            build_cache(a.data, a.cache, seqs=seqs)
            speed, best = best_reader(a.data, seqs, crop, a.cache)
            reader, workers = best, min(cpus, max(1, math.ceil(need / speed[best])))
        else:
            print(f"plan: the cache would need {est / 2 ** 30:.0f} GiB, more than the disk has free", flush=True)
    plan = {"model": name, "gpu": torch.cuda.get_device_name(device), "gpus": gpus,
            "gpu_memory_gib": round(torch.cuda.get_device_properties(device).total_memory / 2 ** 30, 1),
            "precision": precision, "compile": compiled, "compile_probe_s": compile_s, "channels_last": cl,
            "frames": frames, "stages": fits,
            "reader": reader, "reader_pairs_per_s": speed, "cache": a.cache if reader == "cache" else None,
            "workers_per_gpu": workers, "date": time.strftime("%Y-%m-%d %H:%M")}
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    return plan


# ---------------------------------------------------------------------------------------------------- the rest
def psnr(a, b):
    mse = ((a.float() - b.float()) ** 2).mean().item()
    return 10 * math.log10(255 ** 2 / max(mse, 1e-10))


@torch.no_grad()
def validate(net, seqs, max_frames, reader):
    """Mean PSNR of net.restore over the frames of seqs (8-bit outputs, as the benchmark scores them)."""
    net.eval()
    vals, n = [], 0
    for v in seqs.values():
        v = v[:max(1, max_frames - n)]
        blur = torch.stack([torch.from_numpy(np.array(reader(b))).permute(2, 0, 1) for b, _ in v])
        out = net.restore(blur, out="uint8")
        vals += [psnr(o, torch.from_numpy(np.array(reader(g))).permute(2, 0, 1).to(o.device))
                 for o, (_, g) in zip(out, v)]
        n += len(v)
        if n >= max_frames:
            break
    net.train()
    return float(np.mean(vals))


def export(net, source, out):
    """The weights in every store of the source's models: dbis-<source>.safetensors, dbisf-<source>.wpk."""
    sd = {k: v.detach().float().cpu().contiguous() for k, v in net.state_dict().items()}
    meta = {"trained_from": source, "tool": "python -m dbis.train"}
    for name, m in MODELS.items():
        if m["source"] != source:
            continue
        path = Path(out) / m["file"]
        if m["store"] == "float32":
            from safetensors.torch import save_file
            save_file(sd, str(path), metadata={"dbis": json.dumps({**meta, "model": name}, sort_keys=True)})
        else:
            from . import wpack
            tier, s = ("f16", 128) if m["store"] == "f16" else re.fullmatch(r"(qc|q)(\d+)", m["store"]).groups()
            wpack.pack({k: v.numpy() for k, v in sd.items()}, str(path), tier, int(s), threads=8,
                       meta={**meta, "model": name})
        print(f"wrote {path}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dbis.train", description=__doc__.split("\n\n")[0])
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--data", help="folder of the training pairs")
    ap.add_argument("--list", help="keep only the frames / sequences named in this file")
    ap.add_argument("--exclude", help="drop the frames / sequences named in this file")
    ap.add_argument("--val", help="folder of validation pairs (best.pt by its PSNR)")
    ap.add_argument("--val-list", help="keep only these validation frames / sequences")
    ap.add_argument("--val-every", type=int, default=2000, help="steps between checkpoints (and validations)")
    ap.add_argument("--log-every", type=int, default=100, help="steps between lines of log.tsv")
    ap.add_argument("--max-minutes", type=float, help="save last.pt and stop after this many minutes (resume later)")
    ap.add_argument("--val-frames", type=int, default=200, help="frames scored per validation")
    ap.add_argument("--out", default="runs/dbis_train")
    ap.add_argument("--init", choices=("released", "scratch"), default="released")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--loss", help="a loss spec of dbis/losses.py (default: the recipe's), e.g. l1+0.1*fft")
    ap.add_argument("--batch", type=int, help="global batch (default: the recipe's)")
    ap.add_argument("--batch-per-gpu", type=int, help="samples per GPU per forward pass (default: the plan's / the largest "
                    "that fits); the accumulation then makes up the global batch")
    ap.add_argument("--accum", type=int, help="gradient accumulation steps (default: as many as the global batch needs); "
                    "with --batch-per-gpu too, the global batch becomes batch-per-gpu x GPUs x accum")
    ap.add_argument("--crop", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--precision", choices=sorted(DTYPES), help="default: the plan's, else bf16 where supported")
    ap.add_argument("--cache", help="folder of the frame cache (data.py); the plan builds it when decoding is too slow")
    ap.add_argument("--workers", type=int, help="data workers per GPU (default: the plan's, else 8)")
    ap.add_argument("--reader", choices=("pil", "cv2", "tv", "cache", "ram", "gpu"),
                    help="frame reader (default: cache when --cache holds the frames, else the plan's); ram reads every "
                    "frame into memory first (per GPU process); gpu holds every frame on each GPU and cuts the batches "
                    "there, with no data workers (frames of one size)")
    ap.add_argument("--echo", type=int, default=1, help="crops per clip read (data echoing): every clip read feeds this "
                    "many batches with new crops and flips; for when reading is slower than the GPU")
    ap.add_argument("--shuffle", choices=("random", "perm"), default="random",
                    help="with --reader gpu: random draws of clips, or a new permutation of all clips every epoch")
    ap.add_argument("--ema", help="decays of weight averages kept beside the weights, e.g. 0.99,0.999 (written as "
                    "OUT/ema_<decay>.pt); the decay ramps as min(d, (1 + t) / (10 + t))")
    ap.add_argument("--export-ema", action="store_true", help="at the end, export the average of the first --ema decay "
                    "instead of the last weights")
    ap.add_argument("--noise-every", type=int, help="every N steps, the gradient noise scale B = tr(cov) / |G|^2 from the "
                    "gradients of the two halves and the whole batch (McCandlish et al. 2018), written to OUT/noise.tsv; "
                    "one GPU")
    ap.add_argument("--adapt-batch", help="MIN,MAX: start at batch MIN and raise it with the measured noise scale (the "
                    "largest power of two not above it, after 3 measurements in a row; never down, at most MAX); needs "
                    "--noise-every and --reader gpu")
    ap.add_argument("--rho", type=int, help="RHO-LOSS (Mindermann et al. 2022): cut this many times the batch, train on "
                    "the batch of the highest log(loss of the model / loss of a reference); needs --reader gpu")
    ap.add_argument("--rho-ref", help="the reference model of --rho, released weights (default: the trained model's)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="video: also play half of the clips backwards")
    ap.add_argument("--plan", action="store_true", help="find the settings for this machine, write OUT/plan.json, stop")
    ap.add_argument("--no-compile-probe", action="store_true",
                    help="with --plan: do not try torch.compile of the step (it takes a minute for NAFNet, longer for others)")
    ap.add_argument("--sweep", action="store_true", help="our table of settings (train_configs.json), then stop")
    ap.add_argument("--sweep-out", help="with --sweep: write this file instead of train_configs.json")
    a = ap.parse_args(argv)

    source = MODELS[a.model]["source"]
    r = dict(RECIPES[source])
    ft = a.init == "released"
    lr = a.lr or r["lr"] * (FINETUNE["lr"] if ft else 1)
    frames = a.frames or r["frames"]
    if ft or a.crop or a.batch or len(r["stages"]) == 1:
        # one stage: the recipe's last (its largest crop), or what --crop and --batch say
        stages = [(a.steps or int(r["steps"] * (FINETUNE["steps"] if ft else 1)), a.crop or r["crop"], a.batch or r["batch"])]
    else:                                              # the recipe's stages, scaled to --steps if given
        k = (a.steps or r["steps"]) / r["steps"]
        stages = [(max(1, round(n * k)), c, b) for n, c, b in r["stages"]]
    steps = sum(n for n, _, _ in stages)
    ends = np.cumsum([n for n, _, _ in stages]).tolist()
    multi = bool(r.get("multiscale"))
    loss_of = make_loss(a.loss or r["loss"])

    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True              # fixed shapes: the fastest conv algorithms
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    if a.sweep:
        return sweep(a.model, device, a.sweep_out)
    if not a.data:
        ap.error("--data is needed")
    allowed = getattr(build(a.model), "train_crops", None)
    if allowed and any(c not in allowed for _, c, _ in stages):
        ap.error(f"{a.model} trains on crops of {allowed} px only")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seqs = select(find_pairs(a.data), a.list and names_in(a.list), a.exclude and names_in(a.exclude))
    if a.plan:
        plan = make_plan(a, a.model, r, stages, frames, device, seqs, out)
        print(json.dumps(plan, indent=1))
        launcher = f"torchrun --nproc-per-node {plan['gpus']}" if plan["gpus"] > 1 else "python"
        print(f"next: {launcher} -m dbis.train {a.model} --data {a.data} --out {a.out}")
        return
    if world > 1:
        torch.distributed.init_process_group("nccl", device_id=device)
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8")) if (out / "plan.json").exists() else {}
    if plan and ([(f["crop"], f["global_batch"]) for f in plan["stages"]] != [(c, b) for _, c, b in stages]
                 or plan["frames"] != frames or plan["gpus"] != world):
        plan = {}                                      # made for other stages, frames or GPUs: not used
    precision = a.precision or plan.get("precision") or ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
    cache = a.cache or plan.get("cache")
    ready = bool(cache) and cache_ready(a.data, cache, seqs)
    kind = a.reader or ("cache" if ready else plan.get("reader", "pil"))
    if kind == "cache" and not ready:
        ap.error("--reader cache needs --cache with every frame of --data (python -m dbis.data cache)")
    on_gpu = kind == "gpu"
    if on_gpu:                                         # the frames are read once, from the cache if it has them
        kind = "cache" if ready else next(k for k in ("cv2", "tv", "pil") if k in available())
    reader = Reader(kind if kind in available() + ["cache", "ram"] else "pil", a.data, cache if ready else None)
    if (a.adapt_batch or a.rho or a.shuffle != "random") and not on_gpu:
        ap.error("--adapt-batch, --rho and --shuffle perm need --reader gpu")
    if (a.noise_every or a.adapt_batch) and world > 1:
        ap.error("--noise-every and --adapt-batch run on one GPU")
    if a.adapt_batch and not a.noise_every:
        ap.error("--adapt-batch needs --noise-every")
    if on_gpu:
        t_load = time.time()
        on_gpu = GpuFrames(seqs, frames, reader, device)
        if a.rho:
            ref_net = load(a.rho_ref or a.model, device=device).eval()
            on_gpu.add_reference(ref_net)
            del ref_net
            torch.cuda.empty_cache()
        if rank == 0:
            print(f"gpu: {on_gpu.gib:.1f} GiB of frames on each GPU{' with the reference of --rho' if a.rho else ''}, "
                  f"ready in {time.time() - t_load:.0f} s", flush=True)
    if reader.kind == "ram":
        gib = reader.preload([p for v in seqs.values() for pr in v for p in pr])
        if rank == 0:
            print(f"ram: {gib:.1f} GiB of frames held in memory", flush=True)
    workers = a.workers if a.workers is not None else plan.get("workers_per_gpu", 8)

    net = load(a.model, device=device)
    frozen = freeze(net)
    if not ft:                                         # scratch: new random weights except the frozen parts
        for n, mod in net.named_modules():
            if hasattr(mod, "reset_parameters") and not n.startswith(frozen):
                mod.reset_parameters()
    net.train()
    # per stage: the batch per GPU and checkpointing of the plan, else the largest batch that runs (a short search)
    fits = []
    for si, (n, crop, want) in enumerate(stages):
        f = plan["stages"][si] if plan else None
        b, ckpt = (f["batch_per_gpu"], f["checkpoint"]) if f else (None, False)
        if a.batch_per_gpu:
            b = a.batch_per_gpu
        elif a.accum:                                  # the per-GPU batch that gives the global batch with this accum
            b = -(-want // (world * a.accum))
        if b is None:
            b, _ = search_batch(net, r, source, crop, frames, precision, False, device, -(-want // world))
            if not b:
                ckpt = True
                b, _ = search_batch(net, r, source, crop, frames, precision, True, device, -(-want // world))
            if not b:
                raise SystemExit(f"even one sample of crop {crop} does not fit this GPU: give a smaller --crop or --frames")
        if world > 1:
            t = torch.tensor([b], device=device)
            torch.distributed.all_reduce(t, torch.distributed.ReduceOp.MIN)
            b = int(t)
        accum = a.accum or -(-want // (b * world))
        if not (a.batch_per_gpu and a.accum):
            b = -(-want // (world * accum))            # the same global batch with the fewest samples per pass
        if b * world * accum != want and rank == 0:
            print(f"note: global batch {b * world * accum} (= {b} x {world} GPUs x {accum}), the recipe's is {want}", flush=True)
        fits.append({"crop": crop, "batch": b, "accum": accum, "checkpoint": ckpt})
    opt = optimizer(net, r, lr)
    params = [p for g in opt.param_groups for p in g["params"]]
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    if (a.noise_every or a.adapt_batch) and (precision == "fp16" or any(f["accum"] > 1 for f in fits)):
        ap.error("--noise-every and --adapt-batch need bf16 or fp32 and no gradient accumulation")
    pname = {id(p): n for n, p in net.named_parameters()}
    emas = {float(d): [p.detach().clone() for p in params] for d in a.ema.split(",")} if a.ema else {}
    n_ema = 0
    adapt = [int(v) for v in a.adapt_batch.split(",")] if a.adapt_batch else None
    bsz_adapt = adapt[0] if adapt else None
    noise = {"S": None, "G2": None, "B": None, "votes": 0}
    noise_log = open(out / "noise.tsv", "a", encoding="utf-8") if a.noise_every and rank == 0 else None

    def grads():
        return [p.grad for p in params if p.grad is not None]

    def sq_norm(ts):
        return float(torch.stack(torch._foreach_norm(ts)).pow(2).sum())

    def rho_pick(parts, b):
        """The b candidates of the highest reducible loss: log(MSE of the model) - log(MSE of the reference)."""
        xb, xs, xr = parts
        cur = []
        with torch.no_grad(), torch.autocast("cuda", DTYPES[precision] or torch.float32, enabled=precision != "fp32"):
            for c in range(0, len(xb), b):
                x = xb[c:c + b].float().div_(255)
                if cl:
                    x = x[:, 0].contiguous(memory_format=torch.channels_last).unsqueeze(1)
                p = step_fn(x)
                p = p[-1] if isinstance(p, (list, tuple)) else p
                cur.append((p.float() - xs[c:c + b].float().div(255)).pow(2).flatten(1).mean(1))
        ref = (xr.float() - xs.float()).div(255).pow(2).flatten(1).mean(1)
        red = torch.log(torch.cat(cur) + 1e-10) - torch.log(ref + 1e-10)
        sel = red.topk(b).indices
        return xb[sel], xs[sel], float(red.mean()), float(red[sel].mean())

    def lr_at(s):
        if s < r["warmup"]:
            return lr * (s + 1) / r["warmup"]
        q = (s - r["warmup"]) / max(1, steps - r["warmup"])
        return LR_MIN + (lr - LR_MIN) * 0.5 * (1 + math.cos(math.pi * q))

    cl = bool(plan.get("channels_last")) and net.kind == "image"
    if cl:
        net.to(memory_format=torch.channels_last)
    step_fn = Step(net, multi)
    if world > 1:
        step_fn = torch.nn.parallel.DistributedDataParallel(step_fn, [local])
    if plan.get("compile"):                            # compiled around DDP, as PyTorch recommends
        # a new batch or crop size (stages, --adapt-batch, the halves of --noise-every) gets its own static graph
        torch._dynamo.config.automatic_dynamic_shapes = False
        torch._dynamo.config.cache_size_limit = max(16, torch._dynamo.config.cache_size_limit)
        step_fn = torch.compile(step_fn)
    start, best = 0, -1.0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=True)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        start, best = ck["step"], ck.get("best", -1.0)
        for d in emas:                                 # the averages go on from where they were
            if (out / f"ema_{d:g}.pt").exists():
                e = torch.load(out / f"ema_{d:g}.pt", map_location=device, weights_only=True)
                emas[d] = [e["net"][pname[id(p)]].clone() for p in params]
                n_ema = e.get("updates", 0)
    val = a.val and select(find_pairs(a.val), a.val_list and names_in(a.val_list))
    if rank == 0:
        print(f"{a.model}: {sum(len(v) for v in seqs.values())} pairs in {len(seqs)} sequences; frames {frames}; "
              f"{precision}, compile {bool(plan.get('compile'))}; "
              f"{'batches cut on the GPU' if on_gpu else f'reader {reader.kind}, {workers} workers per GPU, echo {a.echo}'}; "
              f"steps {start} -> {steps}, lr {lr:g} ({'fine-tune' if ft else 'scratch'}), loss {a.loss or r['loss']}"
              f"{' on every output' if multi else ''}", flush=True)
        for (n, crop, want), f in zip(stages, fits):
            print(f"  stage of {n} steps: crop {crop}, global batch {f['batch'] * world * f['accum']} = {f['batch']} per "
                  f"GPU x {world} GPUs x {f['accum']} accumulated, checkpointing {f['checkpoint']}", flush=True)
    log = open(out / "log.tsv", "a", encoding="utf-8") if rank == 0 else None
    t0, si, it, bad_steps = time.time(), -1, None, 0
    for step in range(start, steps):
        s_now = next(k for k, e in enumerate(ends) if step < e)
        if s_now != si:                                # a new stage: its crop, batch and checkpointing
            si, f = s_now, fits[s_now]
            first = ends[si] - stages[si][0]
            checkpointing(net, f["checkpoint"])
            if hasattr(net, "prepare"):                # constant tensors a model builds outside a compiled graph
                net.prepare(f["crop"], f["crop"], device)
            origin = (step - first) * f["accum"] * f["batch"]
            if on_gpu:                                 # batch counter and samples drawn in this stage
                g_n, g_pos = origin // f["batch"], origin
            else:
                ds = Clips(seqs, frames, f["crop"], reader, aug=not a.no_aug, reverse=a.reverse, seed=a.seed * 100 + si,
                           echo=a.echo, batch=f["batch"], workers=workers, rank=rank, world=world, origin=origin)
                it = iter(torch.utils.data.DataLoader(
                    ds, batch_size=f["batch"], sampler=Stream(origin, rank, world), num_workers=workers,
                    pin_memory=True, persistent_workers=False, prefetch_factor=4 if workers else None))
        for g in opt.param_groups:
            g["lr"] = lr_at(step) * g["mult"]
        tot, bsz, rho_red = 0.0, bsz_adapt or f["batch"], None
        measure = bool(a.noise_every) and step % a.noise_every == 0 and bsz >= 2
        for k in range(f["accum"]):
            if on_gpu:
                parts = on_gpu.cut(g_n, bsz * (a.rho or 1), f["crop"], not a.no_aug, a.reverse, a.seed * 100 + si,
                                   rank, a.shuffle, g_pos, world)
                g_n, g_pos = g_n + 1, g_pos + bsz * (a.rho or 1)
                if a.rho:
                    *parts, red_all, red_sel = rho_pick(parts, bsz)
                    rho_red = (red_all, red_sel)
                blur, sharp = (t.float().div_(255) for t in parts[:2])
            else:
                blur, sharp = (t.to(device, non_blocking=True).float().div_(255) for t in next(it))
            if cl:
                blur = blur[:, 0].contiguous(memory_format=torch.channels_last).unsqueeze(1)
            sync = world == 1 or k == f["accum"] - 1
            # a noise measurement: the batch in two halves; |G|^2 of each half (B/2 samples) and of the whole (B)
            half = bsz // 2
            halves = ((blur[:half], sharp[:half]), (blur[half:2 * half], sharp[half:2 * half])) if measure \
                else ((blur, sharp),)
            for hi, (xb, xs) in enumerate(halves):
                with torch.autocast("cuda", DTYPES[precision] or torch.float32, enabled=precision != "fp32"), \
                        (contextlib.nullcontext() if sync else step_fn.no_sync()):
                    loss = loss_over(loss_of, step_fn(xb), xs) * len(xb) / (2 * half if measure else bsz) / f["accum"]
                    scaler.scale(loss).backward()
                tot += loss.item()
                if measure and hi == 0:
                    first = [g.detach().clone() for g in grads()]
        if measure:                                    # the stored gradients are halves' means divided by 2
            second = torch._foreach_sub(grads(), first)
            g2_small = 2 * (sq_norm(first) + sq_norm(second))  # mean of the two halves' |G_{B/2}|^2
            g2_big, bs_, bb_ = sq_norm(grads()), half, 2 * half
            del first, second
            s_ = (g2_small - g2_big) / (1 / bs_ - 1 / bb_)
            g2_ = (bb_ * g2_big - bs_ * g2_small) / (bb_ - bs_)
            for key, v in (("S", s_), ("G2", g2_)):            # smoothed over about 20 measurements
                noise[key] = v if noise[key] is None else 0.95 * noise[key] + 0.05 * v
            noise["B"] = noise["S"] / noise["G2"] if noise["G2"] > 0 else float("inf")
            if adapt and math.isfinite(noise["B"]):
                # the largest power of two not above the noise scale, raised only after 3 measurements in a row ask
                want = min(adapt[1], 2 ** int(math.log2(max(1.0, noise["B"]))))
                noise["votes"] = noise["votes"] + 1 if want > bsz_adapt else 0
                if noise["votes"] >= 3:
                    bsz_adapt, noise["votes"] = want, 0
            if noise_log:
                noise_log.write(f"{step}\t{bsz}\t{g2_small:.6g}\t{g2_big:.6g}\t{noise['S']:.6g}\t{noise['G2']:.6g}\t"
                                f"{noise['B']:.4g}\t{time.time() - t0:.0f}\n")
                noise_log.flush()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, r["clip"] or float("inf"))
        if torch.isfinite(gn):
            scaler.step(opt)
            bad_steps = 0
            if emas:
                n_ema += 1
                cur = [p.detach() for p in params]
                for d, e in emas.items():
                    torch._foreach_lerp_(e, cur, 1 - min(d, (1 + n_ema) / (10 + n_ema)))
        elif precision != "fp16":                      # fp16: the loss scaler handles occasional overflows
            bad_steps += 1
            if bad_steps >= 50:
                raise SystemExit(f"step {step}: 50 steps in a row with non-finite gradients; rerun --plan or use "
                                 "--precision fp32")
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if log and (step % a.log_every == 0 or step == steps - 1):
            extra = (f"\t{bsz}" if adapt else "") + (f"\t{rho_red[0]:.4f}\t{rho_red[1]:.4f}" if rho_red else "")
            log.write(f"{step}\t{tot:.5f}\t{float(gn):.4f}\t{lr_at(step):.3e}\t{time.time() - t0:.0f}{extra}\n")
            log.flush()
        timeout = a.max_minutes is not None and time.time() - t0 > 60 * a.max_minutes
        if world > 1 and a.max_minutes is not None:            # every rank stops at the same step
            t = torch.tensor([float(timeout)], device=device)
            torch.distributed.all_reduce(t, torch.distributed.ReduceOp.MAX)
            timeout = bool(t)
        if (step + 1) % a.val_every == 0 or step == steps - 1 or timeout:
            if rank == 0:
                rec = {"net": net.state_dict(), "opt": opt.state_dict(), "step": step + 1, "best": best}
                if val:
                    v = validate(net, val, a.val_frames, Reader("pil"))
                    log.write(f"val\t{step + 1}\t{v:.4f}\n")
                    log.flush()
                    if v > best:
                        rec["best"] = best = v
                        torch.save({"net": net.state_dict(), "step": step + 1, "val_psnr": v}, out / "best.pt")
                torch.save(rec, out / "last.pt.tmp")
                os.replace(out / "last.pt.tmp", out / "last.pt")
                for d, e in emas.items():                  # the averaged weights, loadable like last.pt
                    sd = dict(net.state_dict())
                    sd.update({pname[id(p)]: v for p, v in zip(params, e)})
                    torch.save({"net": sd, "step": step + 1, "ema": d, "updates": n_ema}, out / f"ema_{d:g}.pt")
            if world > 1:
                torch.distributed.barrier()
        if timeout:
            if rank == 0:
                print(f"stopped after {a.max_minutes} minutes at step {step + 1}; last.pt resumes", flush=True)
            if world > 1:
                torch.distributed.destroy_process_group()
            return
    if rank == 0:
        if val and (out / "best.pt").exists():
            net.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True)["net"])
        elif a.export_ema and emas:
            with torch.no_grad():
                torch._foreach_copy_(params, next(iter(emas.values())))
        export(net, source, out)
        (out / "done.json").write_text(json.dumps({"model": a.model, "steps": steps, "best_val_psnr": best}) + "\n")
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
