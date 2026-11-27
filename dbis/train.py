"""dbis.train: one training script for every model of dbis.

    python -m dbis.train dbisf-NAFNet-64 --data DIR --out runs/nafnet               # one GPU
    torchrun --nproc-per-node 8 -m dbis.train dbisf-NAFNet-64 --data DIR --out ...   # several GPUs, the same result
    python -m dbis.train dbisf-NAFNet-64 --measure                                  # the memory table (below)

Data. DIR holds pairs of blurry and sharp frames and the layout is found by itself: a blurry frame is an image whose
path has one of the blurry names of SIDES (blur, Blur, input_noise, input), its sharp frame the same path with that
name replaced (sharp, Sharp, target, GT, gt). So DIR/blur/*.png + DIR/sharp/*.png (image sets), DIR/<seq>/blur/*.png +
DIR/<seq>/sharp/*.png (GoPro), DIR/train_blur/<seq>/*.png + DIR/train_sharp/<seq>/*.png (REDS), <seq>/Blur/RGB +
<seq>/Sharp/RGB (BSD), <seq>/input + <seq>/GT (DVD) all work. Frames of one folder are one sequence, in name order;
video models train on clips of consecutive frames of one sequence. --list FILE keeps the frames or sequences named
in FILE (one name per line), --exclude FILE drops them.

Recipe. RECIPES[source] is the authors' GoPro recipe of the model (crop, frames per clip, global batch, steps, peak
learning rate, loss, clip; DBIS.md, "Training"). Every model then trains the same way: AdamW, a linear warmup and
a cosine decay to 1e-7, bfloat16 autocast, and one set of augmentations (a random crop at the same place in every frame
and in both frames of a pair, then one of the 8 flips and transposes; --reverse also plays clips backwards).
--init released (the default) starts from the released weights of the model (dbis.load), --init scratch from random
ones; a fine-tune takes FINETUNE of the recipe's steps and learning rate unless --steps and --lr say otherwise.

Batch size. The memory of a training step was measured for every model (MEMORY, train_memory.json; --measure: the
peak of a step is fixed + per_sample * batch, at the recipe's crop and frames, on one A100 80 GB). The script takes
the largest batch per GPU that fits the free memory of the GPU it runs on (per_sample scaled by the crop's area and the
frames), checks it with one step and halves it while that step runs out of memory, and reaches the recipe's global
batch by gradient accumulation. So the same command gives the same training on any GPU.

Output in --out: last.pt (a stopped run resumes from it), log.tsv, best.pt (with --val), and at the end the weight
files of the model's source in the stores of dbis.MODELS (dbis-<source>.safetensors, dbisf-<source>.wpk), which
dbis.load(name, weights_dir=OUT) reads.
"""
import argparse
import contextlib
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import MODELS, SOURCES, build, load

MEMORY_FILE = Path(__file__).with_name("train_memory.json")
IMAGES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
SIDES = (("blur", "sharp"), ("Blur", "Sharp"), ("input_noise", "target"), ("input", "target"), ("input", "GT"),
         ("blur", "gt"))

# The authors' GoPro recipes (their training configs at the commits of dbis.SOURCES; DBIS.md, "Training"): crop, frames
# per clip, global batch, steps, AdamW (lr, betas, weight decay), warmup steps, lr multipliers by parameter name, loss,
# gradient-norm clip. The cosine decay ends at 1e-7 in all four. What is the same for all here, and was not in every
# original: bfloat16 autocast (NAFNet and DSTNet+ trained in float32, BSSTNet and Turtle in float16 with a scaler) and
# one set of augmentations (the 8 flips and transposes, uniform; Turtle kept the frame unchanged half of the time).
RECIPES = {
    # NAFNet options/train/GoPro/NAFNet-width64.yml: 8 per GPU x 8 GPUs, 400k, PSNR loss, clip 0.01 (basicsr default)
    "NAFNet-64": {"crop": 256, "frames": 1, "batch": 64, "steps": 400_000, "lr": 1e-3, "betas": (0.9, 0.9),
                  "weight_decay": 1e-3, "warmup": 0, "lr_mult": {}, "loss": "psnr", "clip": 0.01},
    # BSSTNet options/train/BSST/gopro_BSST.yml + Video_BSST_model.py: global batch 8 (the paper), L1 over the 24
    # frames, RAFT frozen (train_frozen), per-group lr of the flow completion, offsets and soft split / composition
    "BSSTNet": {"crop": 256, "frames": 24, "batch": 8, "steps": 600_000, "lr": 4e-4, "betas": (0.9, 0.99),
                "weight_decay": 0.0, "warmup": 1000,
                "lr_mult": {"blur_motion_refine": 0.1, "conv_offset": 0.25, "softsplit": 0.5, "softcomp": 0.5},
                "loss": "l1", "clip": 0.01},
    # DSTNet-plus options/train/DSTNetPlus/DSTPlus_large_GoPro.yml: 1 per GPU x 8, Charbonnier + 0.1 x L1 of the rfft2
    "DSTNetPlus-L": {"crop": 256, "frames": 15, "batch": 8, "steps": 600_000, "lr": 2e-4, "betas": (0.9, 0.999),
                     "weight_decay": 1e-4, "warmup": 0, "lr_mult": {}, "loss": "charbonnier+fft", "clip": 0.01},
    # Turtle options/Turtle_Deblur_Gopro.yml + video_restoration_model.py: 2 per GPU x 8, L1 per frame, no clip; the
    # cache carried through the 5-frame clip without detaching (train_forward)
    "Turtle": {"crop": 192, "frames": 5, "batch": 16, "steps": 200_000, "lr": 4e-4, "betas": (0.9, 0.99),
               "weight_decay": 0.0, "warmup": 0, "lr_mult": {}, "loss": "l1", "clip": None},
}
LR_MIN = 1e-7
FINETUNE = {"steps": 0.1, "lr": 0.1}       # a fine-tune from released weights: 10 % of the steps at 10 % of the lr


# ---------------------------------------------------------------------------------------------------- data
def read(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def find_pairs(root):
    """{sequence: [(blurry path, sharp path), ...] in name order} of every pair under root (see SIDES)."""
    root, seqs = Path(root), {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGES:
            continue
        rel = p.relative_to(root).as_posix()
        for a, b in SIDES:
            if a in rel and (root / rel.replace(a, b)).is_file():
                seqs.setdefault(str(Path(rel).parent), []).append((p, root / rel.replace(a, b)))
                break
    if not seqs:
        raise FileNotFoundError(f"no pairs of blurry and sharp frames under {root} (names: {SIDES})")
    return seqs


def names_in(path):
    return {s.strip() for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()}


def select(seqs, keep=None, drop=None):
    """Frames or sequences named in keep (if given) and not in drop: a frame matches by its file name or stem."""
    def hit(names, seq, b):
        return seq in names or b.name in names or b.stem in names
    out = {}
    for s, v in seqs.items():
        v = [(b, g) for b, g in v if (keep is None or hit(keep, s, b)) and (drop is None or not hit(drop, s, b))]
        if v:
            out[s] = v
    return out


class Clips(torch.utils.data.Dataset):
    """Sample i: a random clip of `frames` consecutive pairs of one sequence (one pair for image models), cropped and
    augmented, as two uint8 tensors (T, 3, crop, crop). Every sample is a function of (seed, i) alone."""

    def __init__(self, seqs, frames, crop, aug=True, reverse=False, seed=0):
        self.items = [(v, i) for v in seqs.values() for i in range(len(v) - frames + 1)]
        if not self.items:
            raise ValueError(f"no sequence has {frames} frames")
        self.frames, self.crop, self.aug, self.reverse, self.seed = frames, crop, aug, reverse and frames > 1, seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)
        v, s = self.items[rng.randrange(len(self.items))]
        pairs = [(read(b), read(g)) for b, g in v[s:s + self.frames]]
        h, w = pairs[0][0].shape[:2]
        c = self.crop
        if h < c or w < c:
            raise ValueError(f"{v[s][0]}: {h} x {w} is smaller than the crop {c}")
        y, x = rng.randrange(h - c + 1), rng.randrange(w - c + 1)
        blur = np.stack([b[y:y + c, x:x + c] for b, _ in pairs])
        sharp = np.stack([g[y:y + c, x:x + c] for _, g in pairs])
        if self.aug:
            flips = [rng.random() < 0.5 for _ in range(4)]

            def aug(a):
                if flips[0]:
                    a = a[:, :, ::-1]
                if flips[1]:
                    a = a[:, ::-1]
                if flips[2]:
                    a = a.transpose(0, 2, 1, 3)
                if flips[3] and self.reverse:
                    a = a[::-1]
                return a
            blur, sharp = aug(blur), aug(sharp)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2)   # noqa: E731
        return t(blur), t(sharp)


class Stream(torch.utils.data.Sampler):
    """Sample indices start, start + 1, ... of this rank (rank, rank + world, ...): resumable, the same over ranks."""

    def __init__(self, start, rank, world):
        self.start, self.rank, self.world = start, rank, world

    def __iter__(self):
        i = self.start * self.world + self.rank
        while True:
            yield i
            i += self.world


# ---------------------------------------------------------------------------------------------------- losses
def loss_fn(kind):
    if kind == "psnr":          # NAFNet's PSNRLoss: 10 log10 of the mean squared error of each sample (= -PSNR)
        return lambda p, y: 10 * torch.log10(((p - y) ** 2).flatten(1).mean(1) + 1e-8).mean()
    if kind == "charbonnier+fft":    # DSTNet+: sqrt(d^2 + 1e-12) + 0.1 x L1 between the rfft2 of output and target
        def f(p, y):
            fp, fy = (torch.view_as_real(torch.fft.rfft2(t.float())) for t in (p, y))
            return torch.sqrt((p - y) ** 2 + 1e-12).mean() + 0.1 * (fp - fy).abs().mean()
        return f
    if kind == "l1":
        return lambda p, y: (p - y).abs().mean()
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------------------- memory
def memory_entry(name, crop, frames):
    """The measured entry of the model's source and its crop / frames scale, or None."""
    if not MEMORY_FILE.exists():
        return None, 1.0
    table = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    m = table.get(MODELS[name]["source"])
    if m is None:
        return None, 1.0
    return m, (crop / m["crop"]) ** 2 * (frames / m["frames"])


def fit_batch(name, crop, frames, want, device):
    """The largest batch per GPU by the memory table (want if the model was not measured)."""
    m, scale = memory_entry(name, crop, frames)
    if m is None:
        return want
    free = torch.cuda.mem_get_info(device)[0] / 2 ** 30
    return max(1, min(want, int((0.93 * free - m["fixed_gib"]) // (m["per_sample_gib"] * scale))))


def measure(name, crop, frames, device, batches=(1, 2, 4, 8)):
    """Peak memory of a training step (forward, backward, AdamW step) at batch 1, 2, 4, 8 (until one runs out of
    memory), fitted as fixed + per_sample * batch; written to train_memory.json under the model's source."""
    peaks = {}
    for b in batches:
        net = build(name, device)                        # random weights: the memory does not depend on them
        freeze(net)
        net.train()
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-6)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            for _ in range(3):
                x = torch.rand(b, frames, 3, crop, crop, device=device)
                with torch.autocast("cuda", torch.bfloat16):
                    p = forward(net, x)
                loss = (p.float() - x).abs().mean()
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            peaks[b] = torch.cuda.max_memory_reserved(device) / 2 ** 30
            print(f"{name}: batch {b}: {peaks[b]:.2f} GiB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{name}: batch {b}: out of memory", flush=True)
            break
        finally:
            del net, opt
            torch.cuda.empty_cache()
    bs = sorted(peaks)
    if len(bs) < 2:
        raise RuntimeError(f"{name}: fewer than two batch sizes fit at crop {crop}, frames {frames}")
    per = (peaks[bs[-1]] - peaks[bs[0]]) / (bs[-1] - bs[0])
    entry = {"crop": crop, "frames": frames, "fixed_gib": round(peaks[bs[0]] - per * bs[0], 3),
             "per_sample_gib": round(per, 3), "peaks_gib": {str(b): round(v, 3) for b, v in peaks.items()},
             "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "precision": "bf16 autocast",
             "date": time.strftime("%Y-%m-%d")}
    table = json.loads(MEMORY_FILE.read_text(encoding="utf-8")) if MEMORY_FILE.exists() else {}
    table[MODELS[name]["source"]] = entry
    MEMORY_FILE.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(entry))
    return entry


# ---------------------------------------------------------------------------------------------------- training
def freeze(net):
    """No gradients for the parts a model keeps fixed in training (net.train_frozen: module name prefixes, e.g.
    BSSTNet's RAFT); returns the prefixes."""
    frozen = tuple(getattr(net, "train_frozen", ()))
    for n, p in net.named_parameters():
        if n.startswith(frozen):
            p.requires_grad_(False)
    return frozen


def forward(net, x):
    """x (B, T, 3, H, W) -> prediction of the same shape (image models see (B, 3, H, W))."""
    if net.kind == "image":
        return net.train_forward(x[:, 0])[:, None]
    return net.train_forward(x)


def psnr(a, b):
    mse = ((a.float() - b.float()) ** 2).mean().item()
    return 10 * math.log10(255 ** 2 / max(mse, 1e-10))


@torch.no_grad()
def validate(net, seqs, max_frames):
    """Mean PSNR of net.restore over the frames of seqs (8-bit outputs, as the benchmark scores them)."""
    net.eval()
    vals, n = [], 0
    for v in seqs.values():
        v = v[:max(1, max_frames - n)]
        blur = torch.stack([torch.from_numpy(read(b).copy()).permute(2, 0, 1) for b, _ in v])
        out = net.restore(blur, out="uint8")
        vals += [psnr(o, torch.from_numpy(read(g).copy()).permute(2, 0, 1).to(o.device)) for o, (_, g) in zip(out, v)]
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
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-frames", type=int, default=200, help="frames scored per validation")
    ap.add_argument("--out", default="runs/dbis_train")
    ap.add_argument("--init", choices=("released", "scratch"), default="released")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch", type=int, help="global batch (default: the recipe's)")
    ap.add_argument("--crop", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--workers", type=int, default=8, help="data workers per GPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="video: also play half of the clips backwards")
    ap.add_argument("--measure", action="store_true", help="measure the memory of a training step and stop")
    a = ap.parse_args(argv)

    source = MODELS[a.model]["source"]
    r = dict(RECIPES[source])
    ft = a.init == "released"
    steps = a.steps or int(r["steps"] * (FINETUNE["steps"] if ft else 1))
    lr = a.lr or r["lr"] * (FINETUNE["lr"] if ft else 1)
    crop, frames, want = a.crop or r["crop"], a.frames or r["frames"], a.batch or r["batch"]

    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    if world > 1:
        torch.distributed.init_process_group("nccl", device_id=device)
    if a.measure:
        if rank == 0:
            measure(a.model, crop, frames, device)
        return
    if not a.data:
        ap.error("--data is needed")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seqs = select(find_pairs(a.data), a.list and names_in(a.list), a.exclude and names_in(a.exclude))
    val = a.val and select(find_pairs(a.val), a.val_list and names_in(a.val_list))
    net = load(a.model, device=device)
    frozen = freeze(net)
    if not ft:                                         # scratch: new random weights except the frozen parts
        for n, mod in net.named_modules():
            if hasattr(mod, "reset_parameters") and not n.startswith(frozen):
                mod.reset_parameters()
    net.train()
    params, groups = [], {}
    for n, p in net.named_parameters():
        if p.requires_grad:
            params.append(p)
            groups.setdefault(next((v for k, v in r["lr_mult"].items() if k in n), 1.0), []).append(p)
    opt = torch.optim.AdamW([{"params": v, "mult": m} for m, v in groups.items()], lr=lr, betas=r["betas"],
                            weight_decay=r["weight_decay"])
    loss_of = loss_fn(r["loss"])

    def lr_at(s):
        if s < r["warmup"]:
            return lr * (s + 1) / r["warmup"]
        q = (s - r["warmup"]) / max(1, steps - r["warmup"])
        return LR_MIN + (lr - LR_MIN) * 0.5 * (1 + math.cos(math.pi * q))

    # the batch per GPU: the memory table's guess, checked with one step, halved while it runs out of memory
    b = fit_batch(a.model, crop, frames, -(-want // world), device)
    while True:
        try:
            x = torch.rand(b, frames, 3, crop, crop, device=device)
            with torch.autocast("cuda", torch.bfloat16):
                loss = loss_of(forward(net, x).float(), x)
            loss.backward()
            opt.zero_grad(set_to_none=True)
            del x, loss
            break
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if b == 1:
                raise
            b //= 2
    if world > 1:
        t = torch.tensor([b], device=device)
        torch.distributed.all_reduce(t, torch.distributed.ReduceOp.MIN)
        b = int(t)
    accum = -(-want // (b * world))
    b = -(-want // (world * accum))                    # the same global batch with the fewest samples per step
    ddp = torch.nn.parallel.DistributedDataParallel(net, [local]) if world > 1 else net

    start, best = 0, -1.0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=True)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        start, best = ck["step"], ck.get("best", -1.0)
    if rank == 0:
        print(f"{a.model}: {sum(len(v) for v in seqs.values())} pairs in {len(seqs)} sequences; crop {crop}, frames "
              f"{frames}; global batch {b * world * accum} = {b} per GPU x {world} GPUs x {accum} accumulated; steps "
              f"{start} -> {steps}, lr {lr:g} ({'fine-tune' if ft else 'scratch'}), loss {r['loss']}", flush=True)
    ds = Clips(seqs, frames, crop, aug=not a.no_aug, reverse=a.reverse, seed=a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=b, sampler=Stream(start * accum * b, rank, world),
                                     num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0,
                                     prefetch_factor=4 if a.workers else None)
    it = iter(dl)
    log = open(out / "log.tsv", "a", encoding="utf-8")
    t0 = time.time()
    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step) * g["mult"]
        tot = 0.0
        for k in range(accum):
            blur, sharp = (t.to(device, non_blocking=True).float().div_(255) for t in next(it))
            sync = world == 1 or k == accum - 1
            with torch.autocast("cuda", torch.bfloat16), (contextlib.nullcontext() if sync else ddp.no_sync()):
                loss = loss_of(forward(ddp, blur).float(), sharp) / accum
                loss.backward()
            tot += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(params, r["clip"] or float("inf"))
        if torch.isfinite(gn):
            opt.step()
        opt.zero_grad(set_to_none=True)
        if rank == 0 and (step % 100 == 0 or step == steps - 1):
            log.write(f"{step}\t{tot:.5f}\t{float(gn):.4f}\t{lr_at(step):.3e}\t{time.time() - t0:.0f}\n")
            log.flush()
        last = step == steps - 1
        if (step + 1) % a.val_every == 0 or last:
            if rank == 0:
                rec = {"net": net.state_dict(), "opt": opt.state_dict(), "step": step + 1, "best": best}
                if val:
                    v = validate(net, val, a.val_frames)
                    log.write(f"val\t{step + 1}\t{v:.4f}\n")
                    log.flush()
                    if v > best:
                        rec["best"] = best = v
                        torch.save({"net": net.state_dict(), "step": step + 1, "val_psnr": v}, out / "best.pt")
                torch.save(rec, out / "last.pt.tmp")
                os.replace(out / "last.pt.tmp", out / "last.pt")
            if world > 1:
                torch.distributed.barrier()
    if rank == 0:
        if val and (out / "best.pt").exists():
            net.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True)["net"])
        export(net, source, out)
        (out / "done.json").write_text(json.dumps({"model": a.model, "steps": steps, "best_val_psnr": best}) + "\n")
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
