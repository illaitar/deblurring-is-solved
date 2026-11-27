# dbis and dbisf

Methods of the benchmark re-implemented in this repository, in the package `dbis/`.

- **dbis-*method*** is our code and weight file for the authors' network: plain PyTorch in one folder per method,
  every folder with the same layout (below), no training framework, weights as safetensors. It gives the same outputs as the authors' code: the largest
  difference of an output value is about 1e-5 and the mean PSNR on the test set is equal to the fourth decimal.
- **dbisf-*method*** is our fast and light version of it. Its mean PSNR is at most 0.05 dB below the authors' code on
  every test set of the benchmark; the tables below give every set.

## Models

| Model | Weights | Load | Per 720p frame | GoPro PSNR, dB |
|---|---:|---:|---:|---:|
| NAFNet-64, authors' code (TLC on) | 272 MB `.pth` | 3.06 s | 34.9 ms | 33.7103 |
| **dbis-NAFNet-64** | 271 MB `.safetensors` | 0.70 s | 33.5 ms | 33.7103 |
| **dbisf-NAFNet-64** | 66 MB `.wpk` | 0.77 s | 26.6 ms | 33.7041 (−0.006) |
| BSSTNet, authors' code (BSST + RAFT) | 209 MB, two `.pth` | 3.50 s | 25.6 s | 35.9662 |
| **dbis-BSSTNet** | 175 MB `.safetensors` | 1.66 s | 30.6 s | 35.9662 |
| **dbisf-BSSTNet** | 42 MB `.wpk` | 1.71 s | 4.6 s | 35.9425 (−0.024) |
| DSTNet+ L, authors' code | 56 MB `.pth` | 2.78 s | 3.01 s | 35.7078 |
| **dbis-DSTNetPlus-L** | 56 MB `.safetensors` | 0.54 s | 3.45 s | 35.7078 |
| **dbisf-DSTNetPlus-L** | 16 MB `.wpk` | 0.60 s | 0.58 s | 35.6999 (−0.008) |
| Turtle, authors' code (tiles 320, overlap 192) | 237 MB `.pth` | 0.69 s | 761 ms | 34.5598 |
| **dbis-Turtle** | 236 MB `.safetensors` | 0.75 s | 724 ms | 34.5598 |
| **dbisf-Turtle** | 68 MB `.wpk` | 0.83 s | 199 ms | 34.5316 (−0.028) |

Load: a fresh Python process, from CUDA ready to the network with its weights on the GPU (importing torch and starting
CUDA take another 2.3 s for every model). Time: the benchmark's latency harness on one A100, 1280x720, batch 1, the
three versions of a method on the same node. NAFNet: fp16 autocast, `torch.compile` max-autotune, CUDA graphs. BSSTNet:
as in the leaderboard, a window for every output frame (dbisf: 24 frames, the others 48), and each version in its
fastest mode that runs: the authors' code with fp16 autocast and `torch.compile` (20 minutes of compilation), dbis fp16
eager (`torch.compile` fails on it), dbisf fp16 with its heavy blocks compiled one by one (`net.compile_blocks()`; the
whole forward does not compile). Without compilation the three take 34.9, 30.6 and 6.6 s. DSTNet+ L: as in the
leaderboard, a 27-frame window for every output frame; the authors' code and dbis in float32 (the authors' CuPy kernel
has no float16), dbisf in float16 with its propagation blocks compiled (0.87 s without). Turtle: one frame per step
with the history carried, as the authors' tiled inference; the authors' code and dbis with fp16 autocast,
`torch.compile` max-autotune and CUDA graphs (13 minutes of compilation; in float32 without it 4.80 and 4.78 s), dbisf
with fp16 autocast and `torch.compile` in its default mode (max-autotune of the whole forward does not compile within
30 minutes; 0.41 s without compilation). PSNR: all 1111 GoPro test frames, 8-bit outputs. Every number is in
[`results/dbis/NAFNet-64.json`](results/dbis/NAFNet-64.json), [`results/dbis/BSSTNet.json`](results/dbis/BSSTNet.json),
[`results/dbis/DSTNetPlus-L.json`](results/dbis/DSTNetPlus-L.json) and [`results/dbis/Turtle.json`](results/dbis/Turtle.json).

## dbisf on every test set

Mean PSNR difference to the authors' code, dB.

**NAFNet-64** (the same run for all versions: float32, TF32 off, 8-bit outputs; RealBlur without the ECC alignment of
its official evaluation, so only the differences are comparable with the leaderboard)

| Test set | Frames | Authors, dB | dbisf-NAFNet-64 | first dbisf (76 MB) |
|---|---:|---:|---:|---:|
| GoPro | 1111 | 33.7103 | −0.006 | −0.000 |
| REDS val | 3000 | 26.6543 | −0.010 | −0.011 |
| BSD 2ms/16ms | 3000 | 25.3429 | +0.044 | −0.049 |
| DVD | 1000 | 29.6117 | −0.027 | −0.018 |
| RealBlur-J | 980 | 26.3325 | −0.019 | −0.006 |
| RealBlur-R | 980 | 33.6260 | −0.006 | −0.018 |

Outside GoPro, the training set of these weights, the network is sensitive to small changes of its weights. The fast TLC
alone (with float32 weights) stays within 0.004 dB of the authors on every set; the differences above come from the
weight rounding, and two roundings of the same weights move the mean in opposite directions on BSD. On 59 frames of
REDS, 27 of BSD and 3 of RealBlur one of the two versions is more than 1 dB from the authors (up to 21 dB); on some of
these BSD frames the authors' own output scores 4.5 to 8 dB. These frames are counted in the means above.

**BSSTNet** (the four video test sets; the authors' code as the leaderboard runs it: fp16 autocast, windows of 48
frames that overlap by 4; dbisf the same way with windows of 24 frames and its heavy blocks compiled, as timed)

| Test set | Frames | Authors, dB | dbisf-BSSTNet | Frames with black patches | dbisf without them | first dbisf (48 frames) |
|---|---:|---:|---:|---:|---:|---:|
| GoPro | 1111 | 35.9662 | −0.024 | 0 | −0.024 | −0.010 |
| REDS val | 3000 | 24.3901 | +3.357 | 466 | −0.031 | +3.351 |
| BSD 2ms/16ms | 3000 | 22.7700 | +1.777 | 376 | +0.037 | +1.808 |
| DVD | 1000 | 32.6798 | −0.034 | 0 | −0.034 | −0.013 |

**DSTNet+ L** (the four video test sets; the authors' code as the leaderboard runs it: float32, clips of 50 frames that
do not overlap, the last one moved back; dbisf the same way, with its propagation blocks compiled, as timed)

| Test set | Frames | Authors, dB | dbisf-DSTNetPlus-L | Frames the authors break | dbisf without them | first dbisf (25.11) |
|---|---:|---:|---:|---:|---:|---:|
| GoPro | 1111 | 35.7078 | −0.008 | 0 | −0.008 | −0.008 |
| REDS val | 3000 | 27.0689 | +0.007 | 0 | +0.007 | +0.006 |
| BSD 2ms/16ms | 3000 | 23.2542 | +0.020 | 23 | +0.015 | +0.019 |
| DVD | 1000 | 32.0308 | +0.013 | 0 | +0.013 | +0.013 |

On 23 BSD frames the authors' output itself scores 4.8 to 13.9 dB; the rule is checked without them as well.

**Turtle** (the four video test sets; the authors' tiled inference as the leaderboard runs it: tiles of 320 px with
192 px overlap, each with its own history, one pass per sequence; every frame compared with the leaderboard's own row
of that frame; dbisf compiled, as timed)

| Test set | Frames | Authors, dB | dbisf-Turtle (15 tiles) | 18 tiles | 45 tiles, float16 and dbisf's weights |
|---|---:|---:|---:|---:|---:|
| GoPro | 1111 | 34.5598 | −0.028 | −0.023 | −0.000 |
| REDS val | 3000 | 27.3461 | −0.028 | −0.014 | +0.006 |
| BSD 2ms/16ms | 3000 | 27.8414 | −0.021 | −0.016 | +0.006 |
| DVD | 1000 | 31.1838 | −0.006 | −0.009 | +0.005 |

dbis-Turtle gives the authors' PSNR on every one of the 8,111 frames. All of dbisf's difference comes from the fewer
tiles (15 at 1280x720 instead of 45, 6 at 640x480 instead of 12): with the authors' 45 tiles, float16 and the rounded
weights stay within 0.006 dB. 12 tiles cost 0.22 dB on GoPro; 15 tiles with 64 px overlap instead of 80 cost up to
0.038 dB at the same speed.

On REDS and BSD the authors' code writes NaN into some 256 px patches, which become black tiles in the 8-bit output.
Two causes: float16 overflow, and the recurrent backward propagation, whose features grow out of the GoPro range until
a LayerNorm overflows in float32 as well (the leaderboard's BSD row is the authors' code in float32, 23.7607 dB, and
still has such tiles). dbisf-BSSTNet runs a patch whose fp16 output is not finite again in float32, without autocast
(the first dbisf: REDS 117 patches, BSD 29), and keeps the blurry input where the float32 result is not finite either
(115 and 14 of them), so no frame has a black tile. This is why dbisf scores higher on these two sets; on the other
frames it is within 0.05 dB of the authors, and the rule is checked on those frames too. Without the redo the first
dbisf's rounded weights give −0.010, +0.070, −0.027 and −0.013 dB, with the black tiles in both outputs.

## Use

The weight files are on Hugging Face, one repository per model named as the model (`dbis.repo(name)`, for example
[Illaitar/dbisf-NAFNet-64](https://huggingface.co/Illaitar/dbisf-NAFNet-64)), each with the terms of its source and a
card; all of them are in the [dbis collection](https://huggingface.co/collections/Illaitar/dbis-6b09930cbf629d8d89c027ba). The package (`dbis`, distributed as `deblurring-is-solved`)
installs with pip from this repository, and `python -m dbis.run` restores an image, a folder of frames or a video file:

```bash
pip install "deblurring-is-solved @ git+https://github.com/illaitar/deblurring-is-solved"   # BSSTNet: "deblurring-is-solved[bsst] @ ..."
python -m dbis.run dbisf-NAFNet-64 blurry.png sharp.png
python -m dbis.run dbisf-DSTNetPlus-L blurry_frames/ sharp_frames/     # a folder: one clip, frames in name order
python -m dbis.run dbisf-Turtle blurry.mp4 sharp.mp4                     # video files through ffmpeg (on PATH)
```

`dbis.load(name)` fetches a missing weight file on first use and checks its sha256 (`dbis.download(name)` does only
that; `fetch=False` turns it off). The files are kept in `$DBIS_WEIGHTS` if it is set, else in `weights/dbis/` of a
checkout of this repository, else in `~/.cache/dbis`; `$HF_ENDPOINT` selects a mirror of Hugging Face. To make the files
yourself from the authors' checkpoints:

```bash
pip install torch numpy safetensors zstandard   # BSSTNet: torchvision too
# NAFNet-GoPro-width64.pth from https://github.com/megvii-research/NAFNet (MIT License)
python -m dbis.convert NAFNet-64 NAFNet-GoPro-width64.pth      # writes weights/dbis/, about 10 s
# BSST_gopro.pth from https://github.com/huicongzhang/BSSTNet, raft-things.pth from https://github.com/princeton-vl/RAFT
python -m dbis.convert BSSTNet BSST_gopro.pth raft-things.pth
# DSTNetPlus_L_gopro.pth from https://github.com/sunny2109/DSTNet-plus/releases/tag/v0.1.0 (Apache License 2.0)
python -m dbis.convert DSTNetPlus-L DSTNetPlus_L_gopro.pth
# GoPro_Deblur.pth from https://github.com/Ascend-Research/Turtle (MIT License)
python -m dbis.convert Turtle GoPro_Deblur.pth
```

Every model has the same inference call, `net.restore(frames, **options)` ([`dbis/restore.py`](dbis/restore.py)):

```python
import dbis
net = dbis.load("dbisf-BSSTNet")          # any model: dbis-/dbisf-NAFNet-64, dbis-/dbisf-BSSTNet; eval mode, on the GPU
out = net.restore(frames)                 # frames: a clip [T, 3, H, W], clips [B, T, 3, H, W] or an image [3, H, W],
                                          # uint8 or float in [0, 1], any device, any H and W -> the same shape
out = net.restore(frames, tile=512, tile_overlap=32, batch=4, precision="fp16", compile=True, out="uint8")
print(net.defaults)                       # the model's own setting of every option
```

| Option | Meaning | NAFNet-64 | BSSTNet | DSTNet+ L | Turtle |
|---|---|---|---|---|---|
| `tile` | tile size, int or (h, w) | whole frame | 256 only (its patches) | whole frame only | 320, square, a multiple of 8; `None`: whole frame |
| `tile_overlap` | pixels shared by neighbouring tiles, int or (rows, columns); outputs averaged | 32 | 64 | not used | dbis 192, dbisf 80 |
| `window` | frames per window, laid out as the benchmark runs the authors' code | not used | dbis 48, dbisf 24, overlap 4 | 50, clips that do not overlap | `None`: one pass over the clip; n: the history starts again every n frames |
| `batch` | tiles per network call (no tiles: frames per call) | 1 | 1 dbis, 14 dbisf | 10 frames through the full-size layers | 1 dbis, 15 dbisf (each tile keeps its own history) |
| `precision` | `"fp32"` or `"fp16"` (autocast) | fp32 | fp16 | dbis fp32, dbisf fp16 | dbis fp32, dbisf fp16 (on the CPU use fp32) |
| `compile` | `False`, `True` or a `torch.compile` mode, on the first call | the network | its heavy blocks | its propagation blocks (dbisf: the dynamic conv with max-autotune) | the network |
| `out` | `"float"` (float32, not clamped) or `"uint8"` (rounded as the benchmark) | float | float | float | float |

Turtle is causal: frame t is restored from blurry frame t and the history of the frames before it, so it also runs as
a stream, `out, state = net.step(frame, state)` (`state=None` at the first frame), with the same options.

A long clip can stay on the CPU: `restore` moves one window (or frame) at a time to the GPU and writes the output on
the device of `frames`. Frames are padded where a network needs it (BSSTNet: multiples of 16, at least 256) and the
padding is cropped. `net(x)` is the bare network (NAFNet: [B, 3, H, W]; BSSTNet: one window [B, T, 3, H, W]).

The converter checks the md5 of each checkpoint, reads it with `weights_only=True`, folds it and writes each model's
file with a JSON card. It compares the file's sha256 with the one recorded in `dbis/__init__.py`; another BLAS can
change the last bit of a folded bias, which the card reports. The hosted files are the ones this converter wrote.

dbisf-BSSTNet takes 18 GiB of GPU memory at 1280x720 (the authors' code 9.4 GiB, the first dbisf with 48-frame windows
36 GiB). On a smaller card give fewer patches per call, `net.restore(frames, batch=4)` for example; it changes only
float rounding.

## Training

`python -m dbis.train` ([`dbis/train.py`](dbis/train.py)) trains every model with one script: from the released
weights (a fine-tune, the default) or from scratch, on one GPU or several, on any folder of blurry and sharp frames.
A plan fitted to the machine is made once, and the training follows it:

```bash
python -m dbis.train dbisf-Turtle --plan --data DIR --out runs/turtle --cache runs/turtle/cache   # once, a minute or two
torchrun --nproc-per-node 8 -m dbis.train dbisf-Turtle --data DIR --out runs/turtle                # or python -m ... on one GPU
```

The download dialog on the site writes this as a script file for any model.

**Data** ([`dbis/data.py`](dbis/data.py)). The folder layout is found by itself: a blurry frame is an image whose path
has `blur`, `Blur`, `input_noise` or `input` in it, and its sharp frame is the same path with `sharp`, `Sharp`,
`target`, `GT` or `gt` there instead, so GoPro, REDS, BSD, DVD, RealBlur, GS-Blur and MIORe work as downloaded
(`python -m dbis.data find DIR` shows what it finds). The frames of one folder are one sequence; video models train on
clips of consecutive frames of one sequence. `--list` and `--exclude` keep or drop frames and sequences by name.
A training sample needs only a crop, but a PNG is decoded whole: 7 to 9 crops (blurry and sharp) per second and process
at 1920x1080, 17 to 21 at 1280x720. The cache (`python -m dbis.data cache DIR CACHE`, or the plan when decoding cannot
keep up) decodes every frame once into an uncompressed `.npy` file that is opened as a memory map, so a crop reads only
its rows: 198 crops per second and process at 1920x1080, 22 times the fastest decoder. `python -m dbis.data bench DIR`
times every reader on a machine.

**One recipe per model, the authors' own.** The numbers are those of each method's GoPro training config at the commit
we ported (`RECIPES` in `dbis/train.py` names the config file of each). A fine-tune from the released weights takes
10 % of the steps at 10 % of the learning rate unless `--steps` and `--lr` say otherwise.

| | NAFNet-64 | BSSTNet | DSTNet+ L | Turtle |
|---|---|---|---|---|
| crop, frames per clip | 256, 1 | 256, 24 | 256, 15 | 192, 5 |
| global batch | 64 | 8 | 8 | 16 |
| steps | 400k | 600k | 600k | 200k |
| AdamW lr, betas, weight decay | 1e-3, (0.9, 0.9), 1e-3 | 4e-4, (0.9, 0.99), 0 | 2e-4, (0.9, 0.999), 1e-4 | 4e-4, (0.9, 0.99), 0 |
| schedule | cosine to 1e-7 | 1000 warmup steps, cosine to 1e-7; lr x0.1 flow completion, x0.25 offsets, x0.5 soft split | cosine to 1e-7 | cosine to 1e-7 |
| loss | PSNR loss | L1 over all frames | Charbonnier + 0.1 x mean modulus of the rfft2 difference | L1 per frame |
| gradient-norm clip | 0.01 | 0.01 | 0.01 | none |
| kept fixed | | RAFT (its flows without gradients) | | |

What is the same for all, and was not in every original: bfloat16 autocast by default (NAFNet and DSTNet+ trained in
float32, BSSTNet and Turtle in float16 with a loss scaler; `--precision fp16` adds the scaler, `fp32` turns autocast off)
and one set of augmentations, the 8 flips and transposes chosen uniformly with the crop at the same place in every frame
of a clip and in both frames of a pair (`--reverse` also plays clips backwards; no original did). Each network's
`train_forward` is the function `restore` runs: checked on GoPro frames, NAFNet, BSSTNet and Turtle give the same
outputs bit for bit, DSTNet+ within 8e-7 (the dynamic convolution's rounding).

**Losses** ([`dbis/losses.py`](dbis/losses.py)): the training losses of the benchmark's methods, each as its authors'
code computes it and checked against that code; `--loss "l1_mean+0.1*fft_l1_full"` trains with any composition.

**The plan.** `--plan` measures, on one GPU of the machine, what the training will use:

- the largest batch per GPU that fits, by real training steps (forward, backward and the AdamW step), with and without
  activation checkpointing (each block of `net.train_checkpoint` recomputed in the backward pass), keeping the faster;
- bfloat16 where the GPU has it, else float16 with a loss scaler;
- with `--compile-probe`, whether `torch.compile` is faster;
- every frame reader on the data, and the data workers per GPU that keep up with the GPU, building the cache when
  decoding cannot.

It writes `OUT/plan.json`; the global batch stays the recipe's (gradient accumulation), so a plan changes the speed,
not the training. On one A100 80 GB, NAFNet-64 on GoPro: batch 64 in one step, bfloat16, the torchvision reader with 4
workers, 57.5 samples per second.

**Feeding the GPU.** After `torch.compile`, one A100 trains NAFNet-64 on about 106 crops per second, more than a few
PNG decoders give. Three ways round it, measured with 2 data workers on GoPro (crops of 256, pairs per second):

| Reader | Pairs per second | |
|---|---:|---|
| PNG (torchvision) | 42 | |
| PNG, `--echo 2` / `4` / `8` | 80 / 151 / 274 | every clip read feeds 2, 4 or 8 batches, each time with a new crop and flips |
| cache (`--cache`) | 557 | uncompressed `.npy` memory maps |
| `--reader ram` | 1,477 | every frame in memory (per GPU process) |
| `--reader gpu` | 74,000 | every frame on the GPU (GoPro: 10.8 GiB), the batch cut there in 0.86 ms, no data workers |

With PNG and 2 workers a 15-minute NAFNet-64 run makes 511 steps (the GPU busy 36 % of the time, GoPro 25.70 dB);
with `--echo 4` 1,465 steps (99 %, 26.28 dB). Echo factors up to 8 cost nothing we could measure (26.24 dB against
25.94 and 25.97 without echo, at the same number of steps). Two crops of one read never share a batch.

**Options that change the training** (off by default). Measured on NAFNet-64 from scratch, one A100, 15 minutes,
GoPro test PSNR (the benchmark's evaluation, 1,111 frames):

| Option | What it does | Steps | PSNR | With EMA |
|---|---|---:|---:|---:|
| (none, batch 64) | the recipe's batch | 1,480 | 26.16 ± 0.25 (11 runs) | ±0.04 |
| `--batch-per-gpu 8 --accum 1` | batch 8 | 9,101 | 26.98 | 27.41 |
| `--noise-every 10 --adapt-batch 8,64` | batch 8, raised with the gradient noise scale | 3,497 | 26.95 | 27.04 |
| `--shuffle perm` | random reshuffling: a new permutation of the clips every epoch | 1,481 | 26.16 | 26.10 |
| `--rho 2` / `--rho 4` | RHO-LOSS (Mindermann et al. 2022): train on the half / quarter of 2x / 4x candidates whose loss is highest relative to the released weights' loss | 1,054 / 821 | 26.35 / 25.78 | 26.30 / 26.12 |

- `--ema 0.99,0.999` keeps averages of the weights (`OUT/ema_<decay>.pt`, loadable like `last.pt`; the decay ramps
  as min(d, (1 + t) / (10 + t)), so after 1,500 steps every decay above 0.994 is the same). `--export-ema` exports the
  first average at the end. The average helps when the steps are many and noisy (batch 8: +0.43 dB), not at batch 64.
- `--noise-every N` measures the gradient noise scale B = tr(Σ) / |G|² (McCandlish et al. 2018, "An Empirical Model of
  Large-Batch Training") from the gradients of the two halves and of the whole batch, every N steps, into
  `OUT/noise.tsv`. Below B, doubling the batch halves the steps a result needs; above it, samples go to waste. On
  NAFNet-64 it grows from 1-3 at the start to 10-20 after 15 minutes, so early on the recipe's batch 64 spends most of
  its samples for nothing, and batch 8 wins. `--adapt-batch MIN,MAX` starts at MIN and raises the batch to the largest
  power of two not above the measured scale (after 3 measurements in a row, never down); the run above used an earlier,
  jumpier rule, which reached 64 after 9 minutes. The scale keeps growing over a full run, so these numbers say
  nothing yet about 400k steps.
- Runs with the same data and initial weights differ by up to 0.8 dB at this length (the GPU's nondeterministic
  kernels), so differences below about 0.5 dB in this table are noise.
- `--adapt-batch`, `--rho` and `--shuffle perm` need `--reader gpu`; `--noise-every` and `--adapt-batch` run on one GPU
  without gradient accumulation.

**Our table of settings** (`python -m dbis.train MODEL --sweep` -> [`dbis/train_configs.json`](dbis/train_configs.json),
which the plan starts from): for each model the recipe's crop and frames and smaller ones, in bfloat16, float16 and
float32, with and without checkpointing, the largest batch up to 64 on an A100 80 GB, the peak memory as fixed + per
sample, and the time per sample. At the recipe's crop, in bfloat16:

| Model | Crop, frames | Per sample | Per sample, checkpointing | Time per sample (checkpointing) | One sample fits in |
|---|---|---:|---:|---:|---:|
| NAFNet-64 | 256, 1 | 1.12 GiB | 0.30 GiB | 17 ms (22 ms) | 4 GB |
| DSTNet+ L | 256, 15 | 8.6 GiB | 4.1 GiB | 317 ms (299 ms) | 8 GB |
| Turtle | 192, 5 | 14.8 GiB | 1.2 GiB | 300 ms (317 ms) | 4 GB |
| BSSTNet | 256, 24 | 34.8 GiB | 8.5 GiB | 1256 ms (2279 ms) | 16 GB |

So with checkpointing every model trains with its own recipe on a 16 GB card (the last column: fixed + one sample
with checkpointing, with room for the CUDA context); only the time per step grows.

Output in `--out`: `plan.json`, `last.pt` (a stopped run resumes from it), `log.tsv`, `best.pt` (with `--val`: the best
validation PSNR, frames restored with `net.restore` and rounded to 8 bits as the benchmark scores them) and, at the end,
the weight files of the model in the stores of `dbis.MODELS` (`dbis-<source>.safetensors`, `dbisf-<source>.wpk`), which
`dbis.load(name, weights_dir=OUT)` reads.

## Layout of a method and its encoder

Every method has the same folder, `dbis/<method>/`:

```
dbis/<method>/
  __init__.py     credits and licenses; exports the network, its encoder and fold
  net.py          the full network (restore() comes from dbis/restore.py) and fold(), which turns the authors'
                  checkpoint into our weights
  encoder.py      the encoder: the full network subclasses it, so its weights keep their names and load on their own
  blocks/         the building blocks, one file per part
```

The encoder of a model loads alone from the model's weight file, without the rest of the network:

```python
enc = dbis.load_encoder("dbisf-NAFNet-64")   # or any other model
feats = enc(x)                              # x: [N, 3, H, W] in [0, 1] -> a list of feature maps, finest first
enc.feature_info                            # channels, reduction and source module of each (as timm's features_only)
```

| Model | Encoder | Features (channels at reduction) | Parameters |
|---|---|---|---:|
| NAFNet-64 | intro conv, the four encoder stages, the middle blocks | 64 at 1, 128 at 2, 256 at 4, 512 at 8, 1024 at 16 | 62.5 M of 67.8 M |
| BSSTNet | the per-frame encoder of BSST, before the temporal propagation | 64 at 1, 64 at 2, 192 at 4 | 1.9 M of 43.7 M |
| DSTNet+ L | the first conv, the Haar split and the high-frequency branch, before the propagation | 96 at 1, 96 (low band) and 288 (high bands) at 2 | 0.06 M of 14.1 M |
| Turtle | the input conv and the three encoder levels, which carry no history, and the last downsampling | 64 at 1, 128 at 2, 256 at 4, 512 at 8 | 9.0 M of 59.1 M |

NAFNet's encoder pads the input to a multiple of 16 as the network does; BSSTNet's needs H and W multiples of 4. Both
are the same modules with the same weights as in the full network (checked: equal features).

## What changed

**dbis-NAFNet-64**
- The network is [`dbis/nafnet/`](dbis/nafnet) (network, encoder, NAFBlock with TLC) instead of three files inside the
  basicsr package. Its import takes 6 ms instead of 1.6 s.
- The scale and shift of both LayerNorms of a block are folded into the next 1x1 conv, the residual scales beta and
  gamma into conv3 and conv5. The folding is exact up to float rounding and removes 216 of 664 tensors.
- TLC (Chu et al., ECCV 2022) averages the channel-attention input over a window 1.5 times the training crop. The
  authors find the window sizes by running a random image through the network when it is built; here they are
  computed from the training crop.
- The network is built on the meta device and takes the weights as they are read, so there is no random
  initialisation and no copy.

**dbisf-NAFNet-64**
- Fast TLC. Most of TLC's cost is the attention's 1x1 conv, which TLC applies to a full-resolution map in every block.
  The window mean is now taken on the map pooled into 2x2 to 8x8 cells (at least 12 cells per window). The 1x1 conv
  runs on that small map before the bilinear upsampling, which is the same function because both are linear and the
  upsampling weights sum to 1. Cost on GoPro: +0.003 dB; the authors' own fast TLC (`fast_imp`) costs −0.019 dB.
- Weights rounded per output channel to std(channel) / 48 and entropy coded with [`dbis/wpack.py`](dbis/wpack.py)
  (tier `qc`): 66 MB. The first dbisf used one step per tensor, std / 128, at 76 MB; one step per tensor at 67 MB
  (std / 64) costs −0.017 dB on 56 GoPro frames, one step per channel at 65 MB −0.006 dB.
- Together: 26.6 ms instead of 34.9 (−24 %), 0.7 ms above NAFNet without TLC, which is 0.63 dB worse on GoPro.

**dbis-BSSTNet**
- BSST, its RAFT optical flow and the authors' inference (flows on half-size frames, then BSST on 256 px patches with
  64 px overlap under fp16 autocast) are [`dbis/bsst/`](dbis/bsst), plain PyTorch, instead of the basicsr package
  with mmcv. With one patch per BSST call, as the authors, the flows, every patch and whole windows are bit for bit the
  authors' in fp16 and in fp32 (checked on two GoPro windows); on the four test sets dbis-BSSTNet gives the authors'
  outputs, black tiles included.
- The checkpoint holds modules that the forward pass never calls (8.3 M values); they are dropped, 175 MB instead of
  209 MB for both checkpoints. Index tables and coordinate grids are rebuilt once per shape, and the CPU syncs of the
  authors' code (a mirror check whose flag is never read, the nonzero() calls of the sparse attention) are gone.
- In fp16 eager a window takes 30.6 s instead of 34.9 (−12 %). `torch.compile` of the whole forward fails on this code
  (an AssertionError of PyTorch 2.10, "incorrect out shape after application of ViewMeta sequence"), so in the harness
  it runs eager and is 20 % slower than the authors' compiled code.

**dbisf-BSSTNet**
- 24-frame windows instead of 48: half the work for every output frame (the leaderboard's sliding window, emitted at
  position 2, now looks 21 frames ahead instead of 45). On the test sets this costs 0.014 dB on GoPro and 0.021 dB on
  DVD against 48-frame windows; 32-frame windows cost less (0.003 and 0.009 dB) but save less (table below).
- RAFT flows reused between windows. The sliding window moves by one frame, so the flows of all frame pairs but the
  newest are those of the previous window; RAFT computes every pair on its own, so this changes nothing but the batch
  size of its kernels. `net.restore()` reuses the pairs where its windows overlap.
- TF32 for the float32 matrix products: torchvision's deformable convolution runs in float32 inside autocast and its
  GEMMs, like RAFT's correlation, use full float32 under PyTorch's default flags. The sparse attention and the Swin
  window attention call `scaled_dot_product_attention` (the same function as the authors' matrix products and softmax).
- The heavy blocks compiled one by one (`net.restore(frames, compile=True)`: encoder, decoder, propagation backbones and offset
  convs, Swin reconstruction, fusion); the whole forward does not compile. 1.44 times faster in the harness (1.33 on a
  window without the flow cache), after about 3.5 minutes of compilation in torch's default mode.
- 14 patches per BSST call: a 720p window has 28 patches, so two calls instead of 28. Inside a call the recurrent
  parts (flow completion, the four propagation branches) run on all 14 patches at once and the feed-forward stages in
  groups of 4. The computation of each patch is the same; batched kernels only round differently.
- Weights rounded per output channel to std(channel) / 48 (tier `qc` of `dbis/wpack.py`): 42 MB, 4.9 times smaller
  than the two checkpoints.
- Patches that are not finite are redone in float32, or kept as the input (above). The check costs one sync per call.
- Together: 4.6 s per frame in the harness instead of 25.6 s for the authors' compiled code (5.6 times faster), 6.6 s
  instead of 34.9 s without compilation. The first dbisf of the same day (48-frame windows, patches in batches,
  eager) took 16.0 s.

What we tried for dbisf-BSSTNet. Time: one GoPro window on an A100, fp16 eager, against the first dbisf (16.1 s);
with the flow cache, the window after the first one. Below the bold row, every line is on top of TF32, the attention
call and the flow cache. Rule: the worst mean difference to the authors over the four test sets, on all frames and on
the frames without black tiles (it must stay above −0.05 dB).

| Change | Faster | Worst difference, dB | |
|---|---:|---:|---|
| TF32 for float32 matrix products | 1.09× | | |
| deformable convolution in float16 (torchvision) | 1.08× | | |
| deformable convolution as grid_sample + one float16 1x1 conv | 1.06× | | |
| `scaled_dot_product_attention` | 1.02× | | |
| TF32 + `scaled_dot_product_attention` | 1.12× | −0.027 | passes |
| + flow cache | 1.22× | −0.027 | passes |
| + 32-frame windows | 1.82× | −0.036 | passes |
| **+ 24-frame windows** | **2.44×** | **−0.034** | **dbisf** |
| patch overlap 24: 18 patches instead of 28 | 1.91× | −0.067 (GoPro) | fails |
| 32-frame windows, overlap 48: 24 patches | 2.12× | −0.041 | passes |
| 32-frame windows, overlap 24 / 64 (rows / columns): 21 patches | 2.39× | −0.046 | passes |
| 24-frame windows, 24 patches | 2.83× | −0.055 (REDS) | fails |
| 24-frame windows, 21 patches | 3.19× | −0.051 (DVD) | fails |
| 24-frame windows, heavy blocks compiled (in the harness) | 1.44× more | −0.034 | passes |
| channels_last convolutions | does not run with the patch code's views | | |

**dbis-DSTNetPlus-L**
- The network is [`dbis/dstnetplus/`](dbis/dstnetplus), plain PyTorch, instead of the basicsr package with CuPy. On the
  first 50-frame GoPro clip its outputs are bit for bit the authors' (float32, as the benchmark runs them).
- The authors apply the dynamic depthwise conv (a 3x3 kernel predicted for every pixel and channel) with a CuPy kernel
  that has no float16. Here it is nine `addcmul`. The kernel adds the products with fused multiply-adds; plain
  `addcmul` rounds twice, 1 unit in the last place per call, which the 50-frame recurrence grows to 1.4e-3. dbis forms
  the product in float64 and rounds the sum to float32 once, which matches the kernel bit for bit and costs 14 % of the
  time (3.45 s per frame in the harness against the authors' 3.01 s).
- The fixed Haar weights, which the authors keep as a parameter, are rebuilt in code; the weight file holds the rest.

**dbisf-DSTNetPlus-L**
- Float16 autocast, which the CuPy kernel prevented, and the dynamic conv with plain `addcmul`: 1.65 times faster.
- The propagation blocks (fusion, dynamic conv, residual blocks) compiled with `torch.compile`: 2.3 times faster
  than dbis on a 27-frame window.
- Only the emitted frame is computed (`net(x, keep=(0, 1))`). The leaderboard's sliding window emits its oldest frame
  and the forward propagation is causal, so one forward step instead of 27 gives the same output for that frame.
- The encoder folded into two convs. The first conv (3x3), the Haar split (2x2, stride 2) and the first 1x1 conv of the
  high bands are linear with no activation between them, so the low band and the high bands are each one 4x4
  stride-2 conv of the frame. The 96-channel full-resolution map (177 MB per 720p frame in float16) is never formed:
  the encoder of a 27-frame window takes 11 ms instead of 161 ms, and the high bands are made only for the emitted
  frame. In float32 the outputs stay within 2e-6 of the plain path.
- The propagation in channels_last, its dynamic conv compiled with max-autotune: one propagation step in 20.1 ms
  instead of 23.1 ms. Of these, the 25 residual blocks take 14.9 ms at 128 TFLOPS, the speed of a matrix multiply of
  the same size on the A100. In float16 the 28 steps of a window therefore cannot take less than about 0.42 s.
- Weights rounded per output channel to std(channel) / 128: 16 MB, 3.5 times smaller than the checkpoint. The step
  std / 48 of the other dbisf models costs 0.031 dB on GoPro for this 14 M-parameter network.
- Clips stay 50 frames. Shorter clips would be faster but cost too much (27 frames: −0.043 dB on GoPro, −0.073 dB on
  BSD; 10 frames: −0.24 dB on GoPro).
- Together: 0.58 s per frame in the harness instead of 3.01 s (5.2 times faster), 3.8 GiB of GPU memory instead of
  18.8 GiB, loading in 0.60 s instead of 2.78 s. The first dbisf (25.11, without the folded encoder, channels_last and
  max-autotune) took 0.76 s and 12.3 GiB.

What we tried for dbisf-DSTNetPlus-L. Time: one 27-frame GoPro window on an A100 against dbis (3.67 s); rule: the four
test sets with clips of 50 frames unless stated.

| Change | Faster | Worst difference, dB | |
|---|---:|---:|---|
| dynamic conv with plain `addcmul` (float32) | 1.11× | | |
| float16 autocast | 1.65× | | |
| float16, channels_last | 1.53× | | no gain |
| float16, propagation blocks compiled | 2.31× | | |
| only the emitted frame (keep), float16, compiled | 4.14× | | the same output for that frame |
| + the encoder folded into two convs | 5.80× | | the same maths |
| + the propagation in channels_last | 5.83× | | |
| + the dynamic conv compiled with max-autotune | 6.29× | −0.008 | **dbisf** (26.11) |
| CUDA graphs over a whole propagation step | | | no gain (0.5 %) |
| channels_last for the whole network | | | slower (the encoder) |
| fused dynamic conv: the kernel values made at the three low resolutions and upsampled inside the application, so the 864-channel kernel map is never written | | | slower: 9.5 ms compiled from PyTorch, 6.2 ms with a Triton kernel (a second Triton kernel was slower still), against 4.4 ms |
| float16, float16 weights (24 MB) | | −0.0001 | passes |
| float16, weights std / 128 (16 MB) | | −0.008 | weights of **dbisf** |
| float16, weights std / 48 (14 MB) | | −0.031 | passes, less margin |
| clips of 27 / 20 / 15 / 10 frames (float16 weights) | | −0.073 / −0.087 / −0.135 / −0.235 | fails |

**dbis-Turtle**
- The network is [`dbis/turtle/`](dbis/turtle) (network, encoder, blocks), plain PyTorch without einops and basicsr:
  the authors' `Turtle_t1` in its GoPro configuration. Only the current frame enters this configuration, so the
  previous blurry frame, which the authors' forward also takes, is not passed.
- The local attention mask of SAB (token pairs at L1 distance 4 or less on its 20x20 grid) is made once and kept on
  the GPU. The authors build it on the CPU with `torch.cdist` and copy it to the GPU in every call, 135 times per 720p
  frame. The caches of every tile stay on the GPU; the authors move them to the CPU and back after each frame.
- The authors' outputs bit for bit: the network alone over four steps (its output and all eight caches), and every
  frame of the four video test sets.
- 724 ms per frame instead of 761 ms in the same mode, 1.98 s instead of 2.24 s in fp16 without compilation.

**dbisf-Turtle**
- All tiles of a frame in one network call. Every tile keeps its own history, so this is the same computation with a
  larger batch: in float32 2.0 s per frame instead of 4.8 s, the output values within 2e-3 of the authors' (76 to 81
  dB between the two).
- Float16 autocast and `torch.compile`: 0.57 s per frame with the authors' 45 tiles, within 0.006 dB of the authors
  on every set (with dbisf's rounded weights). SAB's top-5 and local masks are applied in one product, the same values.
- 15 tiles of 320 px with 80 px overlap at 1280x720 (6 at 640x480) instead of 45 with 192 px (12). The authors'
  tiling averages 5 predictions per pixel and costs three times the compute; dbisf averages 1.7. This is all of
  dbisf's difference: −0.028 dB on GoPro and REDS, −0.021 on BSD, −0.006 on DVD.
- Weights rounded per output channel to std(channel) / 128: 68 MB instead of 237 MB. The step std / 48 (58 MB) fails:
  −0.067 dB on BSD.
- Together: 199 ms per frame in the harness instead of 761 ms (3.8 times faster; 24 times the authors' code in
  float32), 6.4 GiB of GPU memory instead of 7.9 GiB.

What we tried for dbisf-Turtle. Time: 30 consecutive GoPro frames on an A100, one frame per step (dbis: 4.78 s per
frame); rule: the four test sets.

| Change | Per 720p frame | Worst difference, dB | |
|---|---:|---:|---|
| all 45 tiles in one call (float32) | 2.02 s | | |
| float16 autocast | 1.19 s | −0.001 | |
| float16, compiled (mode default) | 571 ms | −0.001 | |
| float16, compiled with max-autotune | 534 ms | | 59 minutes of compilation |
| float16, compiled, channels_last or cudnn.benchmark | 570 ms | | no gain |
| float16, compiled, 45 tiles, weights std / 128 | 577 ms (harness) | +0.006 | |
| 18 tiles (overlap 128), weights std / 128 | 317 ms | −0.023 | passes; 18 tiles take 1.6 times as long as 15 at every batch size |
| 15 tiles (overlap 64) | 196 ms | −0.038 | passes, less margin |
| 15 tiles (overlap 80), weights std / 128 | 199 ms (harness) | −0.028 | **dbisf** |
| 15 tiles (overlap 64), weights std / 48 | | −0.067 | fails |
| 12 tiles | about 165 ms | −0.220 | fails |
| whole frames (the authors' do_patched=False) | | −1.09 (authors' code) | fails; does not compile in this env |

**Credits.** BSSTNet (Zhang, Xie and Yao, CVPR 2024; MIT License stated in its README) reuses code of ProPainter
(Zhou et al., ICCV 2023; S-Lab License 1.0, non-commercial use only) for its sparse transformer and flow completion,
of Shift-Net (Li et al., CVPR 2023), of RAFT (Teed and Deng, ECCV 2020; BSD 3-Clause) and of BasicVSR++ and RVRT.
[`dbis/bsst/`](dbis/bsst) re-implements these parts; its `__init__.py` lists what comes from where, and the header of
every file in `blocks/` names its source and license.

DSTNet+ (Pan, Sun, Xu, Dong and Tang, TPAMI 2025; Apache License 2.0) is re-implemented in [`dbis/dstnetplus/`](dbis/dstnetplus);
the CuPy kernel of its dynamic conv is replaced by PyTorch ops.

Turtle (Ghasemabadi, Janjua, Salameh and Niu, NeurIPS 2024; MIT License, Copyright (c) 2024 Huawei Technologies Co.,
Ltd.) is re-implemented in [`dbis/turtle/`](dbis/turtle).

Not in the leaderboard yet: dbis and dbisf are not rows of the tables.
