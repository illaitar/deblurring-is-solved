# dbis and dbisf

Methods of the benchmark re-implemented in this repository, in the package `dbis/`.

- **dbis-*method*** is our code and weight file for the authors' network: one file of plain PyTorch per architecture,
  no training framework, weights as safetensors. It gives the same outputs as the authors' code: the largest
  difference of an output value is about 1e-5 and the mean PSNR on the test set is equal to the fourth decimal.
- **dbisf-*method*** is our fast and light version of it. Its mean PSNR is at most 0.05 dB below the authors' code on
  every test set of the benchmark; the tables below give every set.

## Models

| Model | Weights | Load | Per 720p frame | GoPro PSNR, dB |
|---|---:|---:|---:|---:|
| NAFNet-64, authors' code (TLC on) | 272 MB `.pth` | 3.06 s | 34.9 ms | 33.7103 |
| **dbis-NAFNet-64** | 271 MB `.safetensors` | 0.70 s | 33.5 ms | 33.7103 |
| **dbisf-NAFNet-64** | 66 MB `.wpk` | 0.77 s | 26.6 ms | 33.7041 (−0.006) |
| BSSTNet, authors' code (BSST + RAFT) | 209 MB, two `.pth` | 3.50 s | 25.7 s | 35.9662 |
| **dbis-BSSTNet** | 175 MB `.safetensors` | 1.66 s | 30.4 s | 35.9662 |
| **dbisf-BSSTNet** | 42 MB `.wpk` | 1.71 s | 16.0 s | 35.9562 (−0.010) |

Load: a fresh Python process, from CUDA ready to the network with its weights on the GPU (importing torch and starting
CUDA take another 2.3 s for every model). Time: the benchmark's latency harness on one A100, 1280x720, batch 1, the
three versions of a method on the same node. NAFNet: fp16 autocast, `torch.compile` max-autotune, CUDA graphs. BSSTNet:
as in the leaderboard, a 48-frame window for every output frame, and each version in its fastest mode that runs: the
authors' code with fp16 autocast and `torch.compile` (18.5 minutes of compilation), ours fp16 eager (`torch.compile`
fails on dbis-BSSTNet and does not finish within the harness's 30 minutes on dbisf-BSSTNet). In fp16 eager the three
take 34.9, 30.4 and 16.0 s. PSNR: all 1111 GoPro test frames, 8-bit outputs. Every number is in
[`results/dbis/NAFNet-64.json`](results/dbis/NAFNet-64.json) and [`results/dbis/BSSTNet.json`](results/dbis/BSSTNet.json).

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

**BSSTNet** (the four video test sets; the authors' code as the leaderboard runs it, fp16 autocast, the leaderboard's
windows of 48 frames)

| Test set | Frames | Authors, dB | dbisf-BSSTNet | Frames with black patches | dbisf without them |
|---|---:|---:|---:|---:|---:|
| GoPro | 1111 | 35.9662 | −0.010 | 0 | −0.010 |
| REDS val | 3000 | 24.3901 | +3.351 | 466 | −0.027 |
| BSD 2ms/16ms | 3000 | 22.7700 | +1.808 | 376 | +0.048 |
| DVD | 1000 | 32.6798 | −0.013 | 0 | −0.013 |

On REDS and BSD the authors' code writes NaN into some 256 px patches, which become black tiles in the 8-bit output.
Two causes: float16 overflow, and the recurrent backward propagation, whose features grow out of the GoPro range until
a LayerNorm overflows in float32 as well (the leaderboard's BSD row is the authors' code in float32, 23.7607 dB, and
still has such tiles). dbisf-BSSTNet runs a patch whose fp16 output is not finite again in float32, without autocast
(REDS 117 patches, BSD 29), and keeps the blurry input where the float32 result is not finite either (115 and 14 of
them), so no frame has a black tile. This is why dbisf scores higher on these two sets; on the other frames it is
within 0.05 dB of the authors.
Without the redo the rounded weights give −0.010, +0.070, −0.027 and −0.013 dB, with the black tiles in both outputs.

## Use

```bash
pip install torch numpy safetensors zstandard   # BSSTNet: torchvision too
# NAFNet-GoPro-width64.pth from https://github.com/megvii-research/NAFNet (MIT License)
python -m dbis.convert NAFNet-64 NAFNet-GoPro-width64.pth      # writes weights/dbis/, about 10 s
# BSST_gopro.pth from https://github.com/huicongzhang/BSSTNet, raft-things.pth from https://github.com/princeton-vl/RAFT
python -m dbis.convert BSSTNet BSST_gopro.pth raft-things.pth
```

```python
import dbis
net = dbis.load("dbisf-NAFNet-64")        # or "dbis-NAFNet-64"; eval mode, on the GPU
y = net(x)                                # x: [B, 3, H, W] in [0, 1], any size

net = dbis.load("dbisf-BSSTNet")          # or "dbis-BSSTNet"
y = net(x)                                # x: [B, T, 3, H, W] in [0, 1] on the GPU, H and W multiples of 16, >= 256
```

The converter checks the md5 of each checkpoint, reads it with `weights_only=True`, folds it and writes each model's
file with a JSON card. It compares the file's sha256 with the one recorded in `dbis/__init__.py`; another BLAS can
change the last bit of a folded bias, which the card reports. We do not host weight files yet.

dbisf-BSSTNet sends 14 patches through BSST at once, which needs 36 GiB of GPU memory (the authors' code: 9.4 GiB).
On a smaller card give fewer patches per call, `net(x, patch_batch=3)` or `net.patch_batch = 3`; it changes only
float rounding. One 720p window of 48 frames on an A100, fp16 eager, weights included:

| patch_batch | 1 | 2 | 3 | 4 | 7 | 14 |
|---|---:|---:|---:|---:|---:|---:|
| Time, s | 30.6 | 20.7 | 18.6 | 17.3 | 16.6 | 16.0 |
| Peak memory, GiB | 8.1 | 13.6 | 18.7 | 23.8 | 27.4 | 35.9 |

## What changed

**dbis-NAFNet-64**
- The network is one file, [`dbis/nafnet.py`](dbis/nafnet.py) (NAFBlock, U-Net, TLC), instead of three files inside
  the basicsr package. Its import takes 4 ms instead of 1.6 s.
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
  64 px overlap under fp16 autocast) are one file, [`dbis/bsst.py`](dbis/bsst.py), instead of the basicsr package with
  mmcv. With one patch per BSST call, as the authors, the flows, every patch and whole windows are bit for bit the
  authors' in fp16 and in fp32 (checked on two GoPro windows); on the four test sets dbis-BSSTNet gives the authors'
  outputs, black tiles included.
- The checkpoint holds modules that the forward pass never calls (8.3 M values); they are dropped, 175 MB instead of
  209 MB for both checkpoints. Index tables and coordinate grids are rebuilt once per shape, and the CPU syncs of the
  authors' code (a mirror check whose flag is never read, the nonzero() calls of the sparse attention) are gone.
- In fp16 eager a window takes 30.4 s instead of 34.9 (−13 %). `torch.compile` fails on this code (an AssertionError
  of PyTorch 2.10, "incorrect out shape after application of ViewMeta sequence"), so in the harness it runs eager and
  is 18 % slower than the authors' compiled code.

**dbisf-BSSTNet**
- 14 patches per BSST call: a 720p window has 28 patches, so two calls instead of 28. Inside a call the recurrent
  parts (flow completion, the four propagation branches) run on all 14 patches at once and the feed-forward stages in
  groups of 4. The computation of each patch is the same; batched kernels only round differently.
- Weights rounded per output channel to std(channel) / 48 (tier `qc` of `dbis/wpack.py`): 42 MB, 4.9 times smaller
  than the two checkpoints.
- Patches that are not finite are redone in float32, or kept as the input (above). The check costs one sync per call.
- Together: 16.0 s per frame in the harness, 38 % less than the authors' compiled code and 2.2 times faster than their
  eager code. The price is GPU memory (table above).

**Credits.** BSSTNet (Zhang, Xie and Yao, CVPR 2024; MIT License stated in its README) reuses code of ProPainter
(Zhou et al., ICCV 2023; S-Lab License 1.0, non-commercial use only) for its sparse transformer and flow completion,
of Shift-Net (Li et al., CVPR 2023), of RAFT (Teed and Deng, ECCV 2020; BSD 3-Clause) and of BasicVSR++ and RVRT.
`dbis/bsst.py` re-implements these parts; its header lists what comes from where.

Not in the leaderboard yet: dbis and dbisf are not rows of the tables.
