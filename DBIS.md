# dbis and dbisf

Methods of the benchmark re-implemented in this repository, in the package `dbis/`.

- **dbis-*method*** is our code and weight file for the authors' network: one file of plain PyTorch per architecture,
  no training framework, weights as safetensors. It gives the same outputs as the authors' code: the largest
  difference of an output value is about 1e-5 and the mean PSNR on the test set is equal to the fourth decimal.
- **dbisf-*method*** is our fast and light version of it. A change is kept only if it costs at most 0.01 dB of mean
  PSNR on every test set it was measured on; the per-frame spread is reported next to it.

## Models

| Model | Weights | Load | ms per 720p frame | GoPro PSNR, dB |
|---|---:|---:|---:|---:|
| NAFNet-64, authors' code (TLC on) | 272 MB `.pth` | 3.06 s | 34.9 | 33.7103 |
| **dbis-NAFNet-64** | 271 MB `.safetensors` | 0.70 s | 33.5 | 33.7103 |
| **dbisf-NAFNet-64** | 76 MB `.wpk` | 0.83 s | 26.6 | 33.7103 (−0.00002) |

Load: a fresh Python process, from CUDA ready to the network with its weights on the GPU (importing torch and starting
CUDA take another 2.3 s for every model). Speed: the benchmark's latency harness on one A100 (fp16 autocast,
`torch.compile` max-autotune, CUDA graphs; 1280x720, batch 1), all models on the same node. PSNR: all 1111 GoPro test
frames, float32 with TF32 off, 8-bit outputs. Every number is in [`results/dbis/NAFNet-64.json`](results/dbis/NAFNet-64.json).

## Use

```bash
pip install torch numpy safetensors zstandard
# the authors' checkpoint: NAFNet-GoPro-width64.pth from https://github.com/megvii-research/NAFNet (MIT License)
python -m dbis.convert NAFNet-64 NAFNet-GoPro-width64.pth      # writes weights/dbis/, about 15 s
```

```python
import dbis
net = dbis.load("dbisf-NAFNet-64")        # or "dbis-NAFNet-64"; eval mode, on the GPU
y = net(x)                                # x: [B, 3, H, W] in [0, 1], any size
```

The converter checks the md5 of the checkpoint, reads it with `weights_only=True`, folds it and writes each model's file
with a JSON card. It compares the file's sha256 with the one recorded in `dbis/__init__.py`; another BLAS can change
the last bit of a folded bias, which the card reports. We do not host weight files yet.

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
  upsampling weights sum to 1. Cost: +0.003 dB on GoPro; the authors' own fast TLC (`fast_imp`) costs −0.019 dB.
- Weights rounded to std / 128 per tensor and entropy coded with [`dbis/wpack.py`](dbis/wpack.py): 9 bits per weight,
  −0.003 dB on its own. Rounding to std / 64 (8 bits, 68 MB) costs −0.014 dB and is not used.
- Together: 26.6 ms instead of 34.9 (−24 %), 0.7 ms above NAFNet without TLC, which is 0.63 dB worse.

Not measured yet: dbisf on the other test sets (only GoPro so far), so the rule "at most 0.01 dB on every test set"
holds for GoPro only. dbis and dbisf are not rows of the leaderboard yet.
