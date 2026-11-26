---
license: other
license_name: per-file-terms-of-the-authors
license_link: LICENSE.md
library_name: pytorch
pipeline_tag: image-to-image
tags:
- deblurring
- motion-deblurring
- video-deblurring
- image-restoration
---

# dbis: weight files

Weights of **dbis** and **dbisf**, our versions of published motion deblurring methods, made from the authors' released
checkpoints by `python -m dbis.convert` of [deblurring is solved?](https://github.com/illaitar/deblurring-is-solved)
([benchmark site](https://illaitar.github.io/deblurring-is-solved/)).

- **dbis-&lt;method&gt;**: our code and weight file, the same outputs as the authors' code.
- **dbisf-&lt;method&gt;**: fast and light. Shorter run time and a smaller file, at most 0.05 dB below the authors' mean
  PSNR on every test set of the benchmark.

| File | Method | Size | GoPro PSNR, dB | Time per 720p frame | Terms |
|---|---|---:|---:|---:|---|
| `dbis-NAFNet-64.safetensors` | NAFNet-64 (ECCV 2022) | 271 MB | 33.7103 (authors 33.7103) | 33.5 ms (authors 34.9 ms) | MIT |
| `dbisf-NAFNet-64.wpk` | NAFNet-64 | 66 MB | 33.7041 | 26.6 ms | MIT |
| `dbis-BSSTNet.safetensors` | BSSTNet (CVPR 2024), with RAFT | 175 MB | 35.9662 (authors 35.9662) | 30.6 s (authors 25.6 s) | MIT; RAFT BSD-3-Clause |
| `dbisf-BSSTNet.wpk` | BSSTNet, with RAFT | 42 MB | 35.9425 | 4.6 s | MIT; RAFT BSD-3-Clause |
| `dbis-DSTNetPlus-L.safetensors` | DSTNet+ L (TPAMI 2025) | 56 MB | 35.7078 (authors 35.7078) | 3.45 s (authors 3.01 s) | Apache 2.0 |
| `dbisf-DSTNetPlus-L.wpk` | DSTNet+ L | 16 MB | 35.6999 | 0.58 s | Apache 2.0 |
| `dbis-Turtle.safetensors` | Turtle (NeurIPS 2024) | 236 MB | 34.5598 (authors 34.5598) | 724 ms (authors 761 ms) | MIT |
| `dbisf-Turtle.wpk` | Turtle | 68 MB | 34.5316 | 199 ms | MIT |

All are the GoPro-trained checkpoints. Time: the benchmark's latency harness, one A100, batch 1; BSSTNet and DSTNet+ L
with a window for every output frame, as in the leaderboard; Turtle one frame per step with its history, in tiles. Every number, the speed options and what was tried are
in [DBIS.md](https://github.com/illaitar/deblurring-is-solved/blob/main/DBIS.md) and `results/dbis/` of the repository.
Each file has a JSON card here (`<file>.json`) with its sha256, the source checkpoint and its md5, and the network's
arguments.

## Use

```bash
git clone https://github.com/illaitar/deblurring-is-solved && cd deblurring-is-solved
pip install torch numpy safetensors zstandard        # BSSTNet: torchvision too
```

```python
import dbis
net = dbis.load("dbisf-NAFNet-64")       # fetches the file from this repository on first use and checks its sha256
out = net.restore(frames)                # frames [T, 3, H, W], [B, T, 3, H, W] or [3, H, W]; the same call for every model
```

Or download them yourself into `weights/dbis/` of the repository: `hf download Illaitar/dbis --local-dir weights/dbis`.
`.safetensors` files are float32. `.wpk` files are the repository's wpack format (`dbis/wpack.py`, needs zstandard):
weights rounded per output channel and entropy coded.

## What the files contain

- The authors' tensors, renamed where our code names them differently and folded where that is exact up to float
  rounding: NAFNet's LayerNorm scale and shift and its residual scales go into the next convs; DSTNet+ drops its fixed
  Haar weights, which the code rebuilds. BSSTNet's files hold BSST and the RAFT flow network it uses.
- dbisf files: the same tensors, each weight tensor of two or more dimensions rounded per output channel to
  std(channel) / 48 (NAFNet-64, BSSTNet) or / 128 (DSTNet+ L, Turtle).

## Terms

Each file keeps the terms of the checkpoint it is made from; see `LICENSE.md` and `licenses/`. Cite the methods' papers
when you use their weights.
