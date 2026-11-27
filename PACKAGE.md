# deblurring-is-solved

Motion deblurring networks in plain PyTorch, from the benchmark
[deblurring is solved?](https://illaitar.github.io/deblurring-is-solved/) ([code](https://github.com/illaitar/deblurring-is-solved)).
The package is `dbis`. Every model has the same calls, `dbis.load(name)` and `net.restore(frames)`, and its weight
file comes from Hugging Face on first use, checked by sha256.

- **dbis-&lt;method&gt;**: our code and weight file, the same outputs as the authors' code.
- **dbisf-&lt;method&gt;**: fast and light: a shorter run time and a smaller file, at most 0.05 dB below the authors'
  mean PSNR on every test set of the benchmark.

```bash
pip install deblurring-is-solved           # BSSTNet: pip install "deblurring-is-solved[bsst]" (torchvision)
python -m dbis.run dbisf-NAFNet-64 blurry.png sharp.png          # an image
python -m dbis.run dbisf-Turtle blurry.mp4 sharp.mp4             # a video file (ffmpeg on PATH) or a folder of frames
```

```python
import dbis
net = dbis.load("dbisf-NAFNet-64")      # eval mode, on the GPU; the weight file is fetched once into ~/.cache/dbis
out = net.restore(frames)               # frames [T, 3, H, W], [B, T, 3, H, W] or [3, H, W], uint8 or float in [0, 1]
out = net.restore(frames, tile=512, precision="fp16", out="uint8")
enc = dbis.load_encoder("dbisf-NAFNet-64")   # the encoder alone: a list of feature maps
```

## Models

All are the authors' GoPro-trained checkpoints. PSNR: the 1,111 GoPro test frames; time: one A100, 1280x720, batch 1.

| Model | Method | File | GoPro PSNR, dB | Per frame | Terms |
|---|---|---:|---:|---:|---|
| [`dbis-NAFNet-64`](https://huggingface.co/Illaitar/dbis-NAFNet-64) / [`dbisf-NAFNet-64`](https://huggingface.co/Illaitar/dbisf-NAFNet-64) | NAFNet (ECCV 2022) | 271 / 66 MB | 33.7103 / 33.7041 | 33.5 / 26.6 ms | MIT |
| [`dbis-BSSTNet`](https://huggingface.co/Illaitar/dbis-BSSTNet) / [`dbisf-BSSTNet`](https://huggingface.co/Illaitar/dbisf-BSSTNet) | BSSTNet (CVPR 2024), video | 175 / 42 MB | 35.9662 / 35.9425 | 30.6 / 4.6 s | MIT; RAFT BSD-3-Clause |
| [`dbis-DSTNetPlus-L`](https://huggingface.co/Illaitar/dbis-DSTNetPlus-L) / [`dbisf-DSTNetPlus-L`](https://huggingface.co/Illaitar/dbisf-DSTNetPlus-L) | DSTNet+ L (TPAMI 2025), video | 56 / 16 MB | 35.7078 / 35.6999 | 3.45 / 0.58 s | Apache 2.0 |
| [`dbis-Turtle`](https://huggingface.co/Illaitar/dbis-Turtle) / [`dbisf-Turtle`](https://huggingface.co/Illaitar/dbisf-Turtle) | Turtle (NeurIPS 2024), video | 236 / 68 MB | 34.5598 / 34.5316 | 724 / 199 ms | MIT |
| [`dbis-Concertormer`](https://huggingface.co/Illaitar/dbis-Concertormer), [`dbis-Concertormer-lite`](https://huggingface.co/Illaitar/dbis-Concertormer-lite) | Concertormer (ICCV 2025) | 202, 116 MB | the authors' outputs | | MIT |

Every model has its own Hugging Face repository, named as the model; all of them are in the
[dbis collection](https://huggingface.co/collections/Illaitar/dbis-6b09930cbf629d8d89c027ba). Each weight file keeps the terms of the checkpoint it is made from (the card of its repository); cite the
method's paper when you use it. `python -m dbis.convert` makes the files from the authors' checkpoints instead.

## Training

One script for every model: the authors' recipe, a plan fitted to the machine once (the largest batch that fits,
precision, activation checkpointing, `torch.compile`, the frame reader and data workers), then the training:

```bash
python -m dbis.train dbisf-NAFNet-64 --plan --data DIR --out runs/naf      # once per machine
python -m dbis.train dbisf-NAFNet-64 --data DIR --out runs/naf             # or: torchrun --nproc-per-node 8 -m dbis.train ...
```

The layout of DIR is found by itself (GoPro, REDS, BSD, DVD, RealBlur, GS-Blur, MIORe as downloaded). Options:
the training losses of the methods (`--loss`), weight averages (`--ema`), the gradient noise scale and a batch that
follows it (`--noise-every`, `--adapt-batch`), data echoing (`--echo`), frames held in memory or on the GPU
(`--reader ram|gpu`). What each does and what it measured: [DBIS.md](https://github.com/illaitar/deblurring-is-solved/blob/main/DBIS.md).

Requires Python 3.9+ and PyTorch; numpy, safetensors, zstandard and pillow come with the package.

## License

Our code is under the MIT License. The networks re-implement their authors' code and keep its terms, listed per
folder in NOTICE: MIT (NAFNet, Concertormer, Turtle, BSSTNet), Apache 2.0 (DSTNet+), BSD 3-Clause (RAFT in
BSSTNet), and the S-Lab License 1.0 for the three ProPainter blocks that BSSTNet reuses, which allows only
non-commercial use.
