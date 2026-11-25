"""DSTNet+ (Pan, Sun, Xu, Dong and Tang, "Learning Efficient Deep Discriminative Spatial and Temporal Networks for Video
Deblurring", TPAMI 2025; extends DSTNet, CVPR 2023) for inference: dbis-DSTNetPlus-L.

Credits and licenses
  DSTNet+     https://github.com/sunny2109/DSTNet-plus (commit 54363c1), by Jinshan Pan, Long Sun, Boming Xu, Jiangxin
              Dong, Jinhui Tang; Apache License 2.0 (LICENSE.txt of the repository, a BasicSR fork). Architecture,
              inference procedure and the GoPro weights (DSTNetPlus_L_gopro.pth) are theirs.
  The dynamic depthwise conv is applied in the authors' code by a CuPy kernel (basicsr/archs/prog_dynconv.py); here it
  is plain PyTorch (blocks/dynconv.py).
This package is a re-implementation in plain PyTorch.

DSTNetPlus.forward(lrs) is the authors' DSTNetPlus_Final.forward with our patch (ext/DSTNet-plus.patch: frame_chunk, the
per-frame full-resolution layers a few frames at a time, the same maths). What differs from the authors' code, none of
it changing the computation:
  * the dynamic depthwise conv: nine torch.addcmul in the order of the CuPy kernel's loop instead of the kernel;
  * the fixed Haar weights are rebuilt in code (the authors store them as a parameter, wave.haar_weights);
  * BackwardProp and ForwardProp are one class with a direction flag (the same modules and weight names).
The benchmark runs the authors' code in float32 on clips of 50 frames that do not overlap (the authors' GoPro test),
frame_chunk 10; restore() does the same by default.

Layout (the same for every method of dbis):
  net.py        the full network: DSTNetPlus (restore()), fold() (the authors' checkpoint -> the state dict)
  encoder.py    DSTNetPlusEncoder: the first conv, the Haar split and the high-frequency branch (features at 1 and
                1/2 of the frame size), which dbis.load_encoder() loads alone
  blocks/       haar.py (Haar wavelet), dynconv.py (dynamic depthwise conv), propagation.py (PAGF fusion, residual
                blocks, the recurrent propagation)
"""
from .encoder import DSTNetPlusEncoder
from .net import DSTNetPlus, fold

ENCODER = "DSTNetPlusEncoder"    # dbis.load_encoder: the encoder class and the prefix of its weights in the state dict
ENCODER_PREFIX = ""
__all__ = ["DSTNetPlus", "DSTNetPlusEncoder", "fold", "ENCODER", "ENCODER_PREFIX"]
