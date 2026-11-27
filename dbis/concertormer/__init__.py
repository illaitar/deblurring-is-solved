"""Concertormer (Kuo, Pan, Chien, Yang, "Efficient Concertormer for Image Deblurring and Beyond", ICCV 2025,
arXiv 2404.06135) for inference: dbis-Concertormer and dbis-Concertormer-lite, the GoPro models.

Credits and licenses
  Concertormer  https://github.com/setsunil/Concertormer (commit 94269c5), MIT License (the repository's LICENSE,
                Copyright (c) 2022 megvii-model, carried over from NAFNet on which the code builds). Architecture
                (basicsr/models/archs/Concertormer_arch.py, arch_util.py LayerNorm2d), test configurations
                (options/test/GoPro/Concertormer{,_lite}.yml) and the weights are theirs.
This package is a re-implementation in plain PyTorch (no einops, no basicsr).

Concertormer.forward(x) is the full-resolution output of the authors' ConcertormerLocal (tlc="exact", the type their
test configs use: train_size 256, grid_factor 1.75 for GoPro) or of their Concertormer (tlc=None, the attention over the
whole map). What differs, none of it changing the computation:
  * the windows of the TLC attention follow from the training crop and grid_factor (no dry run at construction);
  * the shifted-window mask is built once per size and kept on the device (the authors build it on the CPU in every
    call of every odd block), and torch.cuda.empty_cache() is not called after every attention;
  * the three coarser outputs, which only the training loss uses, are not made by forward (forward_all makes them);
  * the einops rearranges are written as the reshapes and permutes they stand for.
The benchmark ran both GoPro models whole-frame (the network reflect-pads to a multiple of 64 itself) with float16
autocast, which is restore()'s default here (Concertormer-lite on BSD, GS-Blur and MIORe in float32, where float16
overflowed).

Layout (the same for every method of dbis):
  net.py        the full network: Concertormer (restore()), fold() (the authors' checkpoint -> the state dict)
  encoder.py    ConcertormerEncoder: intro conv, the three encoder stages with the re-injected downscaled input and
                the middle blocks (features at 1, 1/2, 1/4 and 1/8), loaded alone by dbis.load_encoder()
  blocks/       csa.py (LayerNorm2d, concerto self-attention whole-map and windowed, the CSA block)
"""
from .encoder import ConcertormerEncoder
from .net import Concertormer, fold

ENCODER = "ConcertormerEncoder"   # dbis.load_encoder: the encoder class and the prefix of its weights in the state dict
ENCODER_PREFIX = ""
__all__ = ["Concertormer", "ConcertormerEncoder", "fold", "ENCODER", "ENCODER_PREFIX"]
