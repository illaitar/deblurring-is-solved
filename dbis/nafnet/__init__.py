"""NAFNet (Chen et al., "Simple Baselines for Image Restoration", ECCV 2022) for inference: dbis-NAFNet and dbisf-NAFNet.

Architecture and weights by the authors: https://github.com/megvii-research/NAFNet, MIT License,
Copyright (c) 2022 megvii-model. This package is a re-implementation in plain PyTorch.

dbis-NAFNet (tlc="exact") gives the outputs of the authors' NAFNetLocal (basicsr/models/archs/NAFNet_arch.py and
local_arch.py, the configuration of their GoPro results) with the weights written by `python -m dbis.convert`, which
folds every affine step into the convolution after it:
  * the scale and shift of the two LayerNorms of a block go into conv1 and conv4,
  * the residual scales beta and gamma go into conv3 and conv5.
TLC (test-time local statistics, Chu et al., ECCV 2022): the channel attention averages over a fixed window, 1.5 times
the training crop at that scale, instead of over the whole image. The window sizes follow from the training crop, so
the dry run of the authors' code at construction is not needed.

dbisf-NAFNet (tlc="fast") averages that window on a map pooled into r x r cells, runs the attention's 1x1 conv there and
upsamples bilinearly. Both steps are linear and the upsampling weights sum to 1, so moving the conv before the
upsampling changes nothing; the cell pooling is the only approximation (see DBIS.md for its cost in dB).
tlc=None is the plain NAFNet (global channel attention).

Layout (the same for every method of dbis):
  net.py        the full network: NAFNet (restore()), fold() (the authors' checkpoint -> the state dict of NAFNet)
  encoder.py    NAFNetEncoder: intro, the four encoder stages and the middle blocks (features at 1, 1/2, 1/4, 1/8,
                1/16 of the frame size), which dbis.load_encoder() loads alone
  blocks/       nafblock.py: the NAFBlock, its normalisation and the TLC window mean
"""
from .encoder import NAFNetEncoder
from .net import NAFNet, fold

ENCODER = "NAFNetEncoder"        # dbis.load_encoder: the encoder class and the prefix of its weights in the state dict
ENCODER_PREFIX = ""
__all__ = ["NAFNet", "NAFNetEncoder", "fold", "ENCODER", "ENCODER_PREFIX"]
