"""Turtle (Ghasemabadi, Janjua, Salameh, Niu, "Learning Truncated Causal History Model for Video Restoration", NeurIPS
2024) for inference: dbis-Turtle, the GoPro model.

Credits and licenses
  Turtle      https://github.com/Ascend-Research/Turtle (commit 7094f42), MIT License, Copyright (c) 2024 Huawei
              Technologies Co., Ltd. Architecture (basicsr/models/archs/turtle_t1_arch.py), inference procedure
              (basicsr/inference.py) and the GoPro weights (GoPro_Deblur.pth) are theirs.
This package is a re-implementation in plain PyTorch (no einops, no basicsr).

Turtle.forward(x, k_cached, v_cached) is the authors' Turtle_t1.forward for the GoPro configuration
(Turtle_Deblur_Gopro.yml: use_both_input False, so only the current frame enters the network). What differs, none of it
changing the computation:
  * the local attention mask of SAB is built once per token grid and kept on the device (the authors build it on the
    CPU with torch.cdist and copy it in every call, three times per tile);
  * the caches stay on the GPU (the authors move every tile's caches to the CPU and back after each frame);
  * the einops rearranges are written as the reshapes and permutes they stand for.
The benchmark runs the authors' tiled inference (tiles of 320 px with 192 px overlap, each tile with its own history,
one pass per sequence from its first frame); restore() does the same by default, and step() is one frame of it.
The authors' BSD 3ms-24ms model is another architecture (turtle_arch.py, SAB with several heads and a positional
encoding) and is not here.

Layout (the same for every method of dbis):
  net.py        the full network: Turtle (restore(), step()), fold() (the authors' checkpoint -> the state dict)
  encoder.py    TurtleEncoder: the input conv and the three encoder levels, which see only the current frame (features
                at 1, 1/2, 1/4 and the bottleneck input at 1/8), loaded alone by dbis.load_encoder()
  blocks/       common.py (layer norm, resizing, feed-forward), attention.py (channel attention, ReducedAttn),
                history.py (FHR, SAB, CHM: the blocks with history), level.py (Turtle block, level and bottleneck stacks)
"""
from .encoder import TurtleEncoder
from .net import Turtle, fold

ENCODER = "TurtleEncoder"        # dbis.load_encoder: the encoder class and the prefix of its weights in the state dict
ENCODER_PREFIX = ""
__all__ = ["Turtle", "TurtleEncoder", "fold", "ENCODER", "ENCODER_PREFIX"]
