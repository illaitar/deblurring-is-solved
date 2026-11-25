"""BSSTNet (Zhang, Xie and Yao, "Blur-aware Spatio-temporal Sparse Transformer for Video Deblurring", CVPR 2024) for
inference: dbis-BSSTNet, with its RAFT optical flow and the authors' patch-wise window inference.

Credits and licenses
  BSSTNet     https://github.com/huicongzhang/BSSTNet (commit c35a43c), by Huicong Zhang, Haozhe Xie, Hongxun Yao, MIT
              License (stated in the README of the repository). Architecture, inference procedure and the GoPro weights
              (BSST_gopro.pth) are theirs.
  RAFT        https://github.com/princeton-vl/RAFT (Teed and Deng, ECCV 2020), BSD 3-Clause License, Copyright (c) 2020,
              princeton-vl. Flow network and weights (raft-things.pth), as vendored by BSSTNet (basicsr/archs/RAFT).
  ProPainter  https://github.com/sczhou/ProPainter (Zhou et al., ICCV 2023): the temporal sparse transformer (SoftSplit,
              SoftComp, SparseWindowAttention, FusionFeedForward) and the recurrent flow completion network, reused by
              BSSTNet. ProPainter's own code is under the S-Lab License 1.0 (non-commercial use only).
  Shift-Net   https://github.com/dasongli1/Shift-Net (Li et al., CVPR 2023): the grouped spatial-temporal shift blocks
              and the CAB blocks (basicsr/archs/gshift_arch.py of BSSTNet; no license file in that repository).
  The Swin layers of the reconstruction module (VRT/RVRT lineage, "Facebook, BSD license" header in BSST_arch.py) and
  the second-order deformable alignment (BasicVSR++) come with BSSTNet.
This package is a re-implementation in plain PyTorch (torch, torchvision.ops.deform_conv2d). The mmcv DCN of the flow
completion net is replaced by torchvision.ops.deform_conv2d as in our patch of the authors' code (ext/BSSTNet.patch),
which is the code the benchmark runs.

BSSTNet.forward(lqs) is the authors' ModelBSST.get_bi_flows + test_by_patch as runs/baselines/bsstnet/run_bsst.py runs
them: RAFT (fp32, 20 iterations) on the frames downsampled x0.5, both directions of the T-1 pairs, flows downsampled x0.5
again and divided by 2; BSST on 256 x 256 patches with 64 px overlap (stride 192, the last patch moved back to the
border) under fp16 autocast; the outer 32 px bands of every patch that has a neighbour zeroed; output = sum / count.
Input (B, T, 3, H, W) in [0, 1] on the GPU, H and W multiples of 16 and at least 256 (the authors pad nothing; RAFT
needs the half-size frames to be multiples of 8); output (B, T, 3, H, W) float32 on the GPU.

What differs from the authors' code, none of it changing the computation of a patch:
  * patch_batch patches go through BSST in one call (batch dimension). The authors' forward assumes batch 1 in places
    (lqs[0], .unsqueeze(0), trans_feats[0], and the channel roll of the shift blocks over the flattened batch*channel
    dimension, which moves channels across frames); here each of these acts per patch. Kernels for a larger batch may
    round differently (cuDNN/cuBLAS algorithms; also in "fp32", whose convolutions run in TF32 under torch's default
    flags), which is the only source of differences. Inside a call the
    launch-bound recurrent parts (flow completion, the four propagation branches over the frames) run on all patches
    at once and the feed-forward stages (encoder, Swin reconstruction, sparse transformer, decoder) in sub-batches of
    stage_batch patches, which bounds the memory. The propagation writes its features straight into the float32
    buffer that the authors build afterwards with torch.stack + torch.cat (the float16 spatial features are promoted
    by every torch.cat they enter, so their float32 copies give the same values).
  * blending on the GPU, in the authors' patch order (the same float32 sums), with the bands zeroed by multiplying with
    0 as the authors' `*= 0` does (a NaN in a band still reaches the output).
  * dropped as pure overhead: the mirror-extension check (a CPU sync, its flag is never read), cpu_cache (off for
    T <= 150, it only moves tensors), the unused modules (their weights are dropped by fold()), RAFT's flow upsampling
    and mask head at the 19 intermediate iterations (only the last flow is returned), an unused reshape and a
    cat(chunk(offset)) copy in the flow completion net, coordinate grids and correlation offsets rebuilt (or copied from
    the CPU) at every call (cached on the device), and the 2 x 4 x patches nonzero() syncs of the sparse attention
    (indices taken for all depths and patches at once).
  * kept on purpose: `blur_map_st_win_q_soft = blur_map_st_win` is an alias, so the frames zeroed for one depth of the
    sparse attention stay zeroed for the next ones (from the second depth on the query scores are all zero and the
    sort order decides); the shift blocks roll channels from the last frame of a window into the first; LayerNorm2d,
    softmax, deform_conv2d (fp32 inside torchvision's autocast rule) and the dtype promotions of the authors' code run
    with the same ops in the same order.
precision "fp32" runs BSST without autocast, as run_bsst.py does for BL_PRECISION=fp32; cuDNN TF32 follows torch's
global flags in both cases (default: on for convolutions).

Layout (the same for every method of dbis):
  net.py        the full network: BSST (the restoration network on a batch of patches), BSSTNet (RAFT + BSST on the
                patches of a window; restore()), fold() (the authors' checkpoints -> the state dict of BSSTNet)
  encoder.py    BSSTEncoder: the per-frame encoder of BSST (features at 1, 1/2 and 1/4 of the frame size), which
                dbis.load_encoder() loads alone
  blocks/       common.py (constants, LeakyReLU), flow.py (warping, consistency, blur map), cnn.py (CAB blocks, shift
                blocks), align.py (deformable alignment), flow_completion.py, swin.py (reconstruction),
                sparse_transformer.py, raft.py
"""
from .encoder import BSSTEncoder
from .net import BSST, BSSTNet, fold

ENCODER = "BSSTEncoder"          # dbis.load_encoder: the encoder class and the prefix of its weights in the state dict
ENCODER_PREFIX = "bsst."
__all__ = ["BSST", "BSSTNet", "BSSTEncoder", "fold", "ENCODER", "ENCODER_PREFIX"]
