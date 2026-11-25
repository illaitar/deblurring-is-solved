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
This file is a re-implementation in plain PyTorch (torch, torchvision.ops.deform_conv2d). The mmcv DCN of the flow
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
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

_CONST = {}


def _const(key, make):
    """Constants (grids, masks, index tables) made once per shape and device."""
    v = _CONST.get(key)
    if v is None:
        v = _CONST[key] = make()
    return v


def lrelu(slope=0.1):
    return nn.LeakyReLU(negative_slope=slope, inplace=True)


# ----------------------------------------------------------------------------------------------- flow helpers (BSST)
def _grid(h, w, dtype, device):
    def make():
        gy, gx = torch.meshgrid(torch.arange(0, h, dtype=dtype, device=device),
                                torch.arange(0, w, dtype=dtype, device=device), indexing="ij")
        return torch.stack((gx, gy), 2).float()
    return _const(("grid", h, w, dtype, device), make)


def flow_warp(x, flow):
    """Warp x (n, c, h, w) by flow (n, h, w, 2): bilinear, zero padding, align_corners=True."""
    _, _, h, w = x.size()
    vgrid = _grid(h, w, x.dtype, x.device) + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    return F.grid_sample(x, torch.stack((vgrid_x, vgrid_y), dim=3), mode="bilinear", padding_mode="zeros",
                         align_corners=True)


def length_sq(x):
    return torch.sum(torch.square(x), dim=1, keepdim=True)


def fb_valid(flow_fw, flow_bw, alpha1=0.01, alpha2=0.5):
    """Forward-backward consistency mask (fbConsistencyCheck of the authors, its first output)."""
    flow_bw_warped = flow_warp(flow_bw, flow_fw.permute(0, 2, 3, 1))
    flow_diff_fw = flow_fw + flow_bw_warped
    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)
    occ_thresh_fw = alpha1 * mag_sq_fw + alpha2
    return (length_sq(flow_diff_fw) < occ_thresh_fw).type_as(flow_fw)


def get_blur_map(flow_forwards, flow_backwards):
    """Squared flow magnitude per frame (mean of the two neighbouring flows), min-max normalised per sequence."""
    b, t, c, h, w = flow_forwards.shape
    first = length_sq(flow_backwards[:, 0, ...]).reshape(b, 1, 1, h, w)
    last = length_sq(flow_forwards[:, -1, ...]).reshape(b, 1, 1, h, w)
    mid = (length_sq(flow_backwards[:, 1:, ...].reshape(b * (t - 1), 2, h, w))
           + length_sq(flow_forwards[:, :-1, ...].reshape(b * (t - 1), 2, h, w))) / 2
    blur_map = torch.cat([first, mid.reshape(b, t - 1, 1, h, w), last], dim=1)
    blur_max, _ = torch.max(blur_map.reshape(b, -1), dim=-1)
    blur_min, _ = torch.min(blur_map.reshape(b, -1), dim=-1)
    return (blur_map - blur_min.reshape(b, 1, 1, 1, 1)) / (blur_max - blur_min).reshape(b, 1, 1, 1, 1)


# --------------------------------------------------------------------------------------- CNN blocks (gshift_arch.py)
class LayerNorm2d(nn.Module):
    """LayerNorm over the channels of NCHW, the authors' LayerNormFunction forward (same ops, so the same autocast
    dtypes: pow and mean run in float32)."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        c = x.size(1)
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, c, 1, 1) * y + self.bias.view(1, c, 1, 1)


def conv(cin, cout, k, bias=False, stride=1):
    return nn.Conv2d(cin, cout, k, padding=k // 2, bias=bias, stride=stride)


class CALayer(nn.Module):
    """Channel attention (CALayer and CALayer2 of the authors, reduction 1)."""

    def __init__(self, c):
        super().__init__()
        self.conv_du = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.ReLU(inplace=True),
                                     nn.Conv2d(c, c, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.conv_du(F.adaptive_avg_pool2d(x, 1))


class CAB(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.body = nn.Sequential(conv(c, c, k), lrelu(), conv(c, c, k))
        self.CA = CALayer(c)

    def forward(self, x):
        res = self.CA(self.body(x))
        res += x
        return res


class Gate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SigmoidGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * torch.sigmoid(x2)


class RepConv(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        self.conv_1 = nn.Conv2d(c, c, k, bias=False, padding=k // 2, groups=c)
        self.conv_2 = nn.Conv2d(c, c, 3, bias=False, padding=1, groups=c)

    def forward(self, x):
        return self.conv_1(x) + self.conv_2(x) + x


class RepConv2(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv_2 = nn.Conv2d(c, c, 3, bias=False, padding=1, groups=c)

    def forward(self, x):
        return self.conv_2(x) + x


def cab_body(cin, c, k):
    return nn.Sequential(conv(cin, 2 * c, 1), RepConv2(2 * c), Gate(), RepConv(c, k), conv(c, 2 * c, 1), SigmoidGate(),
                         CALayer(c), conv(c, c, 1))


class CAB1(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        self.norm = LayerNorm2d(c)
        self.body = cab_body(c, c, k)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        return x + self.body(self.norm(x)) * self.beta


class CAB2(nn.Module):
    """CAB1 on [x[:, :c], depthwise 3x3 of x[:, c:]], residual on the first c channels."""

    def __init__(self, c, k, add):
        super().__init__()
        self.n_feat = c
        self.conv1 = nn.Conv2d(add, add, 3, bias=False, padding=1, groups=add)
        self.norm = LayerNorm2d(add + c)
        self.body = cab_body(c + add, c, k)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        shortcut, hw = x[:, 0:self.n_feat], x[:, self.n_feat:]
        res = self.body(self.norm(torch.cat((shortcut, self.conv1(hw)), dim=1)))
        return shortcut + res * self.beta


# spatial shifts (dy, dx) of spatial_shift2: out[y, x] = in[y - dy, x - dx], zero outside; 16 groups of n2 channels,
# then 8 groups of n1 channels
SHIFTS_A = ((8, 8), (8, 4), (8, 0), (8, -4), (8, -8), (-8, 8), (-8, 4), (-8, 0), (-8, -4), (-8, -8),
            (4, 8), (4, -8), (0, 8), (0, -8), (-4, 8), (-4, -8))
SHIFTS_B = ((4, 4), (4, 0), (4, -4), (0, 4), (0, -4), (-4, 4), (-4, 0), (-4, -4))


def spatial_shift(hw, number):
    """spatial_shift2 of the authors (24 slice copies into zeros) as one pad and one cat: the same values."""
    n2 = (number - 1) // 2
    n1 = number - 2 * n2
    h, w = hw.shape[-2:]
    p = F.pad(hw, (8, 8, 8, 8))
    parts, c = [], 0
    for (dy, dx), k in [(s, n2) for s in SHIFTS_A] + [(s, n1) for s in SHIFTS_B]:
        parts.append(p[:, c:c + k, 8 - dy:8 - dy + h, 8 - dx:8 - dx + w])
        c += k
    return torch.cat(parts, dim=1)


class ShiftBlock(nn.Module):
    """Encoder_shift_block: four (CAB2, CAB1) pairs, each after a temporal channel roll and a grouped spatial shift."""

    def __init__(self, c):
        super().__init__()
        self.number = c // 2 // 8
        for name in ("encoder_level1", "encoder_level1_1", "encoder_level1_2", "encoder_level1_3"):
            setattr(self, name, nn.Sequential(CAB2(c, 5, 8 * self.number), CAB1(c, 5)))

    def channel_shift(self, x, n, reverse=False):
        """The authors roll x.view(1, B*C, H, W) by C/2 along dim 1, B = frames of one patch: half of the channels of
        every frame come from the previous (next) frame, with wrap-around. Here per patch: x.view(n, T*C, H, W)."""
        bt, c, h, w = x.shape
        s = -(c // 2) if reverse else c // 2
        y = torch.roll(x.view(n, bt // n * c, h, w), s, 1).view(bt, c, h, w)
        hw = y[:, -8 * self.number:, ...] if reverse else y[:, 0:8 * self.number, ...]
        return torch.cat((y, spatial_shift(hw, self.number)), dim=1)

    def forward(self, x, n):
        x = self.encoder_level1(self.channel_shift(x, n))
        x = self.encoder_level1_1(self.channel_shift(x, n, reverse=True))
        x = self.encoder_level1_2(self.channel_shift(x, n))
        return self.encoder_level1_3(self.channel_shift(x, n, reverse=True))


class PixelShufflePack(nn.Module):
    def __init__(self, cin, cout, scale, k):
        super().__init__()
        self.scale = scale
        self.upsample_conv = nn.Conv2d(cin, cout * scale * scale, k, padding=(k - 1) // 2)

    def forward(self, x):
        return F.pixel_shuffle(self.upsample_conv(x), self.scale)


class SkipUpSample(nn.Module):
    def __init__(self, c, s):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                nn.Conv2d(c + s, c, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        return self.up(x) + y


# ------------------------------------------------------------------------------------- deformable alignment modules
class BlurGuidedAlignment(nn.Module):
    """SecondOrderDeformableAlignmentWithBlurmapguide: offsets = flow + 5 tanh(.), masks shifted by the blur maps."""

    def __init__(self, cin, cout, groups=16, max_residue_magnitude=5):
        super().__init__()
        self.max_residue_magnitude = max_residue_magnitude
        self.weight = nn.Parameter(torch.empty(cout, cin, 3, 3))
        self.bias = nn.Parameter(torch.empty(cout))
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * cout + 4 + 5, cout, 3, 1, 1), lrelu(), nn.Conv2d(cout, cout, 3, 1, 1), lrelu(),
            nn.Conv2d(cout, cout, 3, 1, 1), lrelu(), nn.Conv2d(cout, 27 * groups, 3, 1, 1))

    def forward(self, x, extra_feat, flow_1, flow_2, blur_map_1, blur_map_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        o1, o2, mask = torch.chunk(self.conv_offset(extra_feat), 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)
        mask_1, mask_2 = torch.chunk(mask, 2, dim=1)
        blur_map_1 = (2 * blur_map_1 - 1) * 0.1
        blur_map_2 = (2 * blur_map_2 - 1) * 0.1
        mask_1 = mask_1 + blur_map_1.repeat(1, mask_1.size(1), 1, 1)
        mask_2 = mask_2 + blur_map_2.repeat(1, mask_1.size(1), 1, 1)
        mask = torch.sigmoid(torch.cat([mask_1, mask_2], dim=1))
        return deform_conv2d(x, offset, self.weight, self.bias, (1, 1), (1, 1), (1, 1), mask)


class FlowAlignment(nn.Module):
    """SecondOrderDeformableAlignment of ProPainter's flow completion net (mmcv DCN -> torchvision, as patched)."""

    def __init__(self, cin, cout, groups=16, max_residue_magnitude=5):
        super().__init__()
        self.max_residue_magnitude = max_residue_magnitude
        self.weight = nn.Parameter(torch.empty(cout, cin, 3, 3))
        self.bias = nn.Parameter(torch.empty(cout))
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * cout, cout, 3, 1, 1), lrelu(), nn.Conv2d(cout, cout, 3, 1, 1), lrelu(),
            nn.Conv2d(cout, cout, 3, 1, 1), lrelu(), nn.Conv2d(cout, 27 * groups, 3, 1, 1))

    def forward(self, x, extra_feat):
        o1, o2, mask = torch.chunk(self.conv_offset(extra_feat), 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        # (the authors split offset in two and concatenate it again: the same tensor)
        return deform_conv2d(x, offset, self.weight, self.bias, (1, 1), (1, 1), (1, 1), torch.sigmoid(mask))


# ------------------------------------------------------------------------- flow completion (ProPainter, BSSTNet copy)
class BidirectionalPropagation(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.channel = c
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        for i, m in enumerate(("backward_", "forward_")):
            self.deform_align[m] = FlowAlignment(2 * c, c)
            self.backbone[m] = nn.Sequential(nn.Conv2d((2 + i) * c, c, 3, 1, 1), lrelu(), nn.Conv2d(c, c, 3, 1, 1))
        self.fusion = nn.Conv2d(2 * c, c, 1, 1, 0)

    def forward(self, x):
        b, t, c, h, w = x.shape
        feats = {"spatial": [x[:, i, :, :, :] for i in range(t)]}
        for name in ("backward_", "forward_"):
            feats[name] = []
            frame_idx = range(t - 1, -1, -1) if "backward" in name else range(t)
            feat_prop = x.new_zeros(b, self.channel, h, w)
            for i, idx in enumerate(frame_idx):
                feat_current = feats["spatial"][idx]
                if i > 0:
                    cond_n1 = feat_prop
                    feat_n2 = torch.zeros_like(feat_prop)
                    cond_n2 = torch.zeros_like(cond_n1)
                    if i > 1:
                        feat_n2 = feats[name][-2]
                        cond_n2 = feat_n2
                    cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                    feat_prop = self.deform_align[name](torch.cat([feat_prop, feat_n2], dim=1), cond)
                feat = [feat_current] + [feats[k][idx] for k in feats if k not in ("spatial", name)] + [feat_prop]
                feat_prop = feat_prop + self.backbone[name](torch.cat(feat, dim=1))
                feats[name].append(feat_prop)
            if "backward" in name:
                feats[name] = feats[name][::-1]
        outputs = [self.fusion(torch.cat([feats[k][i] for k in feats if k != "spatial"], dim=1)) for i in range(t)]
        return torch.stack(outputs, dim=1) + x


class Deconv(nn.Module):
    def __init__(self, cin, cout, k=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=k, stride=1, padding=padding)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))


class P3DBlock(nn.Module):
    def __init__(self, cin, cout, k, stride, padding):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(cin, cout, (1, k, k), (1, stride, stride), (0, padding, padding)),
                                   lrelu(0.2))
        self.conv2 = nn.Sequential(nn.Conv3d(cout, cout, (3, 1, 1), (1, 1, 1), (2, 0, 0), dilation=(2, 1, 1)))

    def forward(self, x):
        return self.conv2(self.conv1(x))


class FlowCompletion(nn.Module):
    """RecurrentFlowCompleteNet: refines the RAFT flows of a patch from (blur map, flow)."""

    def __init__(self):
        super().__init__()
        self.downsample = nn.Sequential(nn.Conv3d(3, 32, (1, 5, 5), (1, 2, 2), (0, 2, 2), padding_mode="replicate"),
                                        lrelu(0.2))
        self.encoder1 = nn.Sequential(P3DBlock(32, 32, 3, 1, 1), lrelu(0.2), P3DBlock(32, 64, 3, 2, 1), lrelu(0.2))
        self.mid_dilation = nn.Sequential(
            nn.Conv3d(64, 64, (1, 3, 3), (1, 1, 1), padding=(0, 3, 3), dilation=(1, 3, 3)), lrelu(0.2),
            nn.Conv3d(64, 64, (1, 3, 3), (1, 1, 1), padding=(0, 2, 2), dilation=(1, 2, 2)), lrelu(0.2),
            nn.Conv3d(64, 64, (1, 3, 3), (1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1)), lrelu(0.2))
        self.feat_prop_module = BidirectionalPropagation(64)
        self.decoder1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), lrelu(0.2), Deconv(64, 32, 3, 1), lrelu(0.2))
        self.upsample = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), lrelu(0.2), Deconv(32, 2, 3, 1))

    def forward(self, blur_motion_map, flows):
        b, t, _, h, w = blur_motion_map.size()
        inputs = torch.cat((blur_motion_map.permute(0, 2, 1, 3, 4), flows.permute(0, 2, 1, 3, 4)), dim=1)
        x = self.downsample(inputs)
        feat_e1 = self.encoder1(x)
        feat_prop = self.feat_prop_module(self.mid_dilation(feat_e1).permute(0, 2, 1, 3, 4))
        feat_prop = feat_prop.view(-1, 64, h // 4, w // 4)
        _, c, _, h_f, w_f = feat_e1.shape
        feat_d2 = feat_prop + feat_e1.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h_f, w_f)
        flow = self.upsample(self.decoder1(feat_d2))
        return flow.view(b, t, 2, h, w) + flows

    def forward_bidirect_flow(self, blur_motion_map, flows_forward, flows_backward):
        bm_forward = blur_motion_map[:, :-1, ...].contiguous()
        bm_backward = blur_motion_map[:, 1:, ...].contiguous()
        pred_backward = self.forward(bm_backward, flows_backward)
        pred_forward = self.forward(torch.flip(bm_forward, dims=[1]), torch.flip(flows_forward, dims=[1]))
        return torch.flip(pred_forward, dims=[1]), pred_backward


# ------------------------------------------------------------------------------- Swin reconstruction (BSST_arch.py)
def swin_partition(x, ws):
    b, d, h, w, c = x.shape
    x = x.view(b, d // ws[0], ws[0], h // ws[1], ws[1], w // ws[2], ws[2], c)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, ws[0] * ws[1] * ws[2], c)


def swin_reverse(windows, ws, b, d, h, w):
    x = windows.view(b, d // ws[0], h // ws[1], w // ws[2], ws[0], ws[1], ws[2], -1)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)


def get_window_size(x_size, window_size, shift_size):
    ws, ss = list(window_size), list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            ws[i] = x_size[i]
            ss[i] = 0
    return tuple(ws), tuple(ss)


def swin_mask(d, h, w, ws, ss, device):
    """compute_mask of the authors (lru_cache there): -100 between tokens of different regions of a shifted window."""
    def make():
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        cnt = 0
        for a in slice(-ws[0]), slice(-ws[0], -ss[0]), slice(-ss[0], None):
            for b in slice(-ws[1]), slice(-ws[1], -ss[1]), slice(-ss[1], None):
                for c in slice(-ws[2]), slice(-ws[2], -ss[2]), slice(-ss[2], None):
                    img_mask[:, a, b, c, :] = cnt
                    cnt += 1
        mw = swin_partition(img_mask, ws).squeeze(-1)
        m = mw.unsqueeze(1) - mw.unsqueeze(2)
        return m.masked_fill(m != 0, float(-100.0)).masked_fill(m == 0, float(0.0))
    return _const(("swin_mask", d, h, w, ws, ss, device), make)


def relative_position_index(ws, device="cpu"):
    """The buffer of the authors' WindowAttention (dropped from the weights, rebuilt here)."""
    def make():
        coords = torch.stack(torch.meshgrid(torch.arange(ws[0]), torch.arange(ws[1]), torch.arange(ws[2]),
                                            indexing="ij"))
        cf = torch.flatten(coords, 1)
        rel = (cf[:, :, None] - cf[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws[0] - 1
        rel[:, :, 1] += ws[1] - 1
        rel[:, :, 2] += ws[2] - 1
        rel[:, :, 0] *= (2 * ws[1] - 1) * (2 * ws[2] - 1)
        rel[:, :, 1] *= (2 * ws[2] - 1)
        return rel.sum(-1).to(device)
    return _const(("relpos", tuple(ws), torch.device(device)), make)


class Mlp(nn.Module):
    def __init__(self, c, hidden):
        super().__init__()
        self.fc1 = nn.Linear(c, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, c)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.window_size = tuple(window_size)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        ws = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws[0] - 1) * (2 * ws[1] - 1) * (2 * ws[2] - 1), num_heads))
        self.qkv_self = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv_self(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        index = relative_position_index(self.window_size, x.device)
        bias = self.relative_position_bias_table[index[:n, :n].reshape(-1)].reshape(n, n, -1)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask[:, :n, :n].unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = F.softmax(attn, -1, dtype=q.dtype)
        return self.proj((attn @ v).transpose(1, 2).reshape(b_, n, c))


class STL(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=2.0):
        super().__init__()
        self.window_size, self.shift_size = window_size, shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward_part1(self, x, mask_matrix):
        b, d, h, w, c = x.shape
        ws, ss = get_window_size((d, h, w), self.window_size, self.shift_size)
        x = self.norm1(x)
        pad_d1 = (ws[0] - d % ws[0]) % ws[0]
        pad_b = (ws[1] - h % ws[1]) % ws[1]
        pad_r = (ws[2] - w % ws[2]) % ws[2]
        if pad_d1 or pad_b or pad_r:   # (the authors pad by zero too, a copy of a contiguous tensor)
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_d1), mode="constant")
        _, dp, hp, wp, _ = x.shape
        shifted = any(i > 0 for i in ss)
        if shifted:
            x = torch.roll(x, shifts=(-ss[0], -ss[1], -ss[2]), dims=(1, 2, 3))
        attn = self.attn(swin_partition(x, ws), mask=mask_matrix if shifted else None)
        x = swin_reverse(attn.view(-1, *(ws + (c,))), ws, b, dp, hp, wp)
        if shifted:
            x = torch.roll(x, shifts=(ss[0], ss[1], ss[2]), dims=(1, 2, 3))
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :d, :h, :w, :]
        return x

    def forward(self, x, mask_matrix):
        x = x + self.forward_part1(x, mask_matrix)
        return x + self.mlp(self.norm2(x))


class STG(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=2.0):
        super().__init__()
        self.window_size = list(window_size)
        self.shift_size = [i // 2 for i in window_size]
        self.blocks = nn.ModuleList([
            STL(dim, num_heads, self.window_size, [0, 0, 0] if i % 2 == 0 else self.shift_size, mlp_ratio)
            for i in range(depth)])

    def forward(self, x):
        b, c, d, h, w = x.shape
        ws, ss = get_window_size((d, h, w), self.window_size, self.shift_size)
        x = x.permute(0, 2, 3, 4, 1)
        dp = int(math.ceil(d / ws[0])) * ws[0]
        hp = int(math.ceil(h / ws[1])) * ws[1]
        wp = int(math.ceil(w / ws[2])) * ws[2]
        attn_mask = swin_mask(dp, hp, wp, ws, ss, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return x.view(b, d, h, w, -1).permute(0, 4, 1, 2, 3)


class RSTB(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=2.0):
        super().__init__()
        self.residual_group = STG(dim, depth, num_heads, window_size, mlp_ratio)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.linear(self.residual_group(x).transpose(1, 4)).transpose(1, 4)


class Permute(nn.Module):
    """einops.Rearrange for a pure permutation (a view, as einops returns)."""

    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class Reconstruction(nn.Module):
    """RSTBWithInputConv: Conv3d (1, 3, 3) with groups, LayerNorm, 2 RSTB, LayerNorm; (n, t, c, h, w) in and out."""

    def __init__(self, cin=960, dim=192, groups=3, num_blocks=2, depth=2, num_heads=6, window_size=(2, 8, 8)):
        super().__init__()
        self.main = nn.Sequential(
            Permute(0, 2, 1, 3, 4), nn.Conv3d(cin, dim, (1, 3, 3), stride=1, padding=(0, 1, 1), groups=groups),
            Permute(0, 2, 3, 4, 1), nn.LayerNorm(dim), Permute(0, 4, 1, 2, 3),
            nn.Sequential(*[RSTB(dim, depth, num_heads, window_size) for _ in range(num_blocks)]),
            Permute(0, 2, 3, 4, 1), nn.LayerNorm(dim), Permute(0, 1, 4, 2, 3))

    def forward(self, x):
        return self.main(x)


# ---------------------------------------------------------------------- temporal sparse transformer (ProPainter copy)
class SoftSplit(nn.Module):
    def __init__(self, c, hidden, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.embedding = nn.Linear(kernel_size[0] * kernel_size[1] * c, hidden)

    def forward(self, x, b, output_size):
        f_h = int((output_size[0] + 2 * self.padding[0] - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        f_w = int((output_size[1] + 2 * self.padding[1] - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        feat = F.unfold(x, self.kernel_size, 1, self.padding, self.stride).permute(0, 2, 1)
        feat = self.embedding(feat)
        return feat.view(b, -1, f_h, f_w, feat.size(2))


class SoftComp(nn.Module):
    def __init__(self, c, hidden, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.embedding = nn.Linear(hidden, kernel_size[0] * kernel_size[1] * c)
        self.bias_conv = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, output_size):
        b_, _, _, _, c_ = x.shape
        feat = self.embedding(x.view(b_, -1, c_))
        b, _, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = F.fold(feat, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride,
                      padding=self.padding)
        return self.bias_conv(feat)


class FusionFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=1000, t2t=None):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(dim, hidden_dim))
        self.fc2 = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, dim))
        self.t2t = t2t
        self.kernel_shape = t2t["kernel_size"][0] * t2t["kernel_size"][1]

    def forward(self, x, output_size):
        p = self.t2t
        n_vecs = 1
        for i, d in enumerate(p["kernel_size"]):
            n_vecs *= int((output_size[i] + 2 * p["padding"][i] - (d - 1) - 1) / p["stride"][i] + 1)
        x = self.fc1(x)
        b, n, c = x.size()
        normalizer = x.new_ones(b, n, self.kernel_shape).view(-1, n_vecs, self.kernel_shape).permute(0, 2, 1)
        normalizer = F.fold(normalizer, output_size=output_size, kernel_size=p["kernel_size"], padding=p["padding"],
                            stride=p["stride"])
        x = F.fold(x.view(-1, n_vecs, c).permute(0, 2, 1), output_size=output_size, kernel_size=p["kernel_size"],
                   padding=p["padding"], stride=p["stride"])
        x = F.unfold(x / normalizer, kernel_size=p["kernel_size"], padding=p["padding"],
                     stride=p["stride"]).permute(0, 2, 1).contiguous().view(b, n, c)
        return self.fc2(x)


def sparse_partition(x, window_size, n_head):
    b, t, h, w, c = x.shape
    x = x.view(b, t, h // window_size[0], window_size[0], w // window_size[1], window_size[1], n_head, c // n_head)
    return x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()


def roll_with_zero_padding(x, shifts, dims):
    for dim, shift in zip(dims, shifts):
        size = x.size(dim)
        x = torch.roll(x, shift, dim)
        if shift > 0:
            x.index_fill_(dim, torch.arange(0, shift), 0)
        elif shift < 0:
            x.index_fill_(dim, torch.arange(size + shift, size), 0)
    return x


def sparse_attention_constants(window_size, device="cpu"):
    """(mask_padding, valid_ind_rolled): the buffers of the authors' SparseWindowAttention, built for its hard-coded
    32 x 32 token map (dropped from the weights, rebuilt here)."""
    def make():
        e = tuple((i + 1) // 2 for i in window_size)
        nh = math.ceil(32 / window_size[0]) * window_size[0]
        nw = math.ceil(32 / window_size[1]) * window_size[1]
        masks, pads = [], []
        for sy, sx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            m = torch.ones(window_size[0], window_size[1])
            m[(slice(None, -e[0]) if sy < 0 else slice(e[0], None)),
              (slice(None, -e[1]) if sx < 0 else slice(e[1], None))] = 0
            masks.append(m)
            pads.append(roll_with_zero_padding(torch.ones(nh, nw), (sy * e[0], sx * e[1]), (0, 1)))
        valid = torch.stack(masks, 0).flatten(0).nonzero(as_tuple=False).view(-1)
        return torch.stack(pads, 0).to(device), valid.to(device)
    return _const(("sparse", tuple(window_size), torch.device(device)), make)


class SparseWindowAttention(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size=(4, 4)):
        super().__init__()
        self.key = nn.Linear(dim, dim, True)
        self.query = nn.Linear(dim, dim, True)
        self.value = nn.Linear(dim, dim, True)
        self.proj = nn.Linear(dim, dim)
        self.n_head = n_head
        self.window_size = tuple(window_size)
        self.pool_layer = nn.Conv2d(dim, dim, kernel_size=pool_size, stride=pool_size, padding=(0, 0), groups=dim)
        self.expand_size = tuple((i + 1) // 2 for i in window_size)

    def forward(self, x, indices):
        """x (b, t, h, w, c); indices[i] = (masked queries, masked keys/values, unmasked queries) of patch i, the
        indices the authors take with qmask[i].nonzero(), kvmask[i].nonzero(), (1 - qmask[i]).nonzero()."""
        b, t, h, w, c = x.shape
        w_h, w_w = self.window_size
        c_head = c // self.n_head
        n_wh, n_ww = math.ceil(h / w_h), math.ceil(w / w_w)
        new_h, new_w = n_wh * w_h, n_ww * w_w
        pad_r, pad_b = new_w - w, new_h - h
        t_token = t // 4
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0), mode="constant", value=0)
        mask_padding, valid_ind_rolled = sparse_attention_constants(self.window_size, x.device)
        q, k, v = self.query(x), self.key(x), self.value(x)

        def part(a):
            return sparse_partition(a, self.window_size, self.n_head).view(b, n_wh * n_ww, self.n_head, t,
                                                                         w_h * w_w, c_head)
        win_q, win_k, win_v = part(q.contiguous()), part(k.contiguous()), part(v.contiguous())
        es = self.expand_size
        rolled_k, rolled_v = [], []
        for j, (sy, sx) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            m = mask_padding[j].view(1, 1, new_h, new_w, 1)
            rolled_k.append(part(torch.roll(k, shifts=(sy * es[0], sx * es[1]), dims=(2, 3)) * m))
            rolled_v.append(part(torch.roll(v, shifts=(sy * es[0], sx * es[1]), dims=(2, 3)) * m))
        rool_k = torch.cat(rolled_k, 4).contiguous()[:, :, :, :, valid_ind_rolled]
        rool_v = torch.cat(rolled_v, 4).contiguous()[:, :, :, :, valid_ind_rolled]
        roll_n = rool_k.shape[4]
        win_k = torch.cat((win_k, rool_k.view(b, n_wh * n_ww, self.n_head, t, roll_n, c_head)), dim=4)
        win_v = torch.cat((win_v, rool_v.view(b, n_wh * n_ww, self.n_head, t, roll_n, c_head)), dim=4)
        pool_x = self.pool_layer(x.view(b * t, new_h, new_w, c).permute(0, 3, 1, 2))
        _, _, p_h, p_w = pool_x.shape
        pool_x = pool_x.permute(0, 2, 3, 1).view(b, t, p_h, p_w, c)
        for proj, name in ((self.key, "k"), (self.value, "v")):
            pool = proj(pool_x).unsqueeze(1).repeat(1, n_wh * n_ww, 1, 1, 1, 1)
            pool = pool.view(b, n_wh * n_ww, t, p_h, p_w, self.n_head, c_head).permute(0, 1, 5, 2, 3, 4, 6)
            pool = pool.contiguous().view(b, n_wh * n_ww, self.n_head, t, p_h * p_w, c_head)
            if name == "k":
                win_k = torch.cat((win_k, pool), dim=4)
            else:
                win_v = torch.cat((win_v, pool), dim=4)

        twin_q = win_q.permute(0, 1, 3, 2, 4, 5).reshape(b, n_wh * n_ww * t, self.n_head, w_h * w_w, c_head)
        twin_k = win_k.reshape(b, n_wh * n_ww, self.n_head, t, -1, c_head)
        twin_v = win_v.reshape(b, n_wh * n_ww, self.n_head, t, -1, c_head)
        twin_k = twin_k.permute(0, 1, 3, 2, 4, 5).reshape(b, n_wh * n_ww * t, self.n_head, -1, c_head)
        twin_v = twin_v.permute(0, 1, 3, 2, 4, 5).reshape(b, n_wh * n_ww * t, self.n_head, -1, c_head)
        out = torch.zeros_like(twin_q)
        nt = w_h * w_w
        for i in range(b):
            mask_ind_i, kvmask_ind_i, unmask_ind_i = indices[i]
            mask_n = len(mask_ind_i)
            if mask_n > 0:   # masked windows: attention over the 12 selected frames, all tokens
                win_q_t = twin_q[i, mask_ind_i]
                win_k_t = twin_k[i, kvmask_ind_i].view(mask_n, self.n_head, -1, c_head)
                win_v_t = twin_v[i, kvmask_ind_i].view(mask_n, self.n_head, -1, c_head)
                kv_win_token = win_v_t.shape[-2]
                win_k_t = win_k_t.view(-1, t_token, self.n_head, kv_win_token, c_head)
                win_v_t = win_v_t.view(-1, t_token, self.n_head, kv_win_token, c_head)
                n_win = win_v_t.shape[0]
                win_k_t = win_k_t.permute(0, 2, 1, 3, 4).reshape(n_win, self.n_head, t_token * kv_win_token, c_head)
                win_v_t = win_v_t.permute(0, 2, 1, 3, 4).reshape(n_win, self.n_head, t_token * kv_win_token, c_head)
                win_q_t = win_q_t.view(n_win, t_token, self.n_head, nt, c_head)
                win_q_t = win_q_t.permute(0, 2, 1, 3, 4).reshape(n_win, self.n_head, t_token * nt, c_head)
                att_t = (win_q_t @ win_k_t.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_t.size(-1)))
                y_t = F.softmax(att_t, dim=-1) @ win_v_t
                y_t = y_t.view(n_win, self.n_head, t_token, nt, c_head)
                y_t = y_t.permute(0, 2, 1, 3, 4).reshape(n_win * t_token, self.n_head, nt, c_head)
                out[i, mask_ind_i] = y_t.view(-1, self.n_head, nt, c_head)
            n_un = len(unmask_ind_i)   # unmasked windows: groups of 12 consecutive (window, frame) tokens
            win_q_s = twin_q[i, unmask_ind_i]
            win_k_s = twin_k[i, unmask_ind_i, :, :nt].view(n_un, self.n_head, nt, c_head)
            win_v_s = twin_v[i, unmask_ind_i, :, :nt].view(n_un, self.n_head, nt, c_head)
            win_q_s = win_q_s.view(-1, t_token, self.n_head, nt, c_head)
            n_un_token = win_q_s.shape[0]
            win_q_s = win_q_s.permute(0, 2, 1, 3, 4).reshape(n_un_token, self.n_head, t_token * nt, c_head)
            win_k_s = win_k_s.view(-1, t_token, self.n_head, nt, c_head)
            win_k_s = win_k_s.permute(0, 2, 1, 3, 4).reshape(n_un_token, self.n_head, t_token * nt, c_head)
            win_v_s = win_v_s.view(-1, t_token, self.n_head, nt, c_head)
            win_v_s = win_v_s.permute(0, 2, 1, 3, 4).reshape(n_un_token, self.n_head, t_token * nt, c_head)
            att_s = (win_q_s @ win_k_s.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_s.size(-1)))
            y_s = F.softmax(att_s, dim=-1) @ win_v_s
            y_s = y_s.view(n_un_token, self.n_head, t_token, nt, c_head)
            out[i, unmask_ind_i] = y_s.permute(0, 2, 1, 3, 4).reshape(n_un_token * t_token, self.n_head, nt, c_head)
        out = out.view(b, n_wh, n_ww, t, self.n_head, w_h, w_w, c_head)
        out = out.permute(0, 3, 1, 5, 2, 6, 4, 7).contiguous().view(b, t, new_h, new_w, c)
        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :h, :w, :]
        return self.proj(out)


class TemporalSparseTransformer(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size, t2t):
        super().__init__()
        self.attention = SparseWindowAttention(dim, n_head, window_size, pool_size)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FusionFeedForward(dim, t2t=t2t)

    def forward(self, x, fold_x_size, indices):
        b, t, h, w, c = x.shape
        x = x + self.attention(self.norm1(x), indices)
        y = self.norm2(x)
        return x + self.mlp(y.view(b, t * h * w, c), fold_x_size).view(b, t, h, w, c)


class TemporalSparseTransformerBlock(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size, depths, t2t):
        super().__init__()
        self.transformer = nn.Sequential(*[TemporalSparseTransformer(dim, n_head, window_size, pool_size, t2t)
                                           for _ in range(depths)])
        self.depths = depths

    def forward(self, x, fold_x_size, qmasks, kvmasks):
        b, t = x.shape[:2]
        n_win = math.ceil(x.shape[2] / 8) * math.ceil(x.shape[3] / 8)
        # the authors' qmask.view(b, t, windows).permute(0, 2, 1).reshape(b, windows * t), per depth; the indices of
        # every depth and patch at once (one sync instead of two per depth and patch)
        flat = [torch.stack([m.view(b, t, n_win).permute(0, 2, 1).reshape(b, n_win * t) for m in ms])
                for ms in (qmasks, kvmasks)]
        sel = torch.stack((flat[0] != 0, flat[1] != 0, (1 - flat[0]) != 0))          # (3, depth, b, windows * t)
        counts = sel.sum(-1).flatten().tolist()
        parts = torch.split(sel.nonzero(as_tuple=False)[:, -1], counts)
        d = len(qmasks)
        for i in range(self.depths):
            idx = [tuple(parts[(j * d + i) * b + p] for j in range(3)) for p in range(b)]
            x = self.transformer[i](x, fold_x_size, idx)
        return x


# --------------------------------------------------------------------------------------------------------- BSST
class BSST(nn.Module):
    """The BSST network of the authors (BSST_arch.py, defaults of gopro_BSST.yml) for a batch of patches:
    forward(lqs (n, t, 3, h, w), flows_forward, flows_backward (n, t-1, 2, h/4, w/4)) -> (n, t, 3, h, w)."""

    def __init__(self, mid=192, n_feat=64):
        super().__init__()
        self.mid_channels = mid
        self.act = lrelu()
        self.concat = nn.Sequential(nn.Conv2d(3, n_feat, 3, 1, 1, bias=False), lrelu(), CAB(n_feat))
        self.down01 = nn.Sequential(nn.Conv2d(n_feat, n_feat, 3, 2, 1, bias=False), lrelu())
        self.encoder_level1 = CAB(n_feat)
        self.encoder_level1_1 = CAB(n_feat)
        self.down12 = nn.Sequential(nn.Conv2d(n_feat, mid, 3, 2, 1, bias=False), lrelu())
        self.encoder_level2 = CAB(mid)
        self.encoder_level2_1 = CAB(mid)
        self.decoder_level1 = ShiftBlock(n_feat)
        self.decoder_level1_1 = ShiftBlock(n_feat)
        self.skip_attn1 = CAB(n_feat)
        self.upsample0 = PixelShufflePack(n_feat, n_feat, 2, 3)
        self.skip_conv = CAB(n_feat)
        self.out_conv = CAB(n_feat)
        self.last_conv = conv(n_feat, 3, 3)
        self.conv_hr0 = conv(n_feat, n_feat, 3)
        self.up21 = SkipUpSample(n_feat, 128)
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        for i, m in enumerate(("backward_1", "forward_1", "backward_2", "forward_2")):
            self.deform_align[m] = BlurGuidedAlignment(2 * mid, mid)
            self.backbone[m] = nn.Sequential(nn.Conv2d((2 + i) * mid, mid, 3, 1, 1), lrelu(), CAB1(mid, 3))
        self.reconstruction = Reconstruction(5 * mid, mid)
        t2t = {"kernel_size": (5, 5), "stride": (2, 2), "padding": (2, 2)}
        self.depths = 4
        self.fold_feat_size = (64, 64)
        self.softsplit = SoftSplit(mid, 512, (5, 5), (2, 2), (2, 2))
        self.softcomp = SoftComp(mid, 512, (5, 5), (2, 2), (2, 2))
        self.transsqattn = TemporalSparseTransformerBlock(512, 4, (8, 8), (4, 4), self.depths, t2t)
        self.cabattn = nn.Sequential(CAB2(mid, 5, mid), CAB1(mid, 5))
        self.blur_motion_refine = FlowCompletion()

    def propagate(self, feats, flows, flows_check, blur_motion_map, module_name, out):
        """One propagation branch. The features are written into out (n, t+1, 192, h, w), a channel slice of the
        float32 buffer that the authors build afterwards with stack + cat; feats keeps views of it."""
        n, t, _, h, w = flows.size()
        last_key = list(feats)[-2]
        if "backward" in module_name:
            frame_idx = flow_idx = range(t, -1, -1)
        else:
            frame_idx, flow_idx = range(0, t + 1), range(-1, t)
        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats[last_key][idx]
            blur_map_current = blur_motion_map[:, frame_idx[i], :, :, :]
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                flow_n1_check = flows_check[:, flow_idx[i], :, :, :]
                blur_map_n1 = blur_motion_map[:, frame_idx[i - 1], :, :, :]
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                blur_map_n1 = flow_warp(blur_map_n1, flow_n1.permute(0, 2, 3, 1))
                flow_n1_valid = fb_valid(flow_n1, flow_n1_check)
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
                blur_map_n2 = torch.zeros_like(blur_map_n1)
                flow_n2_valid = torch.zeros_like(flow_n1_valid)
                if i > 1:   # second-order features
                    feat_n2 = feats[module_name][-2]
                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    flow_n2_check = flows_check[:, flow_idx[i - 1], :, :, :]
                    blur_map_n2 = blur_motion_map[:, frame_idx[i - 2], :, :, :]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    flow_n2_check = flow_n1_check + flow_warp(flow_n2_check, flow_n1_check.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
                    blur_map_n2 = flow_warp(blur_map_n2, flow_n2.permute(0, 2, 3, 1))
                    flow_n2_valid = fb_valid(flow_n2, flow_n2_check)
                cond = torch.cat([cond_n1, feat_current, cond_n2, blur_map_n1, blur_map_current, blur_map_n2,
                                  flow_n1_valid, flow_n2_valid], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2,
                                                           blur_map_n1 * flow_n1_valid, blur_map_n2 * flow_n2_valid)
            feat = torch.cat([feats[k][idx] for k in feats if k != module_name] + [feat_prop], dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            out[:, idx].copy_(feat_prop)
            feats[module_name].append(out[:, idx])
        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]
        return feats

    def sparse_masks(self, blur_map_st):
        """Query and key/value masks of the 4 sparse attention depths, (n, t, 4, 4, 1) each, as the authors compute
        them, including the in-place zeroing of blur_map_st_win through its alias."""
        n, t, c, h, w = blur_map_st.shape
        win = F.avg_pool2d(blur_map_st.view(n * t, c, h, w), (5, 5), (2, 2), (2, 2))
        win = F.avg_pool2d(win, (8, 8), (8, 8), (0, 0))
        win = win.view(n * t, 1, win.shape[-2], win.shape[-1]).view(n, t, 1, win.shape[-2], win.shape[-1])
        hard = torch.where(win > 0.3, torch.ones_like(win), torch.zeros_like(win))
        dev = blur_map_st.device
        t_ind = [torch.arange(i, t, 2, device=dev) for i in range(2)] * (self.depths // 2)
        qmasks, kvmasks = [], []
        for t_idxs, t_zero in zip(t_ind, t_ind[::-1]):
            mask = hard[:, t_idxs].sum(1, keepdim=True).clip(0, 1)
            q_soft = win                       # alias: the zeroing below stays in win for the next depths
            q_soft[:, t_zero] = q_soft[:, t_zero] * 0.
            q_idx = torch.argsort(q_soft, 1, True)
            qmask = torch.zeros_like(win)
            kv_soft = (1 - win)
            kv_soft[:, t_zero] = kv_soft[:, t_zero] * 0.
            kv_idx = torch.argsort(kv_soft, 1, True)
            kvmask = torch.zeros_like(win)
            kvmask = kvmask.scatter_(1, kv_idx[:, :len(t_idxs) // 2], 1) * mask
            qmask = qmask.scatter_(1, q_idx[:, :len(t_idxs) // 2], 1) * mask
            qmasks.append(qmask.permute(0, 1, 3, 4, 2).contiguous())
            kvmasks.append(kvmask.permute(0, 1, 3, 4, 2).contiguous())
        return qmasks, kvmasks

    def forward(self, lqs, flows_forwards_gt, flows_backwards_gt, stage_batch=None):
        """stage_batch: patches per sub-batch of the feed-forward stages (encoder, reconstruction, sparse transformer,
        decoder), which bounds the memory; the recurrent parts (flow completion, the four propagation branches) run on
        all n patches at once. None: all n."""
        n, t, _, h, w = lqs.size()
        sb = stage_batch or n
        parts = [slice(s, min(s + sb, n)) for s in range(0, n, sb)]
        pred_map = get_blur_map(flows_forwards_gt, flows_backwards_gt)
        flows_forward, flows_backward = self.blur_motion_refine.forward_bidirect_flow(
            pred_map, flows_forwards_gt, flows_backwards_gt)
        blur_map_st = get_blur_map(flows_forward, flows_backward)
        shortcuts, enc11s, hr = [], [], None
        for sl in parts:
            x = self.concat(lqs[sl].flatten(0, 1))     # the authors: lqs[0]
            enc11 = self.encoder_level1_1(self.encoder_level1(self.down01(x)))
            enc22 = self.encoder_level2_1(self.encoder_level2(self.down12(enc11)))
            shortcuts.append(x)
            enc11s.append(enc11)
            if hr is None:   # the authors' torch.cat of [spatial, backward_1, forward_1, backward_2, forward_2]
                hr = torch.empty(n, t, 5 * self.mid_channels, *enc22.shape[-2:], device=lqs.device,
                                 dtype=torch.promote_types(enc22.dtype, flows_forward.dtype))
            hr[sl, :, :self.mid_channels].copy_(enc22.view(-1, t, *enc22.shape[1:]))
            del x, enc11, enc22
        # the spatial features enter the propagation only through torch.cat with float32 tensors, so their float32
        # copies in hr give the same values as the authors' float16 originals
        feats = {"spatial": [hr[:, i, :self.mid_channels] for i in range(t)]}
        blur_motion = 1 - blur_map_st
        for k, (iter_, direction) in enumerate(((1, "backward"), (1, "forward"), (2, "backward"), (2, "forward"))):
            flows, check = ((flows_backward, flows_forward) if direction == "backward"
                            else (flows_forward, flows_backward))
            name = f"{direction}_{iter_}"
            feats[name] = []
            c0 = (k + 1) * self.mid_channels
            feats = self.propagate(feats, flows, check, blur_motion, name, hr[:, :, c0:c0 + self.mid_channels])
        del feats
        qmasks, kvmasks = self.sparse_masks(blur_map_st)
        outs = []
        for p, sl in enumerate(parts):
            shortcut, enc11 = shortcuts[p], enc11s[p]
            shortcuts[p] = enc11s[p] = None
            nc = sl.stop - sl.start
            input_feats = self.reconstruction(hr[sl])
            b, t, c, h4, w4 = input_feats.shape
            trans = self.softsplit(input_feats.view(b * t, c, h4, w4), nc, self.fold_feat_size)
            trans = self.transsqattn(trans, self.fold_feat_size, [m[sl] for m in qmasks], [m[sl] for m in kvmasks])
            trans = self.softcomp(trans, t, self.fold_feat_size).view(b, t, c, h4, w4)
            trans = self.cabattn(torch.cat([input_feats, trans], 2).flatten(0, 1))   # the authors: [0], unsqueeze(0)
            del input_feats
            x = self.up21(trans, self.skip_attn1(enc11))
            del trans
            dec11 = self.decoder_level1_1(self.decoder_level1(x, nc), nc)
            del x
            out = self.conv_hr0(self.act(self.upsample0(dec11))) + self.skip_conv(shortcut)
            out = self.last_conv(self.out_conv(out))
            outs.append(out.view(nc, t, *out.shape[1:]) + lqs[sl])
            del dec11, out, shortcut, enc11
        return outs[0] if len(outs) == 1 else torch.cat(outs)


# --------------------------------------------------------------------------------------------------------- RAFT
def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij")
    return torch.stack(coords[::-1], dim=0).float()[None].repeat(batch, 1, 1, 1)


def bilinear_sampler(img, coords):
    h, w = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (w - 1) - 1
    ygrid = 2 * ygrid / (h - 1) - 1
    return F.grid_sample(img, torch.cat([xgrid, ygrid], dim=-1), align_corners=True)


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels, self.radius = num_levels, radius
        batch, dim, ht, wd = fmap1.shape
        corr = torch.matmul(fmap1.view(batch, dim, ht * wd).transpose(1, 2), fmap2.view(batch, dim, ht * wd))
        corr = corr.view(batch, ht, wd, 1, ht, wd) / torch.sqrt(torch.tensor(dim).float())
        corr = corr.reshape(batch * ht * wd, 1, ht, wd)
        self.corr_pyramid = [corr]
        for _ in range(num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        def make():   # the authors build it on the CPU and copy it at every call
            d = torch.linspace(-r, r, 2 * r + 1)
            return torch.stack(torch.meshgrid(d, d, indexing="ij"), dim=-1).to(coords.device)
        delta = _const(("corr_delta", r, coords.device), make).view(1, 2 * r + 1, 2 * r + 1, 2)
        out = []
        for i in range(self.num_levels):
            centroid = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            corr = bilinear_sampler(self.corr_pyramid[i], centroid + delta)
            out.append(corr.view(batch, h1, w1, -1))
        return torch.cat(out, dim=-1).permute(0, 3, 1, 2).contiguous().float()


def raft_norm(kind, c):
    return nn.BatchNorm2d(c) if kind == "batch" else nn.InstanceNorm2d(c)


class ResidualBlock(nn.Module):
    """RAFT's ResidualBlock; its norm3 is the second module of downsample (the authors also register it as norm3)."""

    def __init__(self, cin, c, norm_fn, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, c, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.norm1, self.norm2 = raft_norm(norm_fn, c), raft_norm(norm_fn, c)
        self.downsample = (None if stride == 1 else
                           nn.Sequential(nn.Conv2d(cin, c, kernel_size=1, stride=stride), raft_norm(norm_fn, c)))

    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim, norm_fn):
        super().__init__()
        self.norm1 = raft_norm(norm_fn, 64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(ResidualBlock(64, 64, norm_fn, 1), ResidualBlock(64, 64, norm_fn, 1))
        self.layer2 = nn.Sequential(ResidualBlock(64, 96, norm_fn, 2), ResidualBlock(96, 96, norm_fn, 1))
        self.layer3 = nn.Sequential(ResidualBlock(96, 128, norm_fn, 2), ResidualBlock(128, 128, norm_fn, 1))
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

    def forward(self, x):
        is_list = isinstance(x, (tuple, list))
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.conv2(self.layer3(self.layer2(self.layer1(x))))
        return torch.split(x, [batch_dim, batch_dim], dim=0) if is_list else x


class BasicMotionEncoder(nn.Module):
    def __init__(self, cor_planes=4 * 81):
        super().__init__()
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc2(F.relu(self.convc1(corr))))
        flo = F.relu(self.convf2(F.relu(self.convf1(flow))))
        out = F.relu(self.conv(torch.cat([cor, flo], dim=1)))
        return torch.cat([out, flow], dim=1)


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256):
        super().__init__()
        c = hidden_dim + input_dim
        self.convz1 = nn.Conv2d(c, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(c, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(c, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(c, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(c, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(c, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        for cz, cr, cq in ((self.convz1, self.convr1, self.convq1), (self.convz2, self.convr2, self.convq2)):
            hx = torch.cat([h, x], dim=1)
            z = torch.sigmoid(cz(hx))
            r = torch.sigmoid(cr(hx))
            q = torch.tanh(cq(torch.cat([r * h, x], dim=1)))
            h = (1 - z) * h + z * q
        return h


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = BasicMotionEncoder()
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow, with_mask):
        inp = torch.cat([inp, self.encoder(flow, corr)], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        return net, (.25 * self.mask(net) if with_mask else None), delta_flow


class RAFT(nn.Module):
    """RAFT (large, raft-things weights) as vendored by BSSTNet: forward(image1, image2) in [0, 1] -> the flow of the
    last of 20 iterations, upsampled to the input size (test_mode=True)."""

    def __init__(self):
        super().__init__()
        self.hidden_dim = self.context_dim = 128
        self.fnet = BasicEncoder(256, "instance")
        self.cnet = BasicEncoder(256, "batch")
        self.update_block = BasicUpdateBlock(hidden_dim=128)

    @staticmethod
    def upsample_flow(flow, mask):
        n, _, h, w = flow.shape
        mask = torch.softmax(mask.view(n, 1, 9, 8, 8, h, w), dim=2)
        up_flow = F.unfold(8 * flow, [3, 3], padding=1).view(n, 2, 9, 1, 1, h, w)
        up_flow = torch.sum(mask * up_flow, dim=2).permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(n, 2, 8 * h, 8 * w)

    def forward(self, image1, image2, iters=20):
        image1 = (2 * (image1 / 1.) - 1.0).contiguous()
        image2 = (2 * (image2 / 1.) - 1.0).contiguous()
        fmap1, fmap2 = self.fnet([image1, image2])
        corr_fn = CorrBlock(fmap1, fmap2, radius=4)
        net, inp = torch.split(self.cnet(image1), [self.hidden_dim, self.context_dim], dim=1)
        net, inp = torch.tanh(net), torch.relu(inp)
        n, _, h, w = image1.shape
        coords0 = coords_grid(n, h // 8, w // 8, image1.device)
        coords1 = coords_grid(n, h // 8, w // 8, image1.device)
        for itr in range(iters):   # (the authors upsample the flow at every iteration and return the last one)
            corr = corr_fn(coords1)
            net, up_mask, delta_flow = self.update_block(net, inp, corr, coords1 - coords0, itr == iters - 1)
            coords1 = coords1 + delta_flow
        return self.upsample_flow(coords1 - coords0, up_mask)


# ---------------------------------------------------------------------------------------------------- the method
class BSSTNet(nn.Module):
    """RAFT flows + BSST on overlapping patches of a window of frames (the authors' test_by_patch).

    forward(lqs (B, T, 3, H, W) in [0, 1], H and W multiples of 16, >= 256) -> (B, T, 3, H, W) float32 on lqs.device.
    patch_batch: patches per BSST call (1 = one patch per call as the authors, the same outputs bit for bit; default 14
    = two calls for a 720p window); stage_batch: patches per sub-batch of the feed-forward stages inside a call (memory
    bound, see BSST.forward); precision: "fp16" (the authors' autocast) or "fp32". All three can also be given to
    forward(). nan_fallback (dbisf): a patch whose float16 output has a NaN or Inf (float16 overflow; the authors' code
    then writes a black 256 px tile, on 15 % of the REDS and 13 % of the BSD frames with the GoPro weights) is computed
    again in float32; self.fallbacks counts those patches. Values that stay non-finite in float32 too come from the
    recurrent propagation, whose features grow until a LayerNorm overflows (out of domain; the authors' float32 code
    gives the same black tiles): with nan_fallback those pixels take the input frame (blurry, not black);
    self.input_fallbacks counts the patches."""

    def __init__(self, patch_batch=14, precision="fp16", stage_batch=4, patch=256, overlap=64, nan_fallback=False):
        super().__init__()
        self.bsst = BSST()
        self.raft = RAFT()
        self.patch_batch, self.precision, self.stage_batch = patch_batch, precision, stage_batch
        self.patch, self.overlap, self.nan_fallback = patch, overlap, nan_fallback
        self.fallbacks = self.input_fallbacks = 0

    def flows(self, lq):
        """get_bi_flows of the authors: (forward, backward) flows (B, T-1, 2, H/4, W/4), RAFT in float32."""
        b, t, c, h, w = lq.shape
        with torch.autocast(lq.device.type, enabled=False):
            lq1 = F.interpolate(lq[:, :-1, ...].reshape(b * (t - 1), c, h, w), scale_factor=0.5, mode="bilinear",
                                align_corners=False)
            lq2 = F.interpolate(lq[:, 1:, ...].reshape(b * (t - 1), c, h, w), scale_factor=0.5, mode="bilinear",
                                align_corners=False)
            fwd = self.raft(lq2, lq1)
            bwd = self.raft(lq1, lq2)
            fwd = F.interpolate(fwd, scale_factor=0.5, mode="bilinear", align_corners=False)
            bwd = F.interpolate(bwd, scale_factor=0.5, mode="bilinear", align_corners=False)
        return fwd.view(b, t - 1, 2, h // 4, w // 4) / 2, bwd.view(b, t - 1, 2, h // 4, w // 4) / 2

    def band_mask(self, i, j, hs, ws, device):
        """1 inside, 0 on the 32 px bands the authors zero (sides that have a neighbouring patch)."""
        p, half = self.patch, self.overlap // 2
        m = torch.ones(p, p, device=device)
        if i < hs[-1]:
            m[-half:, :] = 0
        if j < ws[-1]:
            m[:, -half:] = 0
        if i > hs[0]:
            m[:half, :] = 0
        if j > ws[0]:
            m[:, :half] = 0
        return m

    @torch.no_grad()
    def forward(self, lqs, patch_batch=None, precision=None, stage_batch=None):
        pb = patch_batch or self.patch_batch
        sb = stage_batch or self.stage_batch
        precision = precision or self.precision
        b, t, c, h, w = lqs.shape
        if h % 16 or w % 16 or h < self.patch or w < self.patch:
            raise ValueError(f"BSSTNet needs H and W multiples of 16 and at least {self.patch}, got {h} x {w}")
        flows_f, flows_b = self.flows(lqs)
        p, stride, q = self.patch, self.patch - self.overlap, self.patch // 4
        hs = list(range(0, h - p, stride)) + [max(0, h - p)]
        ws = list(range(0, w - p, stride)) + [max(0, w - p)]
        coords = [(i, j) for i in hs for j in ws]
        acc = torch.zeros(b, t, c, h, w, device=lqs.device)
        cnt = torch.zeros_like(acc)
        for k in range(0, len(coords), pb):
            chunk = coords[k:k + pb]
            x = torch.cat([lqs[..., i:i + p, j:j + p] for i, j in chunk])
            ff = torch.cat([flows_f[..., i // 4:i // 4 + q, j // 4:j // 4 + q] for i, j in chunk])
            fb = torch.cat([flows_b[..., i // 4:i // 4 + q, j // 4:j // 4 + q] for i, j in chunk])
            with torch.autocast(lqs.device.type, dtype=torch.float16, enabled=precision == "fp16"):
                y = self.bsst(x, ff, fb, stage_batch=sb)
            if precision == "fp16" and self.nan_fallback:
                bad = (~torch.isfinite(y.reshape(len(chunk), -1)).all(1)).nonzero().flatten().tolist()   # one sync per call
                if bad:
                    y = y.float()
                    sel = [m * b + r for m in bad for r in range(b)]
                    y[sel] = self.bsst(x[sel], ff[sel], fb[sel], stage_batch=sb).float()       # float32, no autocast
                    self.fallbacks += len(bad)
                    nf = ~torch.isfinite(y)
                    if bool(nf.any()):                   # still not finite in float32: the input there, not black
                        y = torch.where(nf, x.to(y.dtype), y)
                        self.input_fallbacks += int(nf.reshape(len(chunk), -1).any(1).sum())
            del x, ff, fb
            for m, (i, j) in enumerate(chunk):   # the authors' order of the sums
                mask = self.band_mask(i, j, hs, ws, lqs.device)
                acc[..., i:i + p, j:j + p].add_(y[m * b:(m + 1) * b] * mask)
                cnt[..., i:i + p, j:j + p].add_(mask)
            del y
        return acc.div_(cnt)


# ---------------------------------------------------------------------------------------------------- weights
UNUSED = ("feat_extract.", "decoder_level1_2.", "decoder_level2.", "decoder_level2_1.", "decoder_level2_2.",
          "conv_before_upsampler.", "conv_before_input.", "upsampler.", "conv_last.")


def fold(sd, raft_sd):
    """The authors' state dicts (key "params" of BSST_gopro.pth, and raft-things.pth) -> the state dict of BSSTNet:
    prefixes bsst. and raft., without "module.", without the modules the forward never calls (UNUSED) and without the
    index buffers and the duplicate norm3 of RAFT, which are checked against what this file rebuilds / keeps."""
    def strip(d):
        return {(k[7:] if k.startswith("module.") else k): v for k, v in d.items()}
    sd, raft_sd = strip(sd), strip(raft_sd)
    out = {}
    for k, v in sd.items():
        if k.startswith(UNUSED):
            continue
        if k.endswith("relative_position_index"):
            assert torch.equal(v, relative_position_index((2, 8, 8))), k
            continue
        if k.endswith(("mask_padding", "valid_ind_rolled")):
            mp, vi = sparse_attention_constants((8, 8))
            assert torch.equal(v, mp if k.endswith("mask_padding") else vi), k
            continue
        out["bsst." + k] = v
    for k, v in raft_sd.items():
        if ".norm3." in k:
            assert torch.equal(v, raft_sd[k.replace(".norm3.", ".downsample.1.")]), k
            continue
        out["raft." + k] = v
    return {k: (v.float() if v.is_floating_point() else v).clone().contiguous() for k, v in out.items()}
