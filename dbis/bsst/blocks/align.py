"""Second-order deformable alignment (BasicVSR++ lineage, as in BSSTNet; the flow completion variant from ProPainter (Zhou et al., ICCV 2023, https://github.com/sczhou/ProPainter), reused by BSSTNet. ProPainter's own code is under
the S-Lab License 1.0 (non-commercial use only)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from .common import _const, lrelu


def _dcn_gs(x, offset, weight, bias, mask):
    """Modulated deformable 3x3 conv (stride 1, padding 1), the function of torchvision's deform_conv2d: the 9 sampled
    copies of x (per offset group) by grid_sample in the dtype autocast gives it (float32), then the 3x3 sum as one
    1x1 conv over cin * 9 channels, which runs in float16 on tensor cores under autocast (torchvision: float32 GEMM)."""
    n, cin, h, w = x.shape
    g = mask.shape[1] // 9
    dt = torch.float32
    key = ("dcn_base", h, w, x.device)
    by, bx = _const(key, lambda: (
        (torch.arange(h, device=x.device, dtype=dt).view(1, 1, 1, h, 1)
         + torch.arange(-1, 2, device=x.device, dtype=dt).repeat_interleave(3).view(1, 1, 9, 1, 1)),
        (torch.arange(w, device=x.device, dtype=dt).view(1, 1, 1, 1, w)
         + torch.arange(-1, 2, device=x.device, dtype=dt).repeat(3).view(1, 1, 9, 1, 1))))
    off = offset.view(n, g, 9, 2, h, w).to(dt)                      # torchvision: (dy, dx) of kernel point i * 3 + j
    grid = torch.stack((2 * (bx + off[:, :, :, 1]) / (w - 1) - 1, 2 * (by + off[:, :, :, 0]) / (h - 1) - 1), -1)
    s = F.grid_sample(x.to(dt).reshape(n * g, cin // g, h, w), grid.view(n * g, 9 * h, w, 2), mode="bilinear",
                      padding_mode="zeros", align_corners=True)
    s = s.view(n, g, cin // g, 9, h, w) * mask.view(n, g, 1, 9, h, w).to(dt)
    return F.conv2d(s.view(n, cin * 9, h, w), weight.reshape(weight.shape[0], cin * 9, 1, 1), bias).float()


def _dcn(kind, x, offset, weight, bias, mask):
    """kind "tv": torchvision (the authors' patched code; float32 inside autocast); "tv_half": torchvision in float16;
    "gs": _dcn_gs."""
    if kind == "gs":
        return _dcn_gs(x, offset, weight, bias, mask)
    if kind == "tv_half":
        with torch.autocast(x.device.type, enabled=False):
            return deform_conv2d(x.half(), offset.half(), weight.half(), bias.half(), (1, 1), (1, 1), (1, 1),
                                 mask.half()).float()
    return deform_conv2d(x, offset, weight, bias, (1, 1), (1, 1), (1, 1), mask)


class BlurGuidedAlignment(nn.Module):
    """SecondOrderDeformableAlignmentWithBlurmapguide: offsets = flow + 5 tanh(.), masks shifted by the blur maps."""
    dcn = "tv"

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
        return _dcn(self.dcn, x, offset, self.weight, self.bias, mask)


class FlowAlignment(nn.Module):
    """SecondOrderDeformableAlignment of ProPainter's flow completion net (mmcv DCN -> torchvision, as patched)."""
    dcn = "tv"

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
        return _dcn(self.dcn, x, offset, self.weight, self.bias, torch.sigmoid(mask))
