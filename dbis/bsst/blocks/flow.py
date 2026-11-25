"""Flow helpers of BSST (BSST_arch.py): bilinear warping, forward-backward consistency, the blur map."""
import torch
import torch.nn.functional as F

from .common import _const


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
