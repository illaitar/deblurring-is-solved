"""The NAFBlock (Chen et al., ECCV 2022), its normalisation without scale and shift (folded into the next conv) and
the window mean of TLC (Chu et al., ECCV 2022), exact or on a cell map (dbisf)."""
import torch.nn as nn
import torch.nn.functional as F


def norm(x, eps=1e-6):
    """LayerNorm over the channels of an NCHW tensor, without scale and shift (folded into the next conv)."""
    mu = x.mean(1, keepdim=True)
    var = (x - mu).pow(2).mean(1, keepdim=True)
    return (x - mu) / (var + eps).sqrt()


def window_mean(x, k):
    """Mean over the k x k window of every pixel, as the authors compute it: box sums from an integral image, the valid
    part padded back to the input size by replication."""
    h, w = x.shape[-2:]
    k1, k2 = min(h, k), min(w, k)
    s = F.pad(x.cumsum(-1).cumsum_(-2), (1, 0, 1, 0))
    out = (s[..., k1:, k2:] + s[..., :-k1, :-k2] - s[..., :-k1, k2:] - s[..., k1:, :-k2]) / (k1 * k2)
    return F.pad(out, ((k2 - 1) // 2, k2 // 2, (k1 - 1) // 2, k1 // 2), mode="replicate")


def cell(k, h, w):
    """Cell size for dbisf: the largest of 8, 4, 2 that divides the map and keeps at least 12 cells per window; 1 = exact."""
    return next((r for r in (8, 4, 2) if k // r >= 12 and h % r == 0 and w % r == 0), 1)


class Block(nn.Module):
    """NAFBlock: norm, 1x1 conv, 3x3 depthwise conv, gate, channel attention, 1x1 conv; then norm, 1x1, gate, 1x1."""

    def __init__(self, c, k=None, fast=False):
        super().__init__()
        self.conv1 = nn.Conv2d(c, 2 * c, 1)
        self.conv2 = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sca = nn.Conv2d(c, c, 1)
        self.conv3 = nn.Conv2d(c, c, 1)
        self.conv4 = nn.Conv2d(c, 2 * c, 1)
        self.conv5 = nn.Conv2d(c, c, 1)
        self.k, self.fast = k, fast

    def attention(self, x):
        h, w = x.shape[-2:]
        if self.k is None or (self.k >= h and self.k >= w):
            return self.sca(F.adaptive_avg_pool2d(x, 1))
        r = cell(self.k, h, w) if self.fast else 1
        if r == 1:
            return self.sca(window_mean(x, self.k))
        a = self.sca(window_mean(F.avg_pool2d(x, r), self.k // r))
        return F.interpolate(a, scale_factor=r, mode="bilinear", align_corners=False)

    def forward(self, inp):
        a, b = self.conv2(self.conv1(norm(inp))).chunk(2, 1)
        x = a * b
        y = inp + self.conv3(x * self.attention(x))
        a, b = self.conv4(norm(y)).chunk(2, 1)
        return y + self.conv5(a * b)
