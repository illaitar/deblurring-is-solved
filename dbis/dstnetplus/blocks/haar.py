"""The Haar wavelet of DSTNet+ (HaarDownsampling of basicsr/archs/dstnetplus_deblur_arch.py): a fixed 2x2 grouped conv
that splits a feature map into its low band and three high bands, and the transposed conv that puts them back."""
import torch
import torch.nn as nn
import torch.nn.functional as F

_W = {}


def haar_weights(c, device=None):
    """The authors' fixed weights (4 c, 1, 2, 2), float32, cached per channel count and device."""
    key = (c, torch.device(device) if device is not None else None)
    if key not in _W:
        w = torch.ones(4, 1, 2, 2)
        w[1, 0, 0, 1] = w[1, 0, 1, 1] = -1
        w[2, 0, 1, 0] = w[2, 0, 1, 1] = -1
        w[3, 0, 1, 0] = w[3, 0, 0, 1] = -1
        _W[key] = torch.cat([w] * c, 0).to(device)
    return _W[key]


class Haar(nn.Module):
    """forward(x (n, c, h, w)) -> (low (n, c, h/2, w/2), high (n, 3 c, h/2, w/2)); inverse(low and high concatenated,
    (n, 4 c, h/2, w/2)) -> (n, c, h, w). The authors keep the fixed weights as a parameter (wave.haar_weights); fold()
    checks them against haar_weights() and drops them."""

    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x):
        n, _, h, w = x.shape
        out = F.conv2d(x, haar_weights(self.c, x.device), bias=None, stride=2, groups=self.c) / 4.0
        out = torch.transpose(out.reshape([n, self.c, 4, h // 2, w // 2]), 1, 2).reshape([n, self.c * 4, h // 2, w // 2])
        return out[:, :self.c], out[:, self.c:]

    def inverse(self, x):
        n, _, h, w = x.shape
        out = torch.transpose(x.reshape([n, 4, self.c, h, w]), 1, 2).reshape([n, self.c * 4, h, w])
        return F.conv_transpose2d(out, haar_weights(self.c, x.device), bias=None, stride=2, groups=self.c)
