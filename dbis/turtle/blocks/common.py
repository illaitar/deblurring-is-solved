"""Turtle's plain blocks: the layer norm over channels, the resizing convs and the two feed-forward networks.

Source: basicsr/models/archs/turtle_t1_arch.py of https://github.com/Ascend-Research/Turtle (commit 7094f42), MIT
License, Copyright (c) 2024 Huawei Technologies Co., Ltd. The same modules and weight names. The einops rearranges of
the authors are written as the reshape and permute they stand for (the same strides, so the same kernels and values).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_3d(x):
    """'b c h w -> b (h w) c' (a view)."""
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def to_4d(x, h, w):
    """'b (h w) c -> b c h w' (a view)."""
    b, _, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Normalizes every pixel over its channels (Restormer's layer norm); the weights sit in .body as in the authors'."""

    def __init__(self, dim, LayerNorm_type="WithBias"):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == "BiasFree" else WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Downsample(nn.Module):
    """3x3 conv to half the channels, then pixel unshuffle: half the size, twice the channels."""

    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False), nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """3x3 conv to twice the channels, then pixel shuffle: twice the size, half the channels."""

    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False), nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class GatedFeedForward(nn.Module):
    """GFFW: 1x1 conv to 2 x hidden, 3x3 depthwise conv, GELU(first half) x second half, 1x1 conv back."""

    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class FeedForward(nn.Module):
    """FFW: 1x1 conv to 2 dim, GELU, 1x1 conv back, times the per-channel scale gamma (its expansion is always 2)."""

    def __init__(self, c, FFN_Expand=2):
        super().__init__()
        self.conv4 = nn.Conv2d(c, FFN_Expand * c, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(FFN_Expand * c, c, 1, 1, 0, bias=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp):
        return self.conv5(F.gelu(self.conv4(inp))) * self.gamma
