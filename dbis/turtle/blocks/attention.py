"""Turtle's attention blocks without history: transposed channel attention and the conv block that stands in for it.

Source: basicsr/models/archs/turtle_t1_arch.py of https://github.com/Ascend-Research/Turtle (commit 7094f42), MIT
License, Copyright (c) 2024 Huawei Technologies Co., Ltd. The same modules and weight names. Every block returns
(out, None, None), as the authors' do, so that a Turtle block calls any attention the same way.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def heads(t, n):
    """'b (head c) h w -> b head c (h w)'."""
    b, c, h, w = t.shape
    return t.reshape(b, n, c // n, h * w)


def unheads(t, h, w):
    """'b head c (h w) -> b (head c) h w'."""
    b, n, c, _ = t.shape
    return t.reshape(b, n * c, h, w)


class ChannelAttention(nn.Module):
    """Restormer's transposed attention: per head, a c x c attention over the channels of the whole input."""

    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x, k_cached=None, v_cached=None):
        _, _, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q, k, v = heads(q, self.num_heads), heads(k, self.num_heads), heads(v, self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        return self.project_out(unheads(attn @ v, h, w)), None, None


class ReducedAttn(nn.Module):
    """NAFNet-like conv block in the place of attention: 1x1 conv to 2 dim, 3x3 depthwise, GELU, 1x1 back, times beta."""

    def __init__(self, c, DW_Expand=2.0):
        super().__init__()
        dw = int(c * DW_Expand)
        self.conv1 = nn.Conv2d(c, dw, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw, dw, 3, 1, 1, groups=dw, bias=True)
        self.conv3 = nn.Conv2d(dw, c, 1, 1, 0, bias=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp, k_cached=None, v_cached=None):
        return self.conv3(F.gelu(self.conv2(self.conv1(inp)))) * self.beta, None, None
