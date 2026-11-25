"""Swin reconstruction of BSST (BSST_arch.py; VRT/RVRT lineage, "Facebook, BSD license" header there)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _const


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
    sdpa = False      # True: F.scaled_dot_product_attention with bias (+ shift mask) as a float16 attn_mask

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
        index = relative_position_index(self.window_size, x.device)
        bias = self.relative_position_bias_table[index[:n, :n].reshape(-1)].reshape(n, n, -1)
        if self.sdpa:     # windows of a sample are consecutive in b_: fold them into the head dimension
            nw = mask.shape[0] if mask is not None else 1
            m = bias.permute(2, 0, 1).unsqueeze(0)                                      # (1, heads, n, n)
            if mask is not None:
                m = m + mask[:, :n, :n].unsqueeze(1)                                    # (nw, heads, n, n)
            m = m.reshape(1, nw * self.num_heads, n, n).to(q.dtype)
            d = c // self.num_heads
            y = F.scaled_dot_product_attention(q.reshape(b_ // nw, nw * self.num_heads, n, d),
                                               k.reshape(b_ // nw, nw * self.num_heads, n, d),
                                               v.reshape(b_ // nw, nw * self.num_heads, n, d), attn_mask=m,
                                               scale=self.scale)
            return self.proj(y.reshape(b_, self.num_heads, n, d).transpose(1, 2).reshape(b_, n, c))
        attn = (q * self.scale) @ k.transpose(-2, -1)
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
