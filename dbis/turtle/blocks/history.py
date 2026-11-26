"""Turtle's history blocks (the truncated causal history model): FHR, SAB and CHM, with the K/V caches they carry.

Source: basicsr/models/archs/turtle_t1_arch.py of https://github.com/Ascend-Research/Turtle (commit 7094f42), MIT
License, Copyright (c) 2024 Huawei Technologies Co., Ltd. The same modules, weight names and maths. What differs, none of
it changing a value:
  * the local attention mask of SAB (L1 distance <= 4 on its token grid) is made once per grid size and device and kept,
    where the authors build it on the CPU with torch.cdist and copy it to the GPU in every call;
  * the einops rearranges are the reshape and permute they stand for.
fast=True (dbisf) changes the order of roundings only: the top-k and local masks are applied in one masked softmax.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import heads, unheads

_MASKS = {}


def local_mask(h, w, n, device):
    """(h w, h w) bool: token pairs at L1 distance <= n on an h x w grid (the authors' create_local_attention_mask; made
    on `device`, the distances are small integers and exact in float32)."""
    key = (h, w, n, str(device))
    if key not in _MASKS:
        yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).view(-1, 2).float()
        _MASKS[key] = torch.cdist(coords, coords, p=1) <= n
    return _MASKS[key]


def clipped_softmax(t, dim=-1):
    """Softmax over the nonzero entries only (zeros stay zero), then renormalized, as the authors."""
    zero = t == 0
    s = F.softmax(t.masked_fill(zero, float("-inf")), dim=dim).masked_fill(zero, 0)
    return s / s.sum(dim=dim, keepdim=True)


class FrameHistoryRouter(nn.Module):
    """FHR: channel attention whose keys and values are those of the current frame after those of the cached frames
    (concatenated along the channels). Keeps the last num_frames_tocache frames' keys and values as the new cache."""

    def __init__(self, dim, num_heads, bias, num_frames_tocache=1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
        self.num_frames_tocache = num_frames_tocache

    def forward(self, x, k_cached=None, v_cached=None):
        _, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q, k, v = heads(q, self.num_heads), heads(k, self.num_heads), heads(v, self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        if k_cached is not None and v_cached is not None:
            k = torch.cat([k_cached, k], dim=2)
            v = torch.cat([v_cached, v], dim=2)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out = self.project_out(unheads(attn @ v, h, w))
        keep = int(self.num_frames_tocache * c / self.num_heads)
        return out, k[:, :, -keep:, :], v[:, :, -keep:, :]


class StateAlignBlock(nn.Module):
    """SAB: aligns the cached frames to the current one. Queries and keys are pooled into a grid of tokens (strided
    window_size convs after 1x1 convs); every query token attends, in each cached frame separately, to its top 5 keys and
    to the keys within L1 distance 4 (both sets added, so a key in both counts twice), softmax over those only; the values
    are the frame's pixels gathered per token. One head. Returns the aligned frames (b, frames, c, h, w) and the new cache
    (the last num_frames_tocache frames' keys and values, the current one included)."""

    def __init__(self, dim, num_heads, bias, num_frames_tocache, Scale_patchsize=1, fast=False):
        super().__init__()
        self.num_heads = 1
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.qk = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1, groups=dim * 2, bias=bias)
        self.v = nn.Conv2d(dim, dim, 1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
        self.num_frames_tocache = num_frames_tocache
        self.window_size = ws = 2 * Scale_patchsize
        self.k2 = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.k2_dwconv = nn.Conv2d(dim * 2, dim * 2, ws, stride=ws, padding=1, groups=dim * 2, bias=bias)
        self.q2 = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.q2_dwconv = nn.Conv2d(dim * 2, dim * 2, ws, stride=ws, padding=1, groups=dim * 2, bias=bias)
        self.fast = fast

    def forward(self, x, k_cached=None, v_cached=None):
        b, c, h, w = x.shape
        ws, d = self.window_size, c                                  # one head: head_dim = c
        q, k = self.qk_dwconv(self.qk(x)).chunk(2, dim=1)
        v = self.v_dwconv(self.v(x))
        k = self.k2_dwconv(self.k2(k))
        q = self.q2_dwconv(self.q2(q))
        H, W = q.shape[2], q.shape[3]
        # 'b (h_head d) h w -> b 1 h_head (h w) d' (d = 2c)
        q = q.reshape(b, 1, 2 * d, H, W).permute(0, 1, 3, 4, 2).reshape(b, 1, 1, H * W, 2 * d)
        k = k.reshape(b, 1, 2 * d, H, W).permute(0, 1, 3, 4, 2).reshape(b, 1, 1, H * W, 2 * d)
        # 'b (h_head d) (p1 h) (p2 w) -> b 1 h_head (h w) (p1 p2 d)' (p1 = p2 = ws)
        hh, wh = h // ws, w // ws
        v = v.reshape(b, 1, d, ws, hh, ws, wh).permute(0, 1, 4, 6, 3, 5, 2).reshape(b, 1, 1, hh * wh, ws * ws * d)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        if k_cached is not None and v_cached is not None:
            k = torch.cat([k_cached, k], dim=1)
            v = torch.cat([v_cached, v], dim=1)
        frames = k.shape[1]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        mask = local_mask(H, W, 4, attn.device)
        if self.fast:
            # top-k and local entries added (each counts once or twice), softmax over them: the same function
            top = torch.zeros_like(attn, dtype=torch.bool).scatter_(-1, attn.topk(5, dim=-1).indices, True)
            attn = attn * (top.to(attn.dtype) + mask.to(attn.dtype))
            attn = clipped_softmax(attn)
        else:
            _, idx = torch.topk(attn, k=5, dim=-1)                    # the authors' zero_out_non_top_k(attn, 5)
            attn1 = attn * torch.zeros_like(attn).scatter_(dim=-1, index=idx, value=1)
            attn2 = mask[None, None, None] * attn
            attn = clipped_softmax(attn1 + attn2)
        out = torch.matmul(attn, v)
        # 'b frames h_head (h w) (p1 p2 d) -> (b frames) (h_head d) (p1 h) (p2 w)'
        out = out.reshape(b, frames, 1, hh, wh, ws, ws, d).permute(0, 1, 2, 7, 5, 3, 6, 4).reshape(b * frames, d, ws * hh,
                                                                                                    ws * wh)
        out = self.project_out(out)
        out = out.reshape(b, frames, *out.shape[1:])
        return out, k[:, -self.num_frames_tocache:], v[:, -self.num_frames_tocache:]


class CausalHistoryModel(nn.Module):
    """CHM: SAB aligns the cached frames to the current one; FHR then attends from the current frame over the aligned
    frames' keys and values (and its own). The cache is SAB's."""

    def __init__(self, dim, num_heads, bias, scale_patchsize, num_frames_tocache=1, fast=False):
        super().__init__()
        self.spatial_aligner = StateAlignBlock(dim, num_heads, bias, num_frames_tocache, Scale_patchsize=scale_patchsize,
                                               fast=fast)
        self.ChanAttn = FrameHistoryRouter(dim, num_heads, bias)
        self.kv = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1, groups=dim * 2, bias=bias)
        self.num_heads = num_heads

    def forward(self, x, k_cached=None, v_cached=None):
        xs, k_tocache, v_tocache = self.spatial_aligner(x, k_cached, v_cached)
        b, frames = xs.shape[:2]
        k, v = self.kv_dwconv(self.kv(xs.reshape(b * frames, *xs.shape[2:]))).chunk(2, dim=1)
        n = self.num_heads
        _, cc, h, w = k.shape
        # '(b frames) (head c) h w -> b head (frames c) (h w)'
        k = k.reshape(b, frames, n, cc // n, h, w).permute(0, 2, 1, 3, 4, 5).reshape(b, n, frames * (cc // n), h * w)
        v = v.reshape(b, frames, n, cc // n, h, w).permute(0, 2, 1, 3, 4, 5).reshape(b, n, frames * (cc // n), h * w)
        k = F.normalize(k, dim=-1)
        out, _, _ = self.ChanAttn(x, k, v)
        return out, k_tocache, v_tocache
