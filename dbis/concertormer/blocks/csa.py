"""The blocks of Concertormer (basicsr/models/archs/Concertormer_arch.py and arch_util.py of the authors): LayerNorm2d,
concerto self-attention (CSAttention) in its two modes, and the CSA block.

The attention splits the channels of each head into four quarters, each attending in its own way inside 8 x 8 windows:
b (within a window, over its 64 positions, with a shifted-window mask on odd blocks), l (within a window, over the
quarter's channels), g (over all windows, channels by channels per position) and r (over all windows, positions by
positions per channel); b and l pass through a 3 x 3 conv over the window grid ("talking heads"), g and r through a
learnt mixing of the heads. The einops rearranges of the authors are written as the reshapes and permutes they stand for.

grid: None, the attention over the whole map (the authors' Concertormer); (kh, kw), the authors' ConcertormerLocal:
the projections run on the whole map, the attention (with its channel gate) on overlapping windows of kh x kw that
cover the map, averaged where they overlap.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

KS = 8                     # the window of the attention


def layer_norm(x, weight, bias, eps):
    """The authors' LayerNormFunction.forward (over the channels of an NCHW map), as plain ops."""
    c = x.shape[1]
    mu = x.mean(1, keepdim=True)
    var = (x - mu).pow(2).mean(1, keepdim=True)
    y = (x - mu) / (var + eps).sqrt()
    return weight.view(1, c, 1, 1) * y + bias.view(1, c, 1, 1)


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return layer_norm(x, self.weight, self.bias, self.eps)


def to_windows(t, heads):
    """'b (head c) (h kh) (w kw) -> b head (h w) (kh kw) c'."""
    b, ch, hh, ww = t.shape
    c, h, w = ch // heads, hh // KS, ww // KS
    return t.reshape(b, heads, c, h, KS, w, KS).permute(0, 1, 3, 5, 4, 6, 2).reshape(b, heads, h * w, KS * KS, c)


def talk_dw(conv, a):
    """conv (the talking heads: Conv3d (3, 3, 1), "same", no bias) of a (b, nh, hs, ws, e), as depthwise conv2d on the
    channels-last view of each head (one per pair of heads, summed). dbisf: 8.6x faster than the Conv3d at one head and
    the same values; 3.8x at two heads (float16 roundings of the sum over heads differ); slower from four heads on."""
    nh, e = a.shape[1], a.shape[-1]
    w = conv.weight[..., 0]
    outs = []
    for t in range(nh):
        acc = None
        for h in range(nh):
            y = F.conv2d(a[:, h].permute(0, 3, 1, 2), w[t, h].expand(e, 1, 3, 3), padding=1, groups=e)
            acc = y if acc is None else acc + y
        outs.append(acc.permute(0, 2, 3, 1))
    return outs[0][:, None] if nh == 1 else torch.stack(outs, 1)


def talk_grouped(conv, a):
    """The same as conv for four or more heads: one grouped conv2d over channels-last (b, e * nh, hs, ws), a group per
    entry e mixing its heads. dbisf: 2.3x faster than the Conv3d at four heads and e = 4096 (float16 sums differ)."""
    b, nh, hs, ws, e = a.shape
    x = a.permute(0, 4, 1, 2, 3).reshape(b, e * nh, hs, ws).contiguous(memory_format=torch.channels_last)
    y = F.conv2d(x, conv.weight[..., 0].repeat(e, 1, 1, 1), padding=1, groups=e)
    return y.reshape(b, e, nh, hs, ws).permute(0, 2, 3, 4, 1)


def talk(conv, a, fast):
    """The talking heads of a (b, nh, hs, ws, e): the authors' Conv3d, or with fast (dbisf) the fastest form measured
    on the 1280x720 shapes: depthwise for one or two heads, grouped for four or more heads with e >= 1024."""
    nh, e = a.shape[1], a.shape[-1]
    if fast and nh <= 2:
        return talk_dw(conv, a)
    if fast and e >= 1024:
        return talk_grouped(conv, a)
    return conv(a)


_MASKS = {}


def shift_mask(h, w, device):
    """The authors' calculate_mask (the shifted-window mask, 0 or -100, float32), made once per size and device."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:           # "cuda" and "cuda:0" are one cache entry
        device = torch.device("cuda", torch.cuda.current_device())
    key = (h, w, str(device))
    if key not in _MASKS:
        s = KS // 2
        img = torch.zeros((1, h, w, 1))
        cnt = 0
        for hs in (slice(0, -KS), slice(-KS, -s), slice(-s, None)):
            for ws in (slice(0, -KS), slice(-KS, -s), slice(-s, None)):
                img[:, hs, ws, :] = cnt
                cnt += 1
        # window_partition: 'b (h kh) (w kw) c -> (b h w c) (kh kw)'
        win = img.reshape(1, h // KS, KS, w // KS, KS, 1).permute(0, 1, 3, 5, 2, 4).reshape(-1, KS * KS)
        m = win.unsqueeze(1) - win.unsqueeze(2)
        _MASKS[key] = m.masked_fill(m != 0, float(-100.0)).masked_fill(m == 0, float(0.0)).to(device)
    return _MASKS[key]


class CSAttention(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias=False, shift=0, grid=None, fast=False, level=0):
        super().__init__()
        self.level = level                           # the map is the padded input / 2**level (masks, prepare())
        h_dim = int(ffn_expansion_factor * dim)
        self.dim, self.h_dim, self.shift, self.num_heads, self.grid = dim, h_dim, shift, num_heads, grid
        self.fast = fast                             # dbisf: the talking heads as depthwise / grouped convs (talk)
        self.temperature = nn.Parameter(torch.ones(4, num_heads, 1, 1, 1))
        self.r_talking = nn.Parameter(torch.empty(num_heads, h_dim // num_heads // 4, h_dim // num_heads // 4, num_heads))
        self.g_talking = nn.Parameter(torch.empty(num_heads, KS ** 2, KS ** 2, num_heads))
        nn.init.kaiming_normal_(self.r_talking, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.g_talking, a=math.sqrt(5))
        self.b_talking = nn.Conv3d(num_heads, num_heads, (3, 3, 1), padding="same", bias=bias)
        self.l_talking = nn.Conv3d(num_heads, num_heads, (3, 3, 1), padding="same", bias=bias)
        self.shift_size = KS // 2
        self.to_quz = nn.Conv2d(dim, h_dim * 3, kernel_size=1, bias=bias)
        self.to_kv = nn.Conv2d(dim, h_dim * 2, kernel_size=1, bias=bias)
        self.dw_u = nn.Conv2d(h_dim, h_dim, 3, padding=1, groups=h_dim, bias=bias)
        self.project = nn.Conv2d(h_dim, h_dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(h_dim, dim, kernel_size=1, bias=bias)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(h_dim, h_dim, kernel_size=1, bias=True))

    def attend(self, q, k, v):
        """The four attentions on windows of 8 x 8 -> (b, h_dim, h, w), before the channel gate."""
        b, _, h, w = q.shape
        hs, nh = h // KS, self.num_heads
        q, k, v = (to_windows(t, nh) for t in (q, k, v))
        qb, ql, qg, qr = q.chunk(4, dim=-1)
        kb, kl, kg, kr = k.chunk(4, dim=-1)
        vb, vl, vg, vr = v.chunk(4, dim=-1)
        qb, kb = (F.normalize(t, dim=-1) for t in (qb, kb))
        ql, kl = (F.normalize(t, dim=-2) for t in (ql, kl))
        qr, qg, kr, kg = (F.normalize(t, dim=-3) for t in (qr, qg, kr, kg))
        attn_b = torch.matmul(qb, kb.permute(0, 1, 2, 4, 3)) * self.temperature[0]                   # b head hw k k
        attn_l = torch.matmul(ql.permute(0, 1, 2, 4, 3), kl) * self.temperature[1]                   # b head hw c c
        attn_g = torch.matmul(qg.permute(0, 1, 3, 4, 2), kg.permute(0, 1, 3, 2, 4)) * self.temperature[2]   # b head k c c
        attn_r = torch.matmul(qr.permute(0, 1, 4, 3, 2), kr.permute(0, 1, 4, 2, 3)) * self.temperature[3]   # b head c k k
        n1, n2 = attn_b.shape[-1], attn_l.shape[-1]
        # 'b head (h w) k1 k2 -> b head h w (k1 k2)', the conv, and back
        attn_b = talk(self.b_talking, attn_b.reshape(b, nh, hs, -1, n1 * n1), self.fast).reshape(b, nh, -1, n1, n1)
        attn_l = talk(self.l_talking, attn_l.reshape(b, nh, hs, -1, n2 * n2), self.fast).reshape(b, nh, -1, n2, n2)
        attn_g = torch.einsum("hklt, bhk... -> btl...", self.g_talking, attn_g)
        attn_r = torch.einsum("hcdt, bhc... -> btd...", self.r_talking, attn_r)
        if self.shift:
            attn_b = attn_b + shift_mask(h, w, q.device)[None, None, ...]
        attn_b, attn_l, attn_g, attn_r = (torch.softmax(t, dim=-1) for t in (attn_b, attn_l, attn_g, attn_r))
        ws = w // KS

        def back(t, perm):          # -> b (head c) (h kh) (w kw)
            return t.permute(*perm).reshape(b, -1, hs * KS, ws * KS)
        # 'b head (h w) (kh kw) c -> b (head c) (h kh) (w kw)' and the like
        out_b = back(torch.matmul(attn_b, vb).reshape(b, nh, hs, ws, KS, KS, -1), (0, 1, 6, 2, 4, 3, 5))
        out_l = back(torch.matmul(attn_l, vl.transpose(-1, -2)).reshape(b, nh, hs, ws, -1, KS, KS), (0, 1, 4, 2, 5, 3, 6))
        out_g = back(torch.matmul(attn_g, vg.permute(0, 1, 3, 4, 2)).reshape(b, nh, KS, KS, -1, hs, ws),
                     (0, 1, 4, 5, 2, 6, 3))
        out_r = back(torch.matmul(attn_r, vr.transpose(-1, -3)).reshape(b, nh, -1, KS, KS, hs, ws), (0, 1, 2, 5, 3, 6, 4))
        return torch.cat([out_b, out_l, out_g, out_r], dim=1)

    def forward(self, x, y=None):
        if self.grid is None:          # Concertormer: the attention over the whole map
            k, v = self.to_kv(y if y is not None else x).chunk(2, dim=1)
            q, u, z = self.to_quz(x).chunk(3, dim=1)
            if self.shift:
                q, k, v = (torch.roll(t, shifts=(-self.shift_size, -self.shift_size), dims=(-1, -2)) for t in (q, k, v))
            out = self.attend(q, k, v)
            if self.shift:
                out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(-1, -2))
            out = self.project(out * self.sca(out)) + self.dw_u(u)
            return self.project_out(out * z)
        # ConcertormerLocal: the attention and its gate on overlapping windows of the map
        quz = self.to_quz(x)
        kv = self.to_kv(x) if y is None else self.to_kv(y)
        q, u, z = quz.chunk(3, dim=1)
        boxes, size = grid_boxes(q.shape[-2:], self.grid)
        q, kv = (torch.cat([t[:, :, i:i + size[0], j:j + size[1]] for i, j in boxes], dim=0) for t in (q, kv))
        if self.shift:
            q, kv = (torch.roll(t, shifts=(-self.shift_size, -self.shift_size), dims=(-1, -2)) for t in (q, kv))
        k, v = kv.chunk(2, dim=1)
        out = self.attend(q, k, v)
        if self.shift:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(-1, -2))
        out = out * self.sca(out)
        b, _, h, w = x.shape
        preds = torch.zeros((b, self.h_dim, h, w), device=out.device)       # float32, as the authors' grids_inverse
        count = torch.zeros((b, 1, h, w), device=out.device)
        for n, (i, j) in enumerate(boxes):
            preds[0, :, i:i + size[0], j:j + size[1]] += out[n, :, :, :]
            count[0, 0, i:i + size[0], j:j + size[1]] += 1.
        out = preds / count
        out = self.project(out) + self.dw_u(u)
        return self.project_out(z * out)


def grid_boxes(hw, kernel):
    """The authors' LocalCSAttention.grids: top-left corners of windows of kernel (clipped to the map) that cover it,
    evenly spaced, the last ones flush with the far edges; and the window size."""
    h, w = hw
    k1, k2 = min(h, kernel[0]), min(w, kernel[1])
    num_row, num_col = (h - 1) // k1 + 1, (w - 1) // k2 + 1
    step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
    step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)
    boxes, i, last_i = [], 0, False
    while i < h and not last_i:
        j = 0
        if i + k1 >= h:
            i, last_i = h - k1, True
        last_j = False
        while j < w and not last_j:
            if j + k2 >= w:
                j, last_j = w - k2, True
            boxes.append((i, j))
            j += step_j
        i += step_i
    return boxes, (k1, k2)


class CSABlock(nn.Module):
    """x + attention(norm(x)); a cross block takes the two halves of its input as (x, y): x + attention(norm1(x),
    norm2(y)), the keys and values from y."""

    def __init__(self, dim, head, ffn_expansion_factor, idx, cross=False, grid=None, fast=False, level=0):
        super().__init__()
        self.cross = bool(cross)
        if self.cross:
            self.norm1, self.norm2 = LayerNorm2d(dim), LayerNorm2d(dim)
        else:
            self.norm = LayerNorm2d(dim)
        self.m = CSAttention(dim, head, ffn_expansion_factor, bias=False, shift=idx % 2, grid=grid, fast=fast, level=level)

    def forward(self, x):
        if self.cross:
            x, y = x.chunk(2, dim=1)
            return x + self.m(self.norm1(x), self.norm2(y))
        return x + self.m(self.norm(x))
