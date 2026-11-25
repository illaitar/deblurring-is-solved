"""NAFNet (Chen et al., "Simple Baselines for Image Restoration", ECCV 2022) for inference: dbis-NAFNet and dbisf-NAFNet.

Architecture and weights by the authors: https://github.com/megvii-research/NAFNet, MIT License,
Copyright (c) 2022 megvii-model. This file is a re-implementation in plain PyTorch.

dbis-NAFNet (tlc="exact") gives the outputs of the authors' NAFNetLocal (basicsr/models/archs/NAFNet_arch.py and
local_arch.py, the configuration of their GoPro results) with the weights written by `python -m dbis.convert`, which
folds every affine step into the convolution after it:
  * the scale and shift of the two LayerNorms of a block go into conv1 and conv4,
  * the residual scales beta and gamma go into conv3 and conv5.
TLC (test-time local statistics, Chu et al., ECCV 2022): the channel attention averages over a fixed window, 1.5 times
the training crop at that scale, instead of over the whole image. The window sizes follow from the training crop, so
the dry run of the authors' code at construction is not needed.

dbisf-NAFNet (tlc="fast") averages that window on a map pooled into r x r cells, runs the attention's 1x1 conv there and
upsamples bilinearly. Both steps are linear and the upsampling weights sum to 1, so moving the conv before the
upsampling changes nothing; the cell pooling is the only approximation (see DBIS.md for its cost in dB).
tlc=None is the plain NAFNet (global channel attention).
"""
import torch.nn as nn
import torch.nn.functional as F

from .restore import Restorer


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


class NAFNet(Restorer, nn.Module):
    """U-Net of NAFBlocks. Defaults: NAFNet-64 of the GoPro release (enc 1, 1, 1, 28; middle 1; dec 1, 1, 1, 1).
    tlc: None (plain NAFNet), "exact" (dbis, the authors' test configuration) or "fast" (dbisf).
    restore(frames, ...) is the inference call (dbis/restore.py); forward(x) takes [B, 3, H, W] in [0, 1], any H, W
    (zero-padded to a multiple of 16 inside, as the authors do). Defaults of restore: the whole frame, one frame per
    call, float32 (as the quality of dbis and dbisf was measured)."""
    kind = "image"

    def __init__(self, width=64, enc=(1, 1, 1, 28), middle=1, dec=(1, 1, 1, 1), tlc=None, train_crop=256):
        super().__init__()

        def k(scale):                      # TLC window at 1 / 2**scale: 1.5 x the training crop there
            return None if tlc is None else (train_crop >> scale) * (3 * train_crop // 2) // train_crop

        fast = tlc == "fast"
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.encoders, self.downs, self.ups, self.decoders = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = width
        for s, n in enumerate(enc):
            self.encoders.append(nn.Sequential(*[Block(c, k(s), fast) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, 2))
            c *= 2
        self.middle_blks = nn.Sequential(*[Block(c, k(len(enc)), fast) for _ in range(middle)])
        for i, n in enumerate(dec):
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[Block(c, k(len(enc) - 1 - i), fast) for _ in range(n)]))
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.mult = 2 ** len(enc)

    def forward(self, inp):
        h, w = inp.shape[-2:]
        inp = F.pad(inp, (0, -w % self.mult, 0, -h % self.mult))
        x = self.intro(inp)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = dec(up(x) + skip)
        return (self.ending(x) + inp)[..., :h, :w]


def fold(sd):
    """The authors' state dict (key "params" of NAFNet-GoPro-width64.pth) -> the state dict of NAFNet above.
    Computed in float64, stored as float32."""
    sd = {k: v.double() for k, v in sd.items()}
    out, used = {}, set()
    for p in sorted({k[:-len("norm1.weight")] for k in sd if k.endswith("norm1.weight")}):
        for nm, conv in (("norm1", "conv1"), ("norm2", "conv4")):
            g, b = sd[f"{p}{nm}.weight"], sd[f"{p}{nm}.bias"]
            w, c = sd[f"{p}{conv}.weight"], sd[f"{p}{conv}.bias"]
            out[f"{p}{conv}.weight"] = w * g.view(1, -1, 1, 1)
            out[f"{p}{conv}.bias"] = c + w[:, :, 0, 0] @ b
            used |= {f"{p}{nm}.weight", f"{p}{nm}.bias", f"{p}{conv}.weight", f"{p}{conv}.bias"}
        for sc, conv in (("beta", "conv3"), ("gamma", "conv5")):
            s = sd[f"{p}{sc}"].reshape(-1)
            out[f"{p}{conv}.weight"] = sd[f"{p}{conv}.weight"] * s.view(-1, 1, 1, 1)
            out[f"{p}{conv}.bias"] = sd[f"{p}{conv}.bias"] * s
            used |= {f"{p}{sc}", f"{p}{conv}.weight", f"{p}{conv}.bias"}
        for t in ("weight", "bias"):
            out[f"{p}sca.{t}"] = sd[f"{p}sca.1.{t}"]
            used.add(f"{p}sca.1.{t}")
    for k, v in sd.items():
        if k not in used:
            out[k] = v
    return {k: v.float().contiguous() for k, v in out.items()}
