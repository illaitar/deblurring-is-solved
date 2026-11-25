"""The dynamic depthwise convolution of DSTNet+ (IDynamicDWConv of basicsr/archs/prog_dynconv.py): a small multi-scale
network predicts a 3x3 kernel for every pixel and channel, which is then applied to the input. The authors apply it
with a CuPy kernel (float32 and float64 only); here it is nine multiply-adds of PyTorch, in the kernel's order."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """blocks.ResBlock of the authors with inplanes == planes and stride 1: conv, LeakyReLU(0.1), conv, + input."""

    def __init__(self, c, k=3, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, k, 1, (k - 1) // 2, groups=groups)
        self.conv2 = nn.Conv2d(c, c, k, 1, (k - 1) // 2, groups=groups)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))
        out += x
        return out


def dynamic_dwconv(x, weight, exact=True):
    """out[n, c, i, j] = sum over (a, b) of weight[n, g(c), a, b, i, j] * x[n, c, i + a - p, j + b - p], zero outside the
    map: the authors' idynamic_forward_kernel, the products added in its order (a, then b).
    exact=True (float32 input): every step `value += weight * data` of the kernel is a fused multiply-add on the GPU,
    one rounding to float32; here the product is formed in float64 (exact for two float32 numbers), the sum rounded to
    float64 and then to float32, which gives the same float32 except where the float64 rounding lands exactly on a
    float32 tie (about 1 step in 5e8). exact=False: torch.addcmul in the input's dtype (two roundings per step in
    float32, 1 unit in the last place per call; the dbisf path, also under float16 autocast)."""
    n, c, h, w = x.shape
    g, k = weight.shape[1], weight.shape[2]
    p = (k - 1) // 2
    xp = F.pad(x, (p, p, p, p))
    rep = c // g
    exact = exact and x.dtype == torch.float32 and weight.dtype == torch.float32
    out = None
    for a in range(k):
        for b in range(k):
            wk = weight[:, :, a, b]
            if rep > 1:
                wk = wk.repeat_interleave(rep, 1)
            xk = xp[:, :, a:a + h, b:b + w]
            if out is None:
                out = wk * xk
            elif exact:
                out = torch.addcmul(out.double(), wk.double(), xk.double()).float()
            else:
                out = torch.addcmul(out, wk, xk)
    return out


class IDynamicDWConv(nn.Module):
    """Three scales (max pooling to 1/2, 1/4, 1/8, n_blocks ResBlocks each), upsampled back, a 1x1 conv to k * k weights
    per group and pixel, then the dynamic depthwise conv of the input. exact: see dynamic_dwconv."""
    exact = True

    def __init__(self, channels, kernel_size=3, group_channels=1, n_blocks=5, conv_group=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.groups = channels // group_channels
        for name in ("Block1", "Block2", "Block3"):
            setattr(self, name, nn.Sequential(*[ResBlock(channels, kernel_size, conv_group) for _ in range(n_blocks)]))
        self.tokernel = nn.Conv2d(channels * 3, kernel_size ** 2 * self.groups, 1, 1, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = self.Block1(F.adaptive_max_pool2d(x, output_size=(h // 2, w // 2)))
        x2 = self.Block2(F.adaptive_max_pool2d(x1, output_size=(h // 4, w // 4)))
        x3 = self.Block3(F.adaptive_max_pool2d(x2, output_size=(h // 8, w // 8)))
        x1 = F.interpolate(x1, size=(h, w), mode="bilinear")
        x2 = F.interpolate(x2, size=(h, w), mode="bilinear")
        x3 = F.interpolate(x3, size=(h, w), mode="bilinear")
        weight = self.tokernel(torch.cat([x1, x2, x3], dim=1))
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)
        return dynamic_dwconv(x, weight, self.exact)
