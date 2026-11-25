"""CNN blocks of BSST from Shift-Net (Li et al., CVPR 2023, https://github.com/dasongli1/Shift-Net; basicsr/archs/gshift_arch.py
of BSSTNet): CAB blocks, channel attention, grouped spatial-temporal shift blocks, upsampling."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import lrelu


class LayerNorm2d(nn.Module):
    """LayerNorm over the channels of NCHW, the authors' LayerNormFunction forward (same ops, so the same autocast
    dtypes: pow and mean run in float32)."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        c = x.size(1)
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, c, 1, 1) * y + self.bias.view(1, c, 1, 1)


def conv(cin, cout, k, bias=False, stride=1):
    return nn.Conv2d(cin, cout, k, padding=k // 2, bias=bias, stride=stride)


class CALayer(nn.Module):
    """Channel attention (CALayer and CALayer2 of the authors, reduction 1)."""

    def __init__(self, c):
        super().__init__()
        self.conv_du = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.ReLU(inplace=True),
                                     nn.Conv2d(c, c, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.conv_du(F.adaptive_avg_pool2d(x, 1))


class CAB(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.body = nn.Sequential(conv(c, c, k), lrelu(), conv(c, c, k))
        self.CA = CALayer(c)

    def forward(self, x):
        res = self.CA(self.body(x))
        res += x
        return res


class Gate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SigmoidGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * torch.sigmoid(x2)


class RepConv(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        self.conv_1 = nn.Conv2d(c, c, k, bias=False, padding=k // 2, groups=c)
        self.conv_2 = nn.Conv2d(c, c, 3, bias=False, padding=1, groups=c)

    def forward(self, x):
        return self.conv_1(x) + self.conv_2(x) + x


class RepConv2(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv_2 = nn.Conv2d(c, c, 3, bias=False, padding=1, groups=c)

    def forward(self, x):
        return self.conv_2(x) + x


def cab_body(cin, c, k):
    return nn.Sequential(conv(cin, 2 * c, 1), RepConv2(2 * c), Gate(), RepConv(c, k), conv(c, 2 * c, 1), SigmoidGate(),
                         CALayer(c), conv(c, c, 1))


class CAB1(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        self.norm = LayerNorm2d(c)
        self.body = cab_body(c, c, k)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        return x + self.body(self.norm(x)) * self.beta


class CAB2(nn.Module):
    """CAB1 on [x[:, :c], depthwise 3x3 of x[:, c:]], residual on the first c channels."""

    def __init__(self, c, k, add):
        super().__init__()
        self.n_feat = c
        self.conv1 = nn.Conv2d(add, add, 3, bias=False, padding=1, groups=add)
        self.norm = LayerNorm2d(add + c)
        self.body = cab_body(c + add, c, k)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        shortcut, hw = x[:, 0:self.n_feat], x[:, self.n_feat:]
        res = self.body(self.norm(torch.cat((shortcut, self.conv1(hw)), dim=1)))
        return shortcut + res * self.beta


# spatial shifts (dy, dx) of spatial_shift2: out[y, x] = in[y - dy, x - dx], zero outside; 16 groups of n2 channels,
# then 8 groups of n1 channels
SHIFTS_A = ((8, 8), (8, 4), (8, 0), (8, -4), (8, -8), (-8, 8), (-8, 4), (-8, 0), (-8, -4), (-8, -8),
            (4, 8), (4, -8), (0, 8), (0, -8), (-4, 8), (-4, -8))
SHIFTS_B = ((4, 4), (4, 0), (4, -4), (0, 4), (0, -4), (-4, 4), (-4, 0), (-4, -4))


def spatial_shift(hw, number):
    """spatial_shift2 of the authors (24 slice copies into zeros) as one pad and one cat: the same values."""
    n2 = (number - 1) // 2
    n1 = number - 2 * n2
    h, w = hw.shape[-2:]
    p = F.pad(hw, (8, 8, 8, 8))
    parts, c = [], 0
    for (dy, dx), k in [(s, n2) for s in SHIFTS_A] + [(s, n1) for s in SHIFTS_B]:
        parts.append(p[:, c:c + k, 8 - dy:8 - dy + h, 8 - dx:8 - dx + w])
        c += k
    return torch.cat(parts, dim=1)


class ShiftBlock(nn.Module):
    """Encoder_shift_block: four (CAB2, CAB1) pairs, each after a temporal channel roll and a grouped spatial shift."""

    def __init__(self, c):
        super().__init__()
        self.number = c // 2 // 8
        for name in ("encoder_level1", "encoder_level1_1", "encoder_level1_2", "encoder_level1_3"):
            setattr(self, name, nn.Sequential(CAB2(c, 5, 8 * self.number), CAB1(c, 5)))

    def channel_shift(self, x, n, reverse=False):
        """The authors roll x.view(1, B*C, H, W) by C/2 along dim 1, B = frames of one patch: half of the channels of
        every frame come from the previous (next) frame, with wrap-around. Here per patch: x.view(n, T*C, H, W)."""
        bt, c, h, w = x.shape
        s = -(c // 2) if reverse else c // 2
        y = torch.roll(x.view(n, bt // n * c, h, w), s, 1).view(bt, c, h, w)
        hw = y[:, -8 * self.number:, ...] if reverse else y[:, 0:8 * self.number, ...]
        return torch.cat((y, spatial_shift(hw, self.number)), dim=1)

    def forward(self, x, n):
        x = self.encoder_level1(self.channel_shift(x, n))
        x = self.encoder_level1_1(self.channel_shift(x, n, reverse=True))
        x = self.encoder_level1_2(self.channel_shift(x, n))
        return self.encoder_level1_3(self.channel_shift(x, n, reverse=True))


class PixelShufflePack(nn.Module):
    def __init__(self, cin, cout, scale, k):
        super().__init__()
        self.scale = scale
        self.upsample_conv = nn.Conv2d(cin, cout * scale * scale, k, padding=(k - 1) // 2)

    def forward(self, x):
        return F.pixel_shuffle(self.upsample_conv(x), self.scale)


class SkipUpSample(nn.Module):
    def __init__(self, c, s):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                nn.Conv2d(c + s, c, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        return self.up(x) + y
