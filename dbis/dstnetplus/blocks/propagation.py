"""The recurrent propagation of DSTNet+ (BackwardProp and ForwardProp of basicsr/archs/dstnetplus_deblur_arch.py, with
ResidualBlockNoBN of arch_util.py): over the frames of a clip, the propagated features are fused with the current
frame's (PAGF), filtered by the dynamic depthwise conv and refined by residual blocks."""
import torch
import torch.nn as nn

from .dynconv import IDynamicDWConv


class ResidualBlockNoBN(nn.Module):
    """conv, LeakyReLU(0.1), conv, + input."""

    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return x + self.conv2(self.lrelu(self.conv1(x)))


class ResidualBlocks2D(nn.Module):
    """make_layer of the authors inside a Sequential (weights main.0.<i>.conv1 ...)."""

    def __init__(self, c, n):
        super().__init__()
        self.main = nn.Sequential(nn.Sequential(*[ResidualBlockNoBN(c) for _ in range(n)]))

    def forward(self, x):
        return self.main(x)


class PAGF(nn.Module):
    """Pixel attention guided fusion: a sigmoid of q * k mixes the current and the propagated features."""

    def __init__(self, c):
        super().__init__()
        self.conv0 = nn.Conv2d(c * 2, c * 2, 1, 1, 0)
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x_feat, feat_prop):
        q, k = self.conv0(torch.cat([x_feat, feat_prop], dim=1)).chunk(2, dim=1)
        sim = torch.sigmoid(q * k)
        return self.conv1(x_feat * (1 - sim) + feat_prop * sim)


class Propagation(nn.Module):
    """One direction (backward: from the last frame to the first, the outputs returned in frame order).
    forward(feature (b, t, c, h, w)) -> (b, t, c, h, w). channels_last: every step's features in channels_last (set
    with the weights' layout by DSTNetPlus.prop_channels_last)."""
    channels_last = False

    def __init__(self, c, num_kernel_block, num_block, backward):
        super().__init__()
        self.num_feat, self.backward = c, backward
        self.fusion = PAGF(c)
        self.kernel_conv_pixel = IDynamicDWConv(channels=c, n_blocks=num_kernel_block, kernel_size=3, group_channels=1,
                                                conv_group=1)
        self.resblock_bcakward2d = ResidualBlocks2D(c, num_block)          # (the authors' name)

    def forward(self, feature):
        b, t, c, h, w = feature.size()
        outs = []
        feat_prop = feature.new_zeros(b, c, h, w)
        mf = torch.channels_last if self.channels_last else None
        if mf:
            feat_prop = feat_prop.contiguous(memory_format=mf)
        for i in (range(t - 1, -1, -1) if self.backward else range(t)):
            x = feature[:, i] if mf is None else feature[:, i].contiguous(memory_format=mf)
            feat_prop = self.fusion(x, feat_prop)
            feat_prop = self.kernel_conv_pixel(feat_prop)
            feat_prop = self.resblock_bcakward2d(feat_prop)
            outs.append(feat_prop)
        return torch.stack(outs[::-1] if self.backward else outs, dim=1)
