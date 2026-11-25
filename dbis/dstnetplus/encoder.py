"""The per-frame encoder of DSTNet+: the first conv, the Haar split and the high-frequency branch, before the recurrent
propagation. The full network (net.py, DSTNetPlus) subclasses it, so its modules keep their names (feat_extractor,
x_wave_2_conv1, x_wave_2_conv2) in the weights."""
import torch.nn as nn
import torch.nn.functional as F

from .blocks.haar import Haar


class DSTNetPlusEncoder(nn.Module):
    """forward(x (n, 3, h, w) in [0, 1]) -> [full size (n, num_feat, h, w), low band (n, num_feat, h/2, w/2),
    processed high bands (n, 3 num_feat, h/2, w/2)]: the features DSTNet+ takes from every frame (the low band goes
    through the propagation, the high bands straight to the inverse Haar). Odd h or w are reflect-padded to even, as the
    network does. feature_info lists channels and reduction of each (as timm's features_only models do)."""

    def __init__(self, num_feat=96, nonblind_denoise=False, **_):
        super().__init__()
        self.feat_extractor = nn.Conv2d(4 if nonblind_denoise else 3, num_feat, 3, 1, 1)
        self.wave = Haar(num_feat)
        self.x_wave_2_conv1 = nn.Conv2d(num_feat * 3, num_feat * 3, 1, 1, 0, groups=3)
        self.x_wave_2_conv2 = nn.Conv2d(num_feat * 3, num_feat * 3, 1, 1, 0, groups=3)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.feature_info = [{"num_chs": num_feat, "reduction": 1, "module": "feat_extractor"},
                             {"num_chs": num_feat, "reduction": 2, "module": "wave (low band)"},
                             {"num_chs": 3 * num_feat, "reduction": 2, "module": "x_wave_2_conv2 (high bands)"}]

    def encode(self, x):
        f = self.feat_extractor(x)
        low, high = self.wave(f)
        return [f, low.contiguous(), self.x_wave_2_conv2(self.lrelu(self.x_wave_2_conv1(high)))]

    def forward(self, x):
        h, w = x.shape[-2:]
        if h % 2 or w % 2:
            x = F.pad(x, [0, w % 2, 0, h % 2], mode="reflect")
        return self.encode(x)
