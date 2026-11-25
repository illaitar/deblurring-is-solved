"""The encoder of NAFNet: intro conv, the four encoder stages and the middle blocks. The full network (net.py, NAFNet)
subclasses it, so its modules keep their names (intro, encoders, downs, middle_blks) in the weights."""
import torch.nn as nn
import torch.nn.functional as F

from .blocks.nafblock import Block


def tlc_window(tlc, train_crop, scale):
    """TLC window at 1 / 2**scale: 1.5 x the training crop there (None without TLC)."""
    return None if tlc is None else (train_crop >> scale) * (3 * train_crop // 2) // train_crop


class NAFNetEncoder(nn.Module):
    """forward(x [B, 3, H, W] in [0, 1]) -> [1/1 (width ch), 1/2, 1/4, 1/8 (8 width), 1/16 (16 width, after the middle
    blocks)]: the skip features of the U-Net and its bottleneck, for the input zero-padded to a multiple of 16 as the
    network pads it. feature_info lists channels and reduction of each (as timm's features_only models do)."""

    def __init__(self, width=64, enc=(1, 1, 1, 28), middle=1, tlc=None, train_crop=256, **_):
        super().__init__()
        fast = tlc == "fast"
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        c = width
        info = []
        for s, n in enumerate(enc):
            self.encoders.append(nn.Sequential(*[Block(c, tlc_window(tlc, train_crop, s), fast) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, 2))
            info.append({"num_chs": c, "reduction": 2 ** s, "module": f"encoders.{s}"})
            c *= 2
        self.middle_blks = nn.Sequential(*[Block(c, tlc_window(tlc, train_crop, len(enc)), fast) for _ in range(middle)])
        info.append({"num_chs": c, "reduction": 2 ** len(enc), "module": "middle_blks"})
        self.feature_info = info
        self.mult = 2 ** len(enc)

    def encode(self, inp):
        """inp already padded to a multiple of self.mult."""
        x = self.intro(inp)
        feats = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            feats.append(x)
            x = down(x)
        return feats + [self.middle_blks(x)]

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.encode(F.pad(x, (0, -w % self.mult, 0, -h % self.mult)))
