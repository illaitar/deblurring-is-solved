"""The encoder of Concertormer: the intro conv, the three encoder stages (each fed again with the input image downscaled
to its size through its own conv) and the middle blocks. The full network (net.py) subclasses it, so its modules keep
their names (intro, intros, encoders, downs, middle_blks) in the weights."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.csa import CSABlock


def grid_of(tlc, train_crop, grid_factor, level):
    """The attention windows of ConcertormerLocal at 1 / 2**level: the feature size of a training crop there times
    grid_factor (the authors' Local_Base.convert, without its dry run); None without TLC."""
    if tlc is None:
        return None
    s = train_crop >> level
    k = s * int(train_crop * grid_factor) // train_crop
    return (k, k)


class ConcertormerEncoder(nn.Module):
    """forward(x [B, 3, H, W] in [0, 1]) -> [width ch at 1, 2 width at 1/2, 4 width at 1/4, 8 width at 1/8 (after the
    middle blocks)], for the input reflect-padded to a multiple of 64 as the network pads it. The defaults are the
    authors' GoPro Concertormer (options/test/GoPro/Concertormer.yml); Concertormer-lite has enc (4, 4, 12), dec
    (12, 4, 4)."""

    def __init__(self, img_channel=3, width=48, middle_blk_num=2, ffn_expansion_factor=2, enc_blk_nums=(6, 8, 24),
                 enc_heads=(1, 2, 4), middle_heads=8, tlc=None, train_crop=256, grid_factor=1.75, fast=False, **_):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        n = len(enc_blk_nums)
        self.intros = nn.ModuleList(nn.Conv2d(img_channel, width * 2 ** (i + 1), 3, padding=1, bias=True) for i in range(n))
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        chan, info = width, []
        for lvl, (num, head) in enumerate(zip(enc_blk_nums, enc_heads)):
            g = grid_of(tlc, train_crop, grid_factor, lvl)
            self.encoders.append(nn.Sequential(*[CSABlock(chan, head, ffn_expansion_factor, i, cross=(i == 0 and lvl > 0),
                                                          grid=g, fast=fast, level=lvl) for i in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            info.append({"num_chs": chan, "reduction": 2 ** lvl, "module": f"encoders.{lvl}"})
            chan *= 2
        g = grid_of(tlc, train_crop, grid_factor, n)
        self.middle_blks = (nn.Sequential(*[CSABlock(chan, middle_heads, ffn_expansion_factor, i, cross=(i == 0), grid=g,
                                                     fast=fast, level=n) for i in range(middle_blk_num)])
                            if middle_blk_num else nn.Identity())
        info.append({"num_chs": chan, "reduction": 2 ** n, "module": "middle_blks"})
        self.feature_info = info
        self.padder_size = 2 ** (n + 3)

    def prepare(self, h, w, device):
        """The shifted-window masks for an input of h x w, built now, outside any compiled graph (torch.compile would
        otherwise trace their CPU construction)."""
        from .blocks.csa import CSAttention, shift_mask
        hp, wp = h + (-h % self.padder_size), w + (-w % self.padder_size)
        for m in self.modules():
            if isinstance(m, CSAttention) and m.shift:
                mh, mw = hp >> m.level, wp >> m.level
                shift_mask(*((min(mh, m.grid[0]), min(mw, m.grid[1])) if m.grid else (mh, mw)), device)

    def check_image_size(self, x):
        """Reflect padding of H and W to a multiple of 64 (the authors' check_image_size)."""
        h, w = x.shape[-2:]
        return F.pad(x, (0, -w % self.padder_size, 0, -h % self.padder_size), "reflect")

    def encode(self, inp):
        """inp already padded -> (skip features of the encoder stages, bottleneck, the downscaled inputs)."""
        x = self.intro(inp)
        inps = [F.interpolate(inp, scale_factor=0.5 ** (i + 1), mode="bilinear") for i in range(len(self.intros))]
        skips = []
        for encoder, down, intro, x_in in zip(self.encoders, self.downs, self.intros, inps):
            x = encoder(x)
            skips.append(x)
            x = torch.cat([down(x), intro(x_in)], dim=1)
        return skips, self.middle_blks(x), inps

    def forward(self, x):
        skips, mid, _ = self.encode(self.check_image_size(x))
        return skips + [mid]
