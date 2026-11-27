"""The full network of Concertormer: Concertormer (the encoder of encoder.py, then the decoder; restore() of
dbis/restore.py) and fold() (the authors' checkpoint -> the state dict). See dbis/concertormer/__init__.py."""
import torch
import torch.nn as nn

from ..restore import Restorer
from .blocks.csa import CSABlock
from .encoder import ConcertormerEncoder, grid_of


class Concertormer(Restorer, ConcertormerEncoder):
    """U-Net of CSA blocks with the input image fed again at every scale. tlc: None (the authors' Concertormer, the
    attention over the whole map) or "exact" (their ConcertormerLocal: the attention on windows of grid_factor x the
    training crop's size at each scale, as their test configs run it; grid_factor 1.75 for GoPro, 2.0 for REDS, 1.25 for
    RealBlur). forward(x [B, 3, H, W] in [0, 1]) -> the restored image (the last of the authors' four outputs; the
    three coarser ones, which only the training loss uses, are made by forward_all). precision: the default of
    restore(), "fp16" as the benchmark ran it. fast (dbisf): the talking heads as depthwise or grouped convs
    (blocks/csa.py talk)."""
    kind = "image"
    train_checkpoint = ("CSABlock",)

    def __init__(self, width=48, middle_blk_num=2, ffn_expansion_factor=2, enc_blk_nums=(6, 8, 24),
                 dec_blk_nums=(24, 8, 8), enc_heads=(1, 2, 4), middle_heads=8, dec_heads=(4, 2, 1), tlc="exact",
                 train_crop=256, grid_factor=1.75, precision="fp16", fast=False):
        super().__init__(3, width, middle_blk_num, ffn_expansion_factor, enc_blk_nums, enc_heads, middle_heads, tlc,
                         train_crop, grid_factor, fast)
        n = len(enc_blk_nums)
        self.ending = nn.Conv2d(width, 3, 3, padding=1, bias=True)
        self.endings = nn.ModuleList(nn.Conv2d(width * 2 ** (n - i), 3, 3, padding=1, bias=True) for i in range(n))
        self.ups, self.decoders = nn.ModuleList(), nn.ModuleList()
        chan = width * 2 ** n
        for i, (num, head) in enumerate(zip(dec_blk_nums, dec_heads)):
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            g = grid_of(tlc, train_crop, grid_factor, n - 1 - i)
            self.decoders.append(nn.Sequential(*[CSABlock(chan, head, ffn_expansion_factor, j, cross=(j == 0), grid=g,
                                                          fast=fast, level=n - 1 - i) for j in range(num)]))
        self.precision = precision

    def forward_all(self, inp):
        """The authors' forward: the outputs at 1/8, 1/4, 1/2 (each from an ending conv plus the downscaled input) and
        the full-resolution output, cropped to the input's size."""
        h, w = inp.shape[-2:]
        inp = self.check_image_size(inp)
        skips, x, inps = self.encode(inp)
        decs = []
        for decoder, up, skip, end, x_in in zip(self.decoders, self.ups, skips[::-1], self.endings, inps[::-1]):
            decs.append(end(x) + x_in)
            x = decoder(torch.cat([up(x), skip], dim=1))
        decs.append((self.ending(x) + inp)[:, :, :h, :w])
        return decs

    def forward(self, inp):
        h, w = inp.shape[-2:]
        inp = self.check_image_size(inp)
        skips, x, _ = self.encode(inp)
        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = decoder(torch.cat([up(x), skip], dim=1))
        return (self.ending(x) + inp)[:, :, :h, :w]

    def train_forward(self, x):
        """Training (dbis/train.py): the full-resolution output (the authors' loss also takes the three coarser ones,
        forward_all)."""
        return self(x)

    def _run_images(self, t, o):
        self.prepare(*t.shape[-2:], t.device)          # masks outside the compiled graph
        return super()._run_images(t, o)

    def _defaults(self):
        return {"tile": None, "tile_overlap": 32, "window": None, "batch": 1, "precision": self.precision}

    def _check(self, o):
        if any(getattr(m, "grid", None) for m in self.modules()) and o["batch"] != 1:
            raise ValueError("Concertormer with TLC windows restores one frame per call (batch 1), as the authors' code")


def fold(sd):
    """The authors' state dict (key "params" of Concertormer.pth, Concertormer_lite.pth, REDS.pth, RealJ.pth,
    RealR.pth) -> the state dict of Concertormer (the same names, float32; without a "module." prefix)."""
    return {(k[7:] if k.startswith("module.") else k): v.float().contiguous() for k, v in sd.items()}
