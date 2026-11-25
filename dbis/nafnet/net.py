"""The full network of NAFNet: NAFNet (the encoder of encoder.py, then the decoder; restore() of dbis/restore.py) and
fold() (the authors' checkpoint -> the state dict of NAFNet). See dbis/nafnet/__init__.py."""
import torch.nn as nn
import torch.nn.functional as F

from ..restore import Restorer
from .blocks.nafblock import Block
from .encoder import NAFNetEncoder, tlc_window


class NAFNet(Restorer, NAFNetEncoder):
    """U-Net of NAFBlocks (its first half: NAFNetEncoder). Defaults: NAFNet-64 of the GoPro release (enc 1, 1, 1, 28; middle 1; dec 1, 1, 1, 1).
    tlc: None (plain NAFNet), "exact" (dbis, the authors' test configuration) or "fast" (dbisf).
    restore(frames, ...) is the inference call (dbis/restore.py); forward(x) takes [B, 3, H, W] in [0, 1], any H, W
    (zero-padded to a multiple of 16 inside, as the authors do). Defaults of restore: the whole frame, one frame per
    call, float32 (as the quality of dbis and dbisf was measured)."""
    kind = "image"

    def __init__(self, width=64, enc=(1, 1, 1, 28), middle=1, dec=(1, 1, 1, 1), tlc=None, train_crop=256):
        super().__init__(width, enc, middle, tlc, train_crop)      # intro, encoders, downs, middle_blks (encoder.py)
        fast = tlc == "fast"
        self.ups, self.decoders = nn.ModuleList(), nn.ModuleList()
        c = width * 2 ** len(enc)
        for i, n in enumerate(dec):
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[Block(c, tlc_window(tlc, train_crop, len(enc) - 1 - i), fast)
                                                 for _ in range(n)]))
        self.ending = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, inp):
        h, w = inp.shape[-2:]
        inp = F.pad(inp, (0, -w % self.mult, 0, -h % self.mult))
        *skips, x = self.encode(inp)
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
