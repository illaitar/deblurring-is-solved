"""The full network of DSTNet+: DSTNetPlus (the per-frame encoder of encoder.py, backward and forward propagation of
the low band at half resolution, inverse Haar, reconstruction; restore() of dbis/restore.py) and fold() (the authors'
checkpoint -> the state dict of DSTNetPlus). See dbis/dstnetplus/__init__.py."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..restore import Restorer
from .blocks.haar import haar_weights
from .blocks.propagation import Propagation
from .encoder import DSTNetPlusEncoder


class DSTNetPlus(Restorer, DSTNetPlusEncoder):
    """DSTNetPlus_Final of the authors. forward(lrs (b, t, 3, h, w) in [0, 1]) -> (b, t, 3, h, w), one clip of t frames
    (h and w padded to even inside by reflection, as the authors do). Defaults: DSTNet+ L (num_feat 96, 5 kernel blocks,
    25 residual blocks); base: 64, 3, 15.
    frame_chunk: frames at a time through the per-frame, full-resolution layers (encoder, inverse Haar, reconstruction):
    the same maths in pieces, as our patch of the authors' code runs it (ext/DSTNet-plus.patch, 10 on GoPro; without it
    the 50-frame 720p clips do not fit). precision: "fp32" (the authors' code, whose CuPy kernel has no float16) or
    "fp16" (autocast). exact_dynconv: the dynamic depthwise conv rounded as the authors' CuPy kernel (blocks/dynconv.py).
    restore(frames, ...): window = frames per clip (the authors' GoPro test: 50), clips that do not overlap, the last one
    moved back to end at the last frame (runs/baselines/common.py clip_plan); batch = frame_chunk; no tiles."""
    kind = "video"
    min_frames = 1

    def __init__(self, num_feat=96, num_kernel_block=5, num_block=25, nonblind_denoise=False, window=50, frame_chunk=10,
                 precision="fp32", exact_dynconv=True):
        super().__init__(num_feat, nonblind_denoise)       # feat_extractor, wave, x_wave_2_conv1/2, lrelu (encoder.py)
        self.num_feat, self.nonblind_denoise = num_feat, nonblind_denoise
        self.recons = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.forward_propagation = Propagation(num_feat, num_kernel_block, num_block, backward=False)
        self.backward_propagation = Propagation(num_feat, num_kernel_block, num_block, backward=True)
        self.window, self.frame_chunk, self.precision = window, frame_chunk, precision
        for p in (self.forward_propagation, self.backward_propagation):
            p.kernel_conv_pixel.exact = exact_dynconv

    @torch.no_grad()
    def forward(self, lrs, precision=None, frame_chunk=None, keep=None):
        """keep=(lo, hi): only the outputs of frames lo .. hi-1, (b, hi - lo, 3, h, w). The forward propagation is
        causal, so it runs on frames 0 .. hi-1 only, and the inverse Haar and reconstruction on the kept frames: the
        same values for those frames (a sliding window that emits its oldest frame needs one forward step)."""
        with torch.autocast(lrs.device.type, dtype=torch.float16, enabled=(precision or self.precision) == "fp16"):
            return self._forward(lrs, frame_chunk or self.frame_chunk, keep)

    def _forward(self, lrs, chunk, keep=None):
        b, t, c, h, w = lrs.size()
        if h % 2 != 0 or w % 2 != 0:                       # the authors' spatial_padding (reflect, to a multiple of 2)
            lrs = F.pad(lrs.view(-1, c, h, w), [0, w % 2, 0, h % 2], mode="reflect").view(b, t, c, h + h % 2, w + w % 2)
        flat = lrs.flatten(0, 1)                           # (b t) c h w
        cs = chunk or flat.shape[0]
        lows, highs = [], []
        for x in flat.split(cs, dim=0):
            _, low, high = self.encode(x)
            lows.append(low)
            highs.append(high)
        low = lows[0] if len(lows) == 1 else torch.cat(lows, dim=0)
        high = highs[0] if len(highs) == 1 else torch.cat(highs, dim=0)
        del lows, highs
        low = self.backward_propagation(low.view(b, t, *low.shape[1:]))
        k0, k1 = keep or (0, t)
        low = self.forward_propagation(low[:, :k1])[:, k0:].flatten(0, 1)
        if keep:
            high = high.view(b, t, *high.shape[1:])[:, k0:k1].flatten(0, 1)
            lrs = lrs[:, k0:k1]
        outs = [self.recons(self.wave.inverse(torch.cat([lo, hi], dim=1)))
                for lo, hi in zip(low.split(cs, dim=0), high.split(cs, dim=0))]
        out = (outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)).view(b, k1 - k0, *outs[0].shape[1:])
        out = out.contiguous() + (lrs[:, :, :3] if self.nonblind_denoise else lrs)
        return out[:, :, :, :h, :w]

    # ------------------------------------------------------------ restore() (dbis/restore.py)
    def _defaults(self):
        return {"tile": None, "tile_overlap": 0, "window": self.window, "batch": self.frame_chunk,
                "precision": self.precision}

    def _check(self, o):
        if o["tile"] is not None:
            raise ValueError("DSTNet+ runs on whole frames (tile=None)")

    def _plan(self, t, n):
        """The authors' non-overlapping n-frame clips; the last one moved back to end at the last frame, only its frames
        not covered before are kept (runs/baselines/common.py clip_plan)."""
        if t <= n:
            return [(0, t, 0, t)]
        plan = [(i * n, i * n + n, i * n, i * n + n) for i in range(t // n)]
        if t % n:
            plan.append((t - n, t, (t // n) * n, t))
        return plan

    def _window(self, x, o):
        return self.forward(x, precision=o["precision"], frame_chunk=o["batch"]).float()

    def _compile(self, mode):
        if getattr(self, "_compiled", None) is None:
            object.__setattr__(self, "_compiled", True)
            for p in (self.forward_propagation, self.backward_propagation):
                p.fusion = torch.compile(p.fusion, mode=mode, dynamic=False)
                p.kernel_conv_pixel = torch.compile(p.kernel_conv_pixel, mode=mode, dynamic=False)
                p.resblock_bcakward2d = torch.compile(p.resblock_bcakward2d, mode=mode, dynamic=False)


def fold(sd):
    """The authors' state dict (key "params" of DSTNetPlus_*.pth) -> the state dict of DSTNetPlus: without "module."
    and without wave.haar_weights (the fixed Haar weights, checked equal to haar_weights() and rebuilt at run time)."""
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    hw = sd.pop("wave.haar_weights")
    assert torch.equal(hw, haar_weights(hw.shape[0] // 4)), "wave.haar_weights differ from the Haar filters"
    return {k: v.float().contiguous() for k, v in sd.items()}
