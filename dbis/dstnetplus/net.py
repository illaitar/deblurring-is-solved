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
    The dbisf options below change only the order of roundings; the maths is the same.
      fold_encoder: feat_extractor (3x3 conv), the Haar split (2x2, stride 2, / 4) and x_wave_2_conv1 (1x1) are linear
        with no activation between them, so the low band, and the high bands before their LeakyReLU, are each one 4x4
        stride-2 conv of the frame (_folded). The 96-channel full-resolution map is never formed, and with keep the high
        bands are made only for the kept frames.
      prop_channels_last: the propagation (its weights and features) in channels_last.
      autotune: _compile("default") compiles the dynamic conv with "max-autotune-no-cudagraphs" (the other blocks with
        "default").
    restore(frames, ...): window = frames per clip (the authors' GoPro test: 50), clips that do not overlap, the last one
    moved back to end at the last frame (runs/baselines/common.py clip_plan); batch = frame_chunk; no tiles."""
    kind = "video"
    min_frames = 1
    train_checkpoint = ("ResidualBlocks2D",)

    def __init__(self, num_feat=96, num_kernel_block=5, num_block=25, nonblind_denoise=False, window=50, frame_chunk=10,
                 precision="fp32", exact_dynconv=True, fold_encoder=False, prop_channels_last=False, autotune=False):
        super().__init__(num_feat, nonblind_denoise)       # feat_extractor, wave, x_wave_2_conv1/2, lrelu (encoder.py)
        self.num_feat, self.nonblind_denoise = num_feat, nonblind_denoise
        self.recons = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.forward_propagation = Propagation(num_feat, num_kernel_block, num_block, backward=False)
        self.backward_propagation = Propagation(num_feat, num_kernel_block, num_block, backward=True)
        self.window, self.frame_chunk, self.precision = window, frame_chunk, precision
        self.fold_encoder, self.prop_channels_last, self.autotune = fold_encoder, prop_channels_last, autotune
        for p in (self.forward_propagation, self.backward_propagation):
            p.kernel_conv_pixel.exact = exact_dynconv

    def _folded(self):
        """(low kernel, low bias, high kernel, high bias) of fold_encoder, made in float64 from the current weights and
        cached until they change. With f = feat_extractor(x) and h_k the four 2x2 Haar filters (low first), band k of
        the split is (h_k * f) / 4 at stride 2, so its kernel on x is the full 2-D convolution of h_k with the 3x3
        kernel, / 4 (4x4, stride 2, padding 1: the 3x3 conv's zero padding); the bias goes to the low band only (the
        high filters sum to 0). The high kernel is then multiplied by the 1x1 grouped x_wave_2_conv1 (one group per
        high band) and takes its bias."""
        ws = (self.feat_extractor.weight, self.feat_extractor.bias, self.x_wave_2_conv1.weight, self.x_wave_2_conv1.bias)
        key = tuple((w.data_ptr(), w._version) for w in ws)
        if getattr(self, "_fold_key", None) != key:
            W, b, W1, b1 = (w.detach().double() for w in ws)
            c, ci = W.shape[:2]
            hk = haar_weights(1, W.device)[:, 0].double()                   # (4, 2, 2): low, then the three high bands
            K = W.new_zeros(4, c, ci, 4, 4)
            for u in range(2):
                for v in range(2):
                    K[:, :, :, u:u + 3, v:v + 3] += hk[:, u, v].view(4, 1, 1, 1, 1) * W
            K /= 4
            kh = torch.einsum("gom,gmipq->goipq", W1.view(3, c, c), K[1:]).reshape(3 * c, ci, 4, 4)
            object.__setattr__(self, "_fold", (K[0].float().contiguous(), b.float(), kh.float().contiguous(), b1.float()))
            object.__setattr__(self, "_fold_key", key)
        return self._fold

    def _encode_folded(self, x, high=True):
        kl, bl, kh, bh = self._folded()
        low = F.conv2d(x, kl, bl, stride=2, padding=1)
        if not high:
            return low, None
        return low, self.x_wave_2_conv2(self.lrelu(F.conv2d(x, kh, bh, stride=2, padding=1)))

    def _set_prop_layout(self):
        cl = bool(self.prop_channels_last)
        if getattr(self, "_prop_cl", False) != cl:
            for p in (self.forward_propagation, self.backward_propagation):
                p.to(memory_format=torch.channels_last if cl else torch.contiguous_format)
                p.channels_last = cl
            object.__setattr__(self, "_prop_cl", cl)

    def train_forward(self, lrs):
        """Training (dbis/train.py): lrs (B, T, 3, H, W) -> (B, T, 3, H, W), the whole clip at once. The encoder
        unfolded (the folded kernels are made without gradients) and the dynamic conv as plain addcmul."""
        fold, self.fold_encoder = self.fold_encoder, False
        for p in (self.forward_propagation, self.backward_propagation):
            p.kernel_conv_pixel.exact = False
        try:
            return self._forward(lrs, None)
        finally:
            self.fold_encoder = fold

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
        k0, k1 = keep or (0, t)
        self._set_prop_layout()
        lows, highs = [], []
        if self.fold_encoder:                              # the low band of every frame, the high bands of the kept ones
            lows = [self._encode_folded(x, high=False)[0] for x in flat.split(cs, dim=0)]
            highs = [self._encode_folded(x)[1] for x in lrs[:, k0:k1].flatten(0, 1).split(cs, dim=0)]
        else:
            for x in flat.split(cs, dim=0):
                _, low, high = self.encode(x)
                lows.append(low)
                highs.append(high)
        low = lows[0] if len(lows) == 1 else torch.cat(lows, dim=0)
        high = highs[0] if len(highs) == 1 else torch.cat(highs, dim=0)
        del lows, highs
        low = self.backward_propagation(low.view(b, t, *low.shape[1:]))
        low = self.forward_propagation(low[:, :k1])[:, k0:].flatten(0, 1)
        if keep:
            if not self.fold_encoder:
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
            dyn = "max-autotune-no-cudagraphs" if self.autotune and mode == "default" else mode
            for p in (self.forward_propagation, self.backward_propagation):
                p.fusion = torch.compile(p.fusion, mode=mode, dynamic=False)
                p.kernel_conv_pixel = torch.compile(p.kernel_conv_pixel, mode=dyn, dynamic=False)
                p.resblock_bcakward2d = torch.compile(p.resblock_bcakward2d, mode=mode, dynamic=False)


def fold(sd):
    """The authors' state dict (key "params" of DSTNetPlus_*.pth) -> the state dict of DSTNetPlus: without "module."
    and without wave.haar_weights (the fixed Haar weights, checked equal to haar_weights() and rebuilt at run time)."""
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    hw = sd.pop("wave.haar_weights")
    assert torch.equal(hw, haar_weights(hw.shape[0] // 4)), "wave.haar_weights differ from the Haar filters"
    return {k: v.float().contiguous() for k, v in sd.items()}
