"""The full network of Turtle: Turtle (the per-frame encoder of encoder.py, the bottleneck and decoder levels that carry
the history of past frames, refinement, output conv; restore() of dbis/restore.py and step()) and fold() (the authors'
checkpoint -> the state dict of Turtle). See dbis/turtle/__init__.py."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..restore import Restorer
from .blocks.common import Upsample
from .blocks.history import local_mask
from .blocks.level import LatentCacheBlock, LevelBlock
from .encoder import TurtleEncoder


class Turtle(Restorer, TurtleEncoder):
    """Turtle_t1 of the authors with their GoPro configuration (Turtle_Deblur_Gopro.yml). Causal: frame t is restored
    from blurry frame t and the caches of the frames before it (keys and values of the bottleneck's two FHR blocks, the
    aligned states of the three decoder CHM blocks); the previous blurry frame, which the authors' forward also takes,
    is not used by this configuration (use_both_input False) and is not passed.

    forward(x (b, 3, h, w) in [0, 1], k_cached, v_cached) -> (out (b, 3, h, w), k_cache, v_cache): one step; the caches
    are lists of 8 (None, or the tensors of the blocks with history, in the authors' order), None at the first frame.

    step(frame (3, h, w), state) -> (out (3, h, w), state): one frame as the benchmark runs the authors' code
    (basicsr/inference.py run_inference_patched): the frame reflect-padded to a multiple of 8, cut into square tiles of
    `tile` px that overlap by `tile_overlap`, every tile with its own caches, the outputs averaged where tiles overlap;
    tile=None: the whole frame (their do_patched=False). state=None starts a sequence.

    Options: tile, tile_overlap (the authors' GoPro test: 320 and 192); batch = tiles per network call (1: the
    authors', bit for bit); precision "fp32" or "fp16" (autocast); fast: SAB's two attention masks applied together
    (the same values, fewer kernels)."""
    kind = "video"
    min_frames = 1
    train_checkpoint = ("TurtleAttnBlock",)

    def __init__(self, dim=64, enc_blocks=(2, 6, 10), middle_blocks=11, dec_blocks=(10, 6, 2), num_refinement_blocks=2,
                 num_heads=(1, 2, 4, 8), ffn_expansion_factor=2.5, bias=False, LayerNorm_type="WithBias",
                 num_frames_tocache=3, tile=320, tile_overlap=192, batch=1, precision="fp32", fast=False):
        super().__init__(dim, enc_blocks, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                         num_frames_tocache=num_frames_tocache)        # input_projection, encoder levels (encoder.py)
        kw = dict(ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.latent = LatentCacheBlock(dim * 8, num_blocks=middle_blocks, attn_type1="FHR", attn_type2="Channel",
                                       attn_type3="FHR", FFW_type="GFFW", num_frames_tocache=num_frames_tocache,
                                       num_heads=num_heads[3], **kw)
        dec = dict(attn_type1="Channel", attn_type2="CHM", FFW_type="GFFW", fast=fast, **kw)
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = LevelBlock(dim * 4, num_blocks=dec_blocks[0], num_frames_tocache=num_frames_tocache,
                                         num_heads=num_heads[2], Scale_patchsize=2, **dec)
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = LevelBlock(dim * 2, num_blocks=dec_blocks[1], num_frames_tocache=num_frames_tocache,
                                         num_heads=num_heads[1], Scale_patchsize=4, **dec)
        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Conv2d(dim * 2, dim, 1, bias=bias)
        self.decoder_level1 = LevelBlock(dim, num_blocks=dec_blocks[2], num_frames_tocache=2, num_heads=num_heads[0],
                                         Scale_patchsize=8, **dec)
        self.refinement = LevelBlock(dim, num_blocks=num_refinement_blocks, attn_type1="ReducedAttn",
                                     attn_type2="ReducedAttn", FFW_type="GFFW", num_frames_tocache=num_frames_tocache,
                                     num_heads=num_heads[0], **kw)
        self.ending = nn.Conv2d(dim, 3, 3, 1, 1, bias=True)
        self.tile, self.tile_overlap, self.batch, self.precision = tile, tile_overlap, batch, precision

    # ------------------------------------------------------------ the network
    def forward(self, x, k_cached=None, v_cached=None):
        h, w = x.shape[-2:]
        x = self.pad(x)
        kc, vc = k_cached or [None] * 8, v_cached or [None] * 8
        e1, e2, e3, e4 = self.encode(x)
        lat, k4, v4, k5, v5 = self.latent(e4, kc[3], vc[3], kc[4], vc[4])
        d3, k6, v6 = self.decoder_level3(self.reduce_chan_level3(torch.cat([self.up4_3(lat), e3], 1)), kc[5], vc[5])
        d2, k7, v7 = self.decoder_level2(self.reduce_chan_level2(torch.cat([self.up3_2(d3), e2], 1)), kc[6], vc[6])
        d1, k8, v8 = self.decoder_level1(self.reduce_chan_level1(torch.cat([self.up2_1(d2), e1], 1)), kc[7], vc[7])
        d1, _, _ = self.refinement(d1)
        out = self.ending(d1) + x
        return out[:, :, :h, :w], [None, None, None, k4, k5, k6, k7, k8], [None, None, None, v4, v5, v6, v7, v8]

    def _net(self, x, kv, precision):
        h, w = x.shape[-2:]
        local_mask(-(-h // 32) * 2, -(-w // 32) * 2, 4, x.device)  # SAB's token grid (padded size / 16), made outside
        fwd = getattr(self, "_compiled", None) or self.forward    # a compiled graph (the graph then only reads it)
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=precision == "fp16"):
            out, k, v = fwd(x, *kv)
        return out.float(), (k, v)

    def train_forward(self, x):
        """Training (dbis/train.py): x (B, T, 3, H, W) -> (B, T, 3, H, W), frame by frame from an empty history, the
        caches carried through the clip (gradients through them)."""
        h, w = x.shape[-2:]
        local_mask(-(-h // 32) * 2, -(-w // 32) * 2, 4, x.device)
        kc = vc = None
        outs = []
        for t in range(x.shape[1]):
            y, kc, vc = self(x[:, t], kc, vc)
            outs.append(y)
        return torch.stack(outs, 1)

    # ------------------------------------------------------------ one frame
    @torch.no_grad()
    def step(self, frame, state=None, tile="default", tile_overlap=None, batch=None, precision=None):
        """frame (3, h, w) float in [0, 1] on the model's device -> (out (3, h, w) float32, not clamped, state)."""
        tile = self.tile if tile == "default" else tile
        overlap = self.tile_overlap if tile_overlap is None else tile_overlap
        batch, precision = batch or self.batch, precision or self.precision
        if tile is None:                                   # the whole frame
            out, kv = self._net(frame[None], state or (None, None), precision)
            return out[0], kv
        c, h0, w0 = frame.shape
        ph, pw = -h0 % 8, -w0 % 8                          # the authors' img_multiple_of = 8, reflect
        x = F.pad(frame[None], (0, pw, 0, ph), mode="reflect")
        _, _, h, w = x.shape
        t = min(tile, h, w)
        stride = t - overlap
        boxes = [(i, j) for i in list(range(0, h - t, stride)) + [h - t] for j in list(range(0, w - t, stride)) + [w - t]]
        acc = torch.zeros(1, c, h, w, device=x.device)
        cnt = torch.zeros_like(acc)
        new = []
        for n, s in enumerate(range(0, len(boxes), batch)):
            chunk = boxes[s:s + batch]
            xt = torch.cat([x[..., i:i + t, j:j + t] for i, j in chunk])
            out, kv = self._net(xt, state[n] if state else (None, None), precision)
            new.append(kv)
            for m, (i, j) in enumerate(chunk):
                acc[..., i:i + t, j:j + t].add_(out[m:m + 1])
                cnt[..., i:i + t, j:j + t].add_(torch.ones_like(out[m:m + 1]))
        return acc.div_(cnt)[0, :, :h0, :w0], new

    # ------------------------------------------------------------ restore() (dbis/restore.py)
    def _defaults(self):
        return {"tile": self.tile, "tile_overlap": self.tile_overlap, "window": None, "batch": self.batch,
                "precision": self.precision}

    def _check(self, o):
        if o["tile"] is not None and (not isinstance(o["tile"], int) or o["tile"] % 8):
            raise ValueError(f"Turtle: tile is one int, a multiple of 8 (square tiles, as the authors), not {o['tile']!r}")
        if not isinstance(o["tile_overlap"], int):
            raise ValueError(f"Turtle: tile_overlap is one int, not {o['tile_overlap']!r}")

    def _restore_video(self, x, y, o, prep, put):
        """Frame by frame; the history starts at the first frame of every clip of the batch, and again every `window`
        frames if window is set (None: one pass over the clip, the benchmark)."""
        b, t = x.shape[:2]
        n = o["window"] or t
        for i in range(b):
            state = None
            for j in range(t):
                if j % n == 0:
                    state = None
                out, state = self.step(prep(x[i, j]), state, o["tile"], o["tile_overlap"], o["batch"], o["precision"])
                put(y[i, j], out.to(y.device))

    def _compile(self, mode):
        if getattr(self, "_compiled", None) is None:
            # one graph per cache size (the caches grow over the first 4 frames) and per tile batch and frame size
            cfg = torch._dynamo.config
            key = "recompile_limit" if hasattr(cfg, "recompile_limit") else "cache_size_limit"
            setattr(cfg, key, max(getattr(cfg, key), 64))
            object.__setattr__(self, "_compiled", torch.compile(self.forward, mode=mode, dynamic=False))


def fold(sd):
    """The authors' state dict (key "params" of GoPro_Deblur.pth) -> the state dict of Turtle (the same names; without a
    "module." prefix if the checkpoint has one)."""
    return {(k[7:] if k.startswith("module.") else k): v.float().contiguous() for k, v in sd.items()}
