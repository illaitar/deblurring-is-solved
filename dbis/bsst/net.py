"""The full network of BSSTNet: BSST (the restoration network on a batch of patches), BSSTNet (RAFT flows + BSST on
the 256 px patches of a window, the authors' test_by_patch; restore() of dbis/restore.py), fold() (the authors'
checkpoints -> the state dict of BSSTNet). See dbis/bsst/__init__.py for the credits and what differs from the
authors' code."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..restore import Restorer
from .blocks.align import BlurGuidedAlignment, FlowAlignment
from .blocks.cnn import CAB, CAB1, CAB2, PixelShufflePack, ShiftBlock, SkipUpSample, conv
from .blocks.common import lrelu
from .blocks.flow import fb_valid, flow_warp, get_blur_map
from .blocks.flow_completion import FlowCompletion
from .blocks.raft import RAFT
from .blocks.sparse_transformer import (SoftComp, SoftSplit, SparseWindowAttention, TemporalSparseTransformerBlock,
                                        sparse_attention_constants)
from .blocks.swin import Reconstruction, WindowAttention, relative_position_index
from .encoder import BSSTEncoder


class BSST(BSSTEncoder):
    """The BSST network of the authors (its first part: BSSTEncoder) (BSST_arch.py, defaults of gopro_BSST.yml) for a batch of patches:
    forward(lqs (n, t, 3, h, w), flows_forward, flows_backward (n, t-1, 2, h/4, w/4)) -> (n, t, 3, h, w)."""

    def __init__(self, mid=192, n_feat=64):
        super().__init__(mid, n_feat)                  # the encoder modules (encoder.py)
        self.mid_channels = mid
        self.act = lrelu()
        self.decoder_level1 = ShiftBlock(n_feat)
        self.decoder_level1_1 = ShiftBlock(n_feat)
        self.skip_attn1 = CAB(n_feat)
        self.upsample0 = PixelShufflePack(n_feat, n_feat, 2, 3)
        self.skip_conv = CAB(n_feat)
        self.out_conv = CAB(n_feat)
        self.last_conv = conv(n_feat, 3, 3)
        self.conv_hr0 = conv(n_feat, n_feat, 3)
        self.up21 = SkipUpSample(n_feat, 128)
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        for i, m in enumerate(("backward_1", "forward_1", "backward_2", "forward_2")):
            self.deform_align[m] = BlurGuidedAlignment(2 * mid, mid)
            self.backbone[m] = nn.Sequential(nn.Conv2d((2 + i) * mid, mid, 3, 1, 1), lrelu(), CAB1(mid, 3))
        self.reconstruction = Reconstruction(5 * mid, mid)
        t2t = {"kernel_size": (5, 5), "stride": (2, 2), "padding": (2, 2)}
        self.depths = 4
        self.fold_feat_size = (64, 64)
        self.softsplit = SoftSplit(mid, 512, (5, 5), (2, 2), (2, 2))
        self.softcomp = SoftComp(mid, 512, (5, 5), (2, 2), (2, 2))
        self.transsqattn = TemporalSparseTransformerBlock(512, 4, (8, 8), (4, 4), self.depths, t2t)
        self.cabattn = nn.Sequential(CAB2(mid, 5, mid), CAB1(mid, 5))
        self.blur_motion_refine = FlowCompletion()

    def propagate(self, feats, flows, flows_check, blur_motion_map, module_name, out):
        """One propagation branch. The features are written into out (n, t+1, 192, h, w), a channel slice of the
        float32 buffer that the authors build afterwards with stack + cat; feats keeps views of it. out=None (training,
        where autograd cannot follow writes into a shared buffer): feats keeps the features themselves."""
        n, t, _, h, w = flows.size()
        last_key = list(feats)[-2]
        if "backward" in module_name:
            frame_idx = flow_idx = range(t, -1, -1)
        else:
            frame_idx, flow_idx = range(0, t + 1), range(-1, t)
        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats[last_key][idx]
            blur_map_current = blur_motion_map[:, frame_idx[i], :, :, :]
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                flow_n1_check = flows_check[:, flow_idx[i], :, :, :]
                blur_map_n1 = blur_motion_map[:, frame_idx[i - 1], :, :, :]
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                blur_map_n1 = flow_warp(blur_map_n1, flow_n1.permute(0, 2, 3, 1))
                flow_n1_valid = fb_valid(flow_n1, flow_n1_check)
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
                blur_map_n2 = torch.zeros_like(blur_map_n1)
                flow_n2_valid = torch.zeros_like(flow_n1_valid)
                if i > 1:   # second-order features
                    feat_n2 = feats[module_name][-2]
                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    flow_n2_check = flows_check[:, flow_idx[i - 1], :, :, :]
                    blur_map_n2 = blur_motion_map[:, frame_idx[i - 2], :, :, :]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    flow_n2_check = flow_n1_check + flow_warp(flow_n2_check, flow_n1_check.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
                    blur_map_n2 = flow_warp(blur_map_n2, flow_n2.permute(0, 2, 3, 1))
                    flow_n2_valid = fb_valid(flow_n2, flow_n2_check)
                cond = torch.cat([cond_n1, feat_current, cond_n2, blur_map_n1, blur_map_current, blur_map_n2,
                                  flow_n1_valid, flow_n2_valid], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2,
                                                           blur_map_n1 * flow_n1_valid, blur_map_n2 * flow_n2_valid)
            feat = torch.cat([feats[k][idx] for k in feats if k != module_name] + [feat_prop], dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            if out is None:
                feats[module_name].append(feat_prop)
            else:
                out[:, idx].copy_(feat_prop)
                feats[module_name].append(out[:, idx])
        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]
        return feats

    def sparse_masks(self, blur_map_st):
        """Query and key/value masks of the 4 sparse attention depths, (n, t, 4, 4, 1) each, as the authors compute
        them, including the in-place zeroing of blur_map_st_win through its alias."""
        n, t, c, h, w = blur_map_st.shape
        win = F.avg_pool2d(blur_map_st.view(n * t, c, h, w), (5, 5), (2, 2), (2, 2))
        win = F.avg_pool2d(win, (8, 8), (8, 8), (0, 0))
        win = win.view(n * t, 1, win.shape[-2], win.shape[-1]).view(n, t, 1, win.shape[-2], win.shape[-1])
        hard = torch.where(win > 0.3, torch.ones_like(win), torch.zeros_like(win))
        dev = blur_map_st.device
        t_ind = [torch.arange(i, t, 2, device=dev) for i in range(2)] * (self.depths // 2)
        qmasks, kvmasks = [], []
        for t_idxs, t_zero in zip(t_ind, t_ind[::-1]):
            mask = hard[:, t_idxs].sum(1, keepdim=True).clip(0, 1)
            q_soft = win                       # alias: the zeroing below stays in win for the next depths
            q_soft[:, t_zero] = q_soft[:, t_zero] * 0.
            q_idx = torch.argsort(q_soft, 1, True)
            qmask = torch.zeros_like(win)
            kv_soft = (1 - win)
            kv_soft[:, t_zero] = kv_soft[:, t_zero] * 0.
            kv_idx = torch.argsort(kv_soft, 1, True)
            kvmask = torch.zeros_like(win)
            kvmask = kvmask.scatter_(1, kv_idx[:, :len(t_idxs) // 2], 1) * mask
            qmask = qmask.scatter_(1, q_idx[:, :len(t_idxs) // 2], 1) * mask
            qmasks.append(qmask.permute(0, 1, 3, 4, 2).contiguous())
            kvmasks.append(kvmask.permute(0, 1, 3, 4, 2).contiguous())
        return qmasks, kvmasks

    def forward(self, lqs, flows_forwards_gt, flows_backwards_gt, stage_batch=None):
        """stage_batch: patches per sub-batch of the feed-forward stages (encoder, reconstruction, sparse transformer,
        decoder), which bounds the memory; the recurrent parts (flow completion, the four propagation branches) run on
        all n patches at once. None: all n."""
        n, t, _, h, w = lqs.size()
        sb = stage_batch or n
        parts = [slice(s, min(s + sb, n)) for s in range(0, n, sb)]
        pred_map = get_blur_map(flows_forwards_gt, flows_backwards_gt)
        flows_forward, flows_backward = self.blur_motion_refine.forward_bidirect_flow(
            pred_map, flows_forwards_gt, flows_backwards_gt)
        blur_map_st = get_blur_map(flows_forward, flows_backward)
        shortcuts, enc11s, hr = [], [], None
        train = torch.is_grad_enabled()            # training (train_forward): no shared buffer, hr made by torch.cat
        spatial = []
        for sl in parts:
            x, enc11, enc22 = self.encode(lqs[sl].flatten(0, 1))     # the authors: lqs[0]
            shortcuts.append(x)
            enc11s.append(enc11)
            if train:
                spatial.append(enc22.view(-1, t, *enc22.shape[1:]))
                continue
            if hr is None:   # the authors' torch.cat of [spatial, backward_1, forward_1, backward_2, forward_2]
                hr = torch.empty(n, t, 5 * self.mid_channels, *enc22.shape[-2:], device=lqs.device,
                                 dtype=torch.promote_types(enc22.dtype, flows_forward.dtype))
            hr[sl, :, :self.mid_channels].copy_(enc22.view(-1, t, *enc22.shape[1:]))
            del x, enc11, enc22
        # the spatial features enter the propagation only through torch.cat with float32 tensors, so their float32
        # copies in hr give the same values as the authors' float16 originals
        if train:
            spatial = torch.cat(spatial) if len(spatial) > 1 else spatial[0]
            feats = {"spatial": [spatial[:, i] for i in range(t)]}
        else:
            feats = {"spatial": [hr[:, i, :self.mid_channels] for i in range(t)]}
        blur_motion = 1 - blur_map_st
        for k, (iter_, direction) in enumerate(((1, "backward"), (1, "forward"), (2, "backward"), (2, "forward"))):
            flows, check = ((flows_backward, flows_forward) if direction == "backward"
                            else (flows_forward, flows_backward))
            name = f"{direction}_{iter_}"
            feats[name] = []
            c0 = (k + 1) * self.mid_channels
            feats = self.propagate(feats, flows, check, blur_motion, name,
                                   None if train else hr[:, :, c0:c0 + self.mid_channels])
        if train:
            hr = torch.cat([torch.stack(feats[k], 1) for k in ("spatial", "backward_1", "forward_1", "backward_2",
                                                               "forward_2")], 2)
        del feats
        qmasks, kvmasks = self.sparse_masks(blur_map_st)
        outs = []
        for p, sl in enumerate(parts):
            shortcut, enc11 = shortcuts[p], enc11s[p]
            shortcuts[p] = enc11s[p] = None
            nc = sl.stop - sl.start
            input_feats = self.reconstruction(hr[sl])
            b, t, c, h4, w4 = input_feats.shape
            trans = self.softsplit(input_feats.view(b * t, c, h4, w4), nc, self.fold_feat_size)
            trans = self.transsqattn(trans, self.fold_feat_size, [m[sl] for m in qmasks], [m[sl] for m in kvmasks])
            trans = self.softcomp(trans, t, self.fold_feat_size).view(b, t, c, h4, w4)
            trans = self.cabattn(torch.cat([input_feats, trans], 2).flatten(0, 1))   # the authors: [0], unsqueeze(0)
            del input_feats
            x = self.up21(trans, self.skip_attn1(enc11))
            del trans
            dec11 = self.decoder_level1_1(self.decoder_level1(x, nc), nc)
            del x
            out = self.conv_hr0(self.act(self.upsample0(dec11))) + self.skip_conv(shortcut)
            out = self.last_conv(self.out_conv(out))
            outs.append(out.view(nc, t, *out.shape[1:]) + lqs[sl])
            del dec11, out, shortcut, enc11
        return outs[0] if len(outs) == 1 else torch.cat(outs)


class BSSTNet(Restorer, nn.Module):
    """RAFT flows + BSST on overlapping patches of a window of frames (the authors' test_by_patch).

    restore(frames, ...) is the inference call for a clip of any length (dbis/restore.py): windows of `window` frames,
    tile = the 256 px patches, tile_overlap = overlap, batch = patch_batch, precision as below.

    forward(lqs (B, T, 3, H, W) in [0, 1], H and W multiples of 16, >= 256) -> (B, T, 3, H, W) float32 on lqs.device.
    patch_batch: patches per BSST call (1 = one patch per call as the authors, the same outputs bit for bit; default 14
    = two calls for a 720p window); stage_batch: patches per sub-batch of the feed-forward stages inside a call (memory
    bound, see BSST.forward); precision: "fp16" (the authors' autocast) or "fp32". All three can also be given to
    forward(). nan_fallback (dbisf): a patch whose float16 output has a NaN or Inf (float16 overflow; the authors' code
    then writes a black 256 px tile, on 15 % of the REDS and 13 % of the BSD frames with the GoPro weights) is computed
    again in float32; self.fallbacks counts those patches. Values that stay non-finite in float32 too come from the
    recurrent propagation, whose features grow until a LayerNorm overflows (out of domain; the authors' float32 code
    gives the same black tiles): with nan_fallback those pixels take the input frame (blurry, not black);
    self.input_fallbacks counts the patches.
    Speed options (25.11 tests; all off for dbis):
      flow_cache  keep the flows of the last window; a window that repeats its frames shifted by k (a sliding window)
                  runs RAFT only on its k new pairs (RAFT works per pair, so only the batch size of its kernels changes)
      tf32        float32 matrix products in TF32 inside forward (torchvision's deform_conv2d GEMMs, RAFT correlation)
      dcn         "tv" (torchvision, as the authors), "tv_half" (torchvision in float16), "gs" (grid_sample sampling
                  in float32 and the 3x3 sum as a float16 1x1 conv, _dcn_gs)
      sdpa        scaled_dot_product_attention in the sparse attention and the Swin window attention
      window      the number of frames of a window for restore() and the benchmark's sliding window (the authors: 48)
      overlap     of the 256 px patches, one int or (rows, columns) (the authors: 64; 24 gives 3 x 6 = 18 patches for 720p
                  instead of 4 x 7 = 28, 48 gives 4 x 6 = 24, (24, 64) gives 3 x 7 = 21)"""

    def __init__(self, patch_batch=14, precision="fp16", stage_batch=4, patch=256, overlap=64, nan_fallback=False,
                 flow_cache=False, tf32=False, dcn="tv", sdpa=False, window=48):
        super().__init__()
        self.bsst = BSST()
        self.raft = RAFT()
        self.patch_batch, self.precision, self.stage_batch = patch_batch, precision, stage_batch
        self.patch, self.overlap, self.nan_fallback = patch, overlap, nan_fallback
        self.fallbacks = self.input_fallbacks = 0
        self.flow_cache, self.tf32, self._cache = flow_cache, tf32, None
        self.window = window
        self.set_kernels(dcn, sdpa)

    COMPILE_BLOCKS = ("concat", "down01", "encoder_level1", "encoder_level1_1", "down12", "encoder_level2",
                      "encoder_level2_1", "decoder_level1", "decoder_level1_1", "skip_attn1", "upsample0", "skip_conv",
                      "out_conv", "up21", "reconstruction", "cabattn")

    def compile_blocks(self, mode="default"):
        """torch.compile of the heavy blocks of BSST one by one, each its own graph (the whole BSST forward does not
        compile): 1.3x faster on an A100 after about 3.5 min of compilation (mode "default") for each frame size; the
        outputs change by float rounding only."""
        b = self.bsst
        for name in self.COMPILE_BLOCKS:
            setattr(b, name, torch.compile(getattr(b, name), mode=mode, dynamic=False))
        for k in list(b.backbone):
            b.backbone[k] = torch.compile(b.backbone[k], mode=mode, dynamic=False)
            b.deform_align[k].conv_offset = torch.compile(b.deform_align[k].conv_offset, mode=mode, dynamic=False)
        return self

    # ------------------------------------------------------------ restore() (dbis/restore.py)
    kind = "video"

    def _defaults(self):
        return {"tile": self.patch, "tile_overlap": self.overlap, "window": self.window, "batch": self.patch_batch,
                "precision": self.precision}

    def _check(self, o):
        t = o["tile"]
        if (t, t) != (self.patch, self.patch) if isinstance(t, int) else tuple(t) != (self.patch, self.patch):
            raise ValueError(f"BSSTNet runs on {self.patch} px patches only (its sparse attention is built for them)")

    def _pad(self, x):
        return 16, self.patch

    def _window(self, x, o):
        saved = self.overlap
        self.overlap = o["tile_overlap"]
        try:
            return self.forward(x, patch_batch=o["batch"], precision=o["precision"])
        finally:
            self.overlap = saved

    def _compile(self, mode):
        if not getattr(self, "_blocks_compiled", False):
            self.compile_blocks(mode)
            self._blocks_compiled = True

    def set_kernels(self, dcn="tv", sdpa=False):
        for m in self.modules():
            if isinstance(m, (BlurGuidedAlignment, FlowAlignment)):
                m.dcn = dcn
            elif isinstance(m, (WindowAttention, SparseWindowAttention)):
                m.sdpa = sdpa

    def cached_flows(self, lqs):
        """flows(lqs), reusing the pairs of the previous window when lqs[:, :t-k] equals its frames [k:]."""
        t, c = lqs.shape[1], self._cache
        ff = fb = None
        if c is not None and c[0].shape == lqs.shape:
            for k in range(1, t - 1):
                if torch.equal(lqs[:, 0], c[0][:, k]) and torch.equal(lqs[:, :t - k], c[0][:, k:]):
                    nf, nb = self.flows(lqs[:, t - k - 1:])
                    ff, fb = torch.cat([c[1][:, k:], nf], 1), torch.cat([c[2][:, k:], nb], 1)
                    self.cache_hits += 1
                    break
        if ff is None:
            ff, fb = self.flows(lqs)
        self._cache = (lqs.clone(), ff, fb)
        return ff, fb

    cache_hits = 0

    def flows(self, lq):
        """get_bi_flows of the authors: (forward, backward) flows (B, T-1, 2, H/4, W/4), RAFT in float32."""
        b, t, c, h, w = lq.shape
        with torch.autocast(lq.device.type, enabled=False):
            lq1 = F.interpolate(lq[:, :-1, ...].reshape(b * (t - 1), c, h, w), scale_factor=0.5, mode="bilinear",
                                align_corners=False)
            lq2 = F.interpolate(lq[:, 1:, ...].reshape(b * (t - 1), c, h, w), scale_factor=0.5, mode="bilinear",
                                align_corners=False)
            fwd = self.raft(lq2, lq1)
            bwd = self.raft(lq1, lq2)
            fwd = F.interpolate(fwd, scale_factor=0.5, mode="bilinear", align_corners=False)
            bwd = F.interpolate(bwd, scale_factor=0.5, mode="bilinear", align_corners=False)
        return fwd.view(b, t - 1, 2, h // 4, w // 4) / 2, bwd.view(b, t - 1, 2, h // 4, w // 4) / 2

    def band_mask(self, i, j, hs, ws, device):
        """1 inside, 0 on the overlap / 2 bands the authors zero (sides that have a neighbouring patch; 32 px)."""
        p = self.patch
        oy, ox = (self.overlap, self.overlap) if isinstance(self.overlap, int) else self.overlap
        hy, hx = oy // 2, ox // 2
        m = torch.ones(p, p, device=device)
        if i < hs[-1]:
            m[-hy:, :] = 0
        if j < ws[-1]:
            m[:, -hx:] = 0
        if i > hs[0]:
            m[:hy, :] = 0
        if j > ws[0]:
            m[:, :hx] = 0
        return m

    train_frozen = ("raft.",)

    def train_forward(self, lqs):
        """Training (dbis/train.py): lqs (B, T, 3, 256, 256) -> (B, T, 3, 256, 256), one patch per clip: RAFT frozen,
        its flows without gradients, then BSST (whose sparse attention is built for 256 px patches)."""
        with torch.no_grad(), torch.autocast(lqs.device.type, enabled=False):
            ff, fb = self.flows(lqs.float())
        return self.bsst(lqs, ff, fb)

    @torch.no_grad()
    def forward(self, lqs, patch_batch=None, precision=None, stage_batch=None):
        if not self.tf32:
            return self._forward(lqs, patch_batch, precision, stage_batch)
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            return self._forward(lqs, patch_batch, precision, stage_batch)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    def _forward(self, lqs, patch_batch=None, precision=None, stage_batch=None):
        pb = patch_batch or self.patch_batch
        sb = stage_batch or self.stage_batch
        precision = precision or self.precision
        b, t, c, h, w = lqs.shape
        if h % 16 or w % 16 or h < self.patch or w < self.patch:
            raise ValueError(f"BSSTNet needs H and W multiples of 16 and at least {self.patch}, got {h} x {w}")
        flows_f, flows_b = self.cached_flows(lqs) if self.flow_cache else self.flows(lqs)
        p, q = self.patch, self.patch // 4
        oy, ox = (self.overlap, self.overlap) if isinstance(self.overlap, int) else self.overlap   # (rows, columns)
        hs = list(range(0, h - p, p - oy)) + [max(0, h - p)]
        ws = list(range(0, w - p, p - ox)) + [max(0, w - p)]
        coords = [(i, j) for i in hs for j in ws]
        acc = torch.zeros(b, t, c, h, w, device=lqs.device)
        cnt = torch.zeros_like(acc)
        for k in range(0, len(coords), pb):
            chunk = coords[k:k + pb]
            x = torch.cat([lqs[..., i:i + p, j:j + p] for i, j in chunk])
            ff = torch.cat([flows_f[..., i // 4:i // 4 + q, j // 4:j // 4 + q] for i, j in chunk])
            fb = torch.cat([flows_b[..., i // 4:i // 4 + q, j // 4:j // 4 + q] for i, j in chunk])
            with torch.autocast(lqs.device.type, dtype=torch.float16, enabled=precision == "fp16"):
                y = self.bsst(x, ff, fb, stage_batch=sb)
            if precision == "fp16" and self.nan_fallback:
                bad = (~torch.isfinite(y.reshape(len(chunk), -1)).all(1)).nonzero().flatten().tolist()   # one sync per call
                if bad:
                    y = y.float()
                    sel = [m * b + r for m in bad for r in range(b)]
                    y[sel] = self.bsst(x[sel], ff[sel], fb[sel], stage_batch=sb).float()       # float32, no autocast
                    self.fallbacks += len(bad)
                    nf = ~torch.isfinite(y)
                    if bool(nf.any()):                   # still not finite in float32: the input there, not black
                        y = torch.where(nf, x.to(y.dtype), y)
                        self.input_fallbacks += int(nf.reshape(len(chunk), -1).any(1).sum())
            del x, ff, fb
            for m, (i, j) in enumerate(chunk):   # the authors' order of the sums
                mask = self.band_mask(i, j, hs, ws, lqs.device)
                acc[..., i:i + p, j:j + p].add_(y[m * b:(m + 1) * b] * mask)
                cnt[..., i:i + p, j:j + p].add_(mask)
            del y
        return acc.div_(cnt)


UNUSED = ("feat_extract.", "decoder_level1_2.", "decoder_level2.", "decoder_level2_1.", "decoder_level2_2.",
          "conv_before_upsampler.", "conv_before_input.", "upsampler.", "conv_last.")


def fold(sd, raft_sd):
    """The authors' state dicts (key "params" of BSST_gopro.pth, and raft-things.pth) -> the state dict of BSSTNet:
    prefixes bsst. and raft., without "module.", without the modules the forward never calls (UNUSED) and without the
    index buffers and the duplicate norm3 of RAFT, which are checked against what this file rebuilds / keeps."""
    def strip(d):
        return {(k[7:] if k.startswith("module.") else k): v for k, v in d.items()}
    sd, raft_sd = strip(sd), strip(raft_sd)
    out = {}
    for k, v in sd.items():
        if k.startswith(UNUSED):
            continue
        if k.endswith("relative_position_index"):
            assert torch.equal(v, relative_position_index((2, 8, 8))), k
            continue
        if k.endswith(("mask_padding", "valid_ind_rolled")):
            mp, vi = sparse_attention_constants((8, 8))
            assert torch.equal(v, mp if k.endswith("mask_padding") else vi), k
            continue
        out["bsst." + k] = v
    for k, v in raft_sd.items():
        if ".norm3." in k:
            assert torch.equal(v, raft_sd[k.replace(".norm3.", ".downsample.1.")]), k
            continue
        out["raft." + k] = v
    return {k: (v.float() if v.is_floating_point() else v).clone().contiguous() for k, v in out.items()}
