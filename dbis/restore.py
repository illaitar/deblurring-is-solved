"""net.restore(frames, **options): the one inference call of every dbis model.

    import dbis
    net = dbis.load("dbisf-BSSTNet")             # any model of dbis.MODELS
    out = net.restore(frames)                    # the model's own settings (net.defaults)
    out = net.restore(frames, tile=512, tile_overlap=32, batch=4, precision="fp16", compile=True, out="uint8")

frames   a clip [T, 3, H, W], a batch of clips [B, T, 3, H, W] or one image [3, H, W]; uint8 (0 to 255) or float in
         [0, 1]; on any device (moved to the model's device one piece at a time, so a long clip can stay on the CPU);
         any H and W (padded inside where a network needs it, the padding cropped from the output)
returns  the same shape on the device of `frames`: float32 in [0, 1], not clamped, or uint8 with out="uint8" (rounded as
         the benchmark does: NaN to 0, clamped, round half up)

Options (None keeps the model's own setting; net.defaults lists them):
  tile          size of the tiles a frame is cut into: int or (h, w). Image models: None = the whole frame. BSSTNet runs
                on 256 px patches only (its sparse attention is built for them)
  tile_overlap  pixels shared by neighbouring tiles: int or (rows, columns). The outputs are averaged where tiles
                overlap (BSSTNet: the authors' blending, which first drops overlap / 2 px at every inner tile border)
  window        video models: frames per window, laid out as the benchmark runs the authors' code (net._plan):
                BSSTNet: windows overlap by 4 frames and every frame is taken from a window where it is at least 2
                frames from the ends (so at least 6 frames); DSTNet+: clips that do not overlap, the last one moved back
                to end at the last frame. Image models: no
  batch         tiles per network call (image models without tiles: frames per call): memory against speed
  precision     "fp32" or "fp16" (autocast)
  compile       False, True or a torch.compile mode: compile on the first call (BSSTNet: its heavy blocks one by one,
                the whole forward does not compile)
  out           "float" or "uint8"
"""
import torch
import torch.nn.functional as F

OPTIONS = ("tile", "tile_overlap", "window", "batch", "precision", "compile", "out")


def to_uint8(y):
    return (torch.nan_to_num(y, 0.0).clamp(0, 1) * 255 + 0.5).floor().to(torch.uint8)


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


class Restorer:
    """Mixed into every dbis network: restore() for image and video models alike."""
    kind = "image"

    # ---------------------------------------------------------------- what a model sets
    def _defaults(self):
        return {"tile": None, "tile_overlap": 32, "window": None, "batch": 1, "precision": "fp32"}

    def _check(self, o):
        pass

    @property
    def defaults(self):
        return dict(self._defaults(), compile=False, out="float")

    # ---------------------------------------------------------------- the call
    @torch.no_grad()
    def restore(self, frames, *, tile=None, tile_overlap=None, window=None, batch=None, precision=None, compile=False,
                out="float"):
        given = {"tile": tile, "tile_overlap": tile_overlap, "window": window, "batch": batch, "precision": precision}
        o = self._defaults()
        o.update({k: v for k, v in given.items() if v is not None})
        if o["precision"] not in ("fp32", "fp16"):
            raise ValueError(f"precision: 'fp32' or 'fp16', not {o['precision']!r}")
        if out not in ("float", "uint8"):
            raise ValueError(f"out: 'float' or 'uint8', not {out!r}")
        if self.kind == "image" and window is not None:
            raise ValueError(f"{type(self).__name__} is an image model: window does not apply")
        if self.kind == "video" and (not isinstance(o["window"], int) or o["window"] < 1):
            raise ValueError(f"window: a positive int, not {o['window']!r}")
        if o["tile"] is not None:
            th, tw = _pair(o["tile"])
            oy, ox = _pair(o["tile_overlap"])
            if not (0 <= oy < th and 0 <= ox < tw):
                raise ValueError(f"tile_overlap {o['tile_overlap']} must be smaller than the tile {o['tile']}")
        if not isinstance(o["batch"], int) or o["batch"] < 1:
            raise ValueError(f"batch: a positive int, not {o['batch']!r}")
        self._check(o)
        if compile:
            self._compile("default" if compile is True else compile)

        shape = frames.shape
        if frames.dim() == 3:
            x = frames[None, None]
        elif frames.dim() == 4:
            x = frames[None]
        elif frames.dim() == 5:
            x = frames
        else:
            raise ValueError(f"frames: [3, H, W], [T, 3, H, W] or [B, T, 3, H, W], got {tuple(shape)}")
        if x.shape[2] != 3:
            raise ValueError(f"frames need 3 channels (RGB) in dimension -3, got {tuple(shape)}")
        y = torch.empty(x.shape, dtype=torch.uint8 if out == "uint8" else torch.float32, device=frames.device)
        dev = next(self.parameters()).device

        def prep(t):                                   # a piece of the clip -> float in [0, 1] on the model's device
            t = t.to(dev, non_blocking=True)
            return t.float().div_(255) if t.dtype == torch.uint8 else t.float()

        def put(dst, t):
            dst.copy_(to_uint8(t) if out == "uint8" else t)

        if self.kind == "image":
            self._restore_images(x, y, o, prep, put)
        else:
            self._restore_video(x, y, o, prep, put)
        return y.view(shape)

    # ---------------------------------------------------------------- image models
    def _run_images(self, t, o):
        """t (n, 3, h, w) on the device -> float32."""
        fwd = getattr(self, "_compiled", None) or self.forward
        with torch.autocast(t.device.type, dtype=torch.float16, enabled=o["precision"] == "fp16"):
            return fwd(t).float()

    def _restore_images(self, x, y, o, prep, put):
        b, t = x.shape[:2]
        flat_x, flat_y = x.reshape(b * t, *x.shape[2:]), y.view(b * t, *y.shape[2:])
        if o["tile"] is None:
            for s in range(0, b * t, o["batch"]):
                put(flat_y[s:s + o["batch"]], self._run_images(prep(flat_x[s:s + o["batch"]]), o).to(y.device))
            return
        for i in range(b * t):
            put(flat_y[i], self._tiled(prep(flat_x[i:i + 1]), o)[0].to(y.device))

    def _tiled(self, x, o):
        """One frame (1, 3, H, W) in tiles of o['tile'] that overlap by o['tile_overlap'], the outputs averaged."""
        _, _, h, w = x.shape
        th, tw = _pair(o["tile"])
        oy, ox = _pair(o["tile_overlap"])
        th, tw = min(th, h), min(tw, w)
        ys = list(range(0, h - th, max(1, th - oy))) + [h - th]
        xs = list(range(0, w - tw, max(1, tw - ox))) + [w - tw]
        boxes = [(i, j) for i in ys for j in xs]
        acc = torch.zeros(1, 3, h, w, device=x.device)
        cnt = torch.zeros(1, 1, h, w, device=x.device)
        for k in range(0, len(boxes), o["batch"]):
            chunk = boxes[k:k + o["batch"]]
            out = self._run_images(torch.cat([x[..., i:i + th, j:j + tw] for i, j in chunk]), o)
            for m, (i, j) in enumerate(chunk):
                acc[..., i:i + th, j:j + tw] += out[m:m + 1]
                cnt[..., i:i + th, j:j + tw] += 1
        return acc / cnt

    def _compile(self, mode):
        if getattr(self, "_compiled", None) is None:
            object.__setattr__(self, "_compiled", torch.compile(self.forward, mode=mode, dynamic=False))

    # ---------------------------------------------------------------- video models
    def _window(self, x, o):
        """One window (B, T, 3, H, W) on the device -> float32; set by a video model."""
        raise NotImplementedError

    def _pad(self, x):
        """Padding a video model needs: (multiple, minimum) of H and W; (1, 1) for none."""
        return 1, 1

    min_frames = 4                  # video models: the fewest frames a window may have

    def _plan(self, t, n):
        """The windows of a clip of t frames: [(start, end, keep_from, keep_to)]. Default (BSSTNet, as
        runs/baselines/bsstnet/run_bsst.py window_plan): n-frame windows that overlap by 4, every frame taken from a
        window where it is at least 2 frames from the ends, except at the ends of the clip."""
        if n < 6:
            raise ValueError(f"window: at least 6 frames (windows overlap by 4), not {n}")
        starts = [0] if t <= n else list(range(0, t - n, n - 4)) + [t - n]
        plan, covered = [], 0
        for k, s in enumerate(starts):
            lo = 0 if k == 0 else max(s + 2, covered)
            hi = t if k == len(starts) - 1 else s + n - 2
            plan.append((s, min(s + n, t), lo, hi))
            covered = hi
        return plan

    def _restore_video(self, x, y, o, prep, put):
        b, t, _, h, w = x.shape
        if t < self.min_frames:
            raise ValueError(f"{type(self).__name__} needs at least {self.min_frames} frames, got {t}")
        mult, least = self._pad(x)
        ph, pw = max(least, -(-h // mult) * mult) - h, max(least, -(-w // mult) * mult) - w
        for s, e, lo, hi in self._plan(t, o["window"]):
            xw = prep(x[:, s:e])
            if ph or pw:
                xw = F.pad(xw.flatten(0, 1), (0, pw, 0, ph), mode="replicate").view(b, e - s, 3, h + ph, w + pw)
            out = self._window(xw, o)[..., :h, :w]
            put(y[:, lo:hi], out[:, lo - s:hi - s].to(y.device))
