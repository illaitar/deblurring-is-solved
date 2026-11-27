"""dbis.data: training pairs of blurry and sharp frames, read as fast as the machine allows.

    python -m dbis.data find DIR                    # what the layout search finds in DIR
    python -m dbis.data cache DIR CACHE [--jobs N]  # every frame once to an uncompressed .npy (a crop is then read, not decoded)
    python -m dbis.data bench DIR [--cache CACHE]   # crops per second of every reader on this machine

Layout. A blurry frame is an image whose path has one of the blurry names of SIDES (blur, Blur, input_noise, input),
its sharp frame the same path with that name replaced (sharp, Sharp, target, GT, gt). The frames of one folder are one
sequence, in name order. So GoPro, REDS, BSD, DVD, RealBlur, GS-Blur and MIORe work as downloaded.

Readers. A training sample needs a crop of a few frames; decoding a whole PNG for it costs 20-80 ms at 1280x720 to
1920x1080. The readers: "pil" (Pillow), "cv2" (OpenCV, if installed), "tv" (torchvision.io), and "cache": the frames
decoded once into uint8 .npy files under CACHE (the same relative paths), opened as memory maps, so a crop reads only
its rows (and repeated reads come from the page cache, i.e. from RAM once the set fits there), and "ram": every frame
read once into memory before training (Reader.preload), shared by the data workers. best_reader() picks the fastest
one available.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

IMAGES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
SIDES = (("blur", "sharp"), ("Blur", "Sharp"), ("input_noise", "target"), ("input", "target"), ("input", "GT"),
         ("blur", "gt"))


# ---------------------------------------------------------------------------------------------------- layout
def find_pairs(root):
    """{sequence: [(blurry path, sharp path), ...] in name order} of every pair under root (see SIDES)."""
    root, seqs = Path(root), {}
    for dirpath, _, files in os.walk(root, followlinks=True):
        for f in sorted(files):
            if not f.lower().endswith(IMAGES):
                continue
            p = Path(dirpath) / f
            rel = p.relative_to(root).as_posix()
            for a, b in SIDES:
                if a in rel and (root / rel.replace(a, b)).is_file():
                    seqs.setdefault(str(Path(rel).parent), []).append((p, root / rel.replace(a, b)))
                    break
    if not seqs:
        raise FileNotFoundError(f"no pairs of blurry and sharp frames under {root} (names: {SIDES})")
    return {k: sorted(v) for k, v in sorted(seqs.items())}


def names_in(path):
    return {s.strip() for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()}


def select(seqs, keep=None, drop=None):
    """Frames or sequences named in keep (if given) and not in drop: a frame matches by its file name or stem."""
    def hit(names, seq, b):
        return seq in names or b.name in names or b.stem in names
    out = {}
    for s, v in seqs.items():
        v = [(b, g) for b, g in v if (keep is None or hit(keep, s, b)) and (drop is None or not hit(drop, s, b))]
        if v:
            out[s] = v
    return out


# ---------------------------------------------------------------------------------------------------- readers
def read_pil(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def read_cv2(path):
    import cv2
    a = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if a is None:
        raise OSError(f"cv2 cannot read {path}")
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def read_tv(path):
    from torchvision.io import ImageReadMode, decode_image, read_file
    return decode_image(read_file(str(path)), mode=ImageReadMode.RGB).permute(1, 2, 0).numpy()


def available():
    """The readers this machine has (cache is added when a cache folder is given)."""
    out = ["pil"]
    for mod, name in (("cv2", "cv2"), ("torchvision.io", "tv")):
        try:
            __import__(mod)
            out.append(name)
        except ImportError:
            pass
    return out


class Reader:
    """read(path) -> uint8 (H, W, 3); with a cache, a memory map of the cached frame (slicing reads only the crop),
    kept open (keep_open) so that a crop costs no file open; "ram": every frame held in memory (preload())."""

    def __init__(self, kind="pil", root=None, cache=None, keep_open=True):
        self.kind, self.root, self.cache = kind, Path(root) if root else None, Path(cache) if cache else None
        self.fn = {"pil": read_pil, "cv2": read_cv2, "tv": read_tv}.get(kind)
        self.keep_open, self.maps, self.frames, self.max_open = keep_open, {}, None, 0
        if kind == "cache" and (self.cache is None or self.root is None):
            raise ValueError("the cache reader needs root and cache")
        if kind == "cache" and keep_open:              # every open map holds a file descriptor: allow the hard limit
            try:
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                if soft < hard:
                    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
                self.max_open = resource.getrlimit(resource.RLIMIT_NOFILE)[0] - 256
            except (ImportError, ValueError, OSError):
                self.max_open = 512
        if kind == "cv2":
            import cv2
            cv2.setNumThreads(1)                       # one thread per data worker

    def _map(self, path):
        return np.load(self.cache / (Path(path).relative_to(self.root).as_posix() + ".npy"), mmap_mode="r")

    def preload(self, paths, jobs=16):
        """kind "ram": the frames of paths read once into memory (from the cache when it has them, else decoded), so
        data workers forked afterwards share them. Returns the GiB held."""
        paths = [str(p) for p in paths]
        if self.cache and self.root and all((self.cache / (Path(p).relative_to(self.root).as_posix() + ".npy")).exists()
                                            for p in paths):
            one = lambda p: np.array(self._map(p))     # noqa: E731
        else:
            one = read_pil
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(jobs) as ex:
            self.frames = dict(zip(paths, ex.map(one, paths)))
        return sum(a.nbytes for a in self.frames.values()) / 2 ** 30

    def __call__(self, path):
        if self.kind == "ram":
            return self.frames[str(path)]
        if self.kind == "cache":
            if not self.keep_open:
                return self._map(path)
            m = self.maps.get(path)
            if m is None:
                m = self._map(path)
                if len(self.maps) < self.max_open:
                    self.maps[path] = m
            return m
        return self.fn(path)


def _cache_one(job):
    src, dst = job
    if dst.exists():
        return 0
    a = read_pil(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp.npy")
    np.save(tmp, a)
    os.replace(tmp, dst)
    return a.nbytes


def build_cache(root, cache, jobs=None, seqs=None):
    """Every frame of the pairs under root decoded once into cache/<relative path>.npy (frames already there are
    kept). Returns the bytes written."""
    root, cache = Path(root), Path(cache)
    seqs = seqs or find_pairs(root)
    todo = [(p, cache / (p.relative_to(root).as_posix() + ".npy")) for v in seqs.values() for pair in v for p in pair]
    t0, done = time.time(), 0
    with ProcessPoolExecutor(jobs or os.cpu_count()) as ex:
        for i, n in enumerate(ex.map(_cache_one, todo, chunksize=16)):
            done += n
            if (i + 1) % 2000 == 0:
                print(f"cache: {i + 1} of {len(todo)} frames, {done / 2 ** 30:.1f} GiB, {time.time() - t0:.0f} s", flush=True)
    return done


def cache_ready(root, cache, seqs):
    root, cache = Path(root), Path(cache)
    return all((cache / (p.relative_to(root).as_posix() + ".npy")).exists() for v in seqs.values() for pr in v for p in pr)


def crop_speed(reader, pairs, crop, n=64, seed=0):
    """Crops (of both frames of a pair) per second of one process."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for i in range(n):
        b, g = pairs[rng.integers(len(pairs))]
        x, y = reader(b), reader(g)
        h, w = x.shape[:2]
        r, c = rng.integers(max(1, h - crop + 1)), rng.integers(max(1, w - crop + 1))
        np.ascontiguousarray(x[r:r + crop, c:c + crop]), np.ascontiguousarray(y[r:r + crop, c:c + crop])
    return n / (time.perf_counter() - t0)


def best_reader(root, seqs, crop, cache=None, n=48):
    """{reader: crops per second} on this machine and the fastest of them."""
    pairs = [p for v in seqs.values() for p in v]
    kinds = available() + (["cache"] if cache and cache_ready(root, cache, seqs) else [])
    speed = {}
    for k in kinds:
        rd = Reader(k, root, cache)
        crop_speed(rd, pairs, crop, n=4)               # warm the imports
        speed[k] = round(crop_speed(rd, pairs, crop, n=n), 1)
    return speed, max(speed, key=speed.get)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dbis.data", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find")
    f.add_argument("data")
    c = sub.add_parser("cache")
    c.add_argument("data")
    c.add_argument("cache")
    c.add_argument("--jobs", type=int)
    b = sub.add_parser("bench")
    b.add_argument("data")
    b.add_argument("--cache")
    b.add_argument("--crop", type=int, default=256)
    a = ap.parse_args(argv)
    seqs = find_pairs(a.data)
    n = sum(len(v) for v in seqs.values())
    if a.cmd == "find":
        print(f"{n} pairs in {len(seqs)} sequences under {a.data}")
        for s, v in list(seqs.items())[:5]:
            print(f"  {s}: {len(v)} pairs, first {v[0][0]} + {v[0][1]}")
    elif a.cmd == "cache":
        print(f"{n} pairs: caching {2 * n} frames into {a.cache}")
        print(f"{build_cache(a.data, a.cache, a.jobs, seqs) / 2 ** 30:.1f} GiB written")
    else:
        speed, best = best_reader(a.data, seqs, a.crop, a.cache)
        print(f"crops of {a.crop} px per second, one process: {speed}; fastest: {best}")


if __name__ == "__main__":
    main()
