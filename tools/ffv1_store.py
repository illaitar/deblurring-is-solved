"""19.11: lossless storage of restored frames as FFV1 video, one file per sequence (instead of PNGs).

With BL_INMEM=1 and BL_FFV1=1, every frame that an adapter hands to common.write_png is scored in memory
(inmem_score.py) and appended to
    results/<method>/<dataset>/ffv1_<tag>/<seq>.mkv          FFV1 level 3, bgr0 (exact RGB, 8 bit), intra only
    results/<method>/<dataset>/ffv1_<tag>/<seq>.frames.tsv   stem, CRC-32 of the HxWx3 uint8 RGB bytes, H, W
in arrival order (FFV1 codes every frame on its own, so order costs nothing; the sidecar gives each frame its name).
A sequence is written to <seq>.mkv.part / <seq>.frames.part and renamed when it is complete (all reference frames of
the dataset received) or at process exit. When a process starts a sequence again (a rerun after a crash), older
files of that sequence are moved to ffv1_<tag>/superseded/<time>/ (never deleted). About 0.64 MB per 720p frame,
55% of PNG level 1, decoding is bit-exact (checked 19.11 on NAFNet GoPro outputs).

    python ffv1_store.py verify   results/<method>/<dataset>/ffv1_<tag>        # CRC of every frame, counts, .part
    python ffv1_store.py finalize results/<method>/<dataset>/ffv1_<tag>        # rename leftover .part after a kill
    python ffv1_store.py png2ffv1 results/<method>/<dataset> [--tag png]       # existing PNG folders -> ffv1_png/
    read_sequence(dir, seq) -> list of (stem, HxWx3 uint8), CRC-checked
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path

import numpy as np

FF = os.environ.get("FFMPEG", "ffmpeg")   # an ffmpeg build with FFV1 (the runs used the imageio-ffmpeg 7.0.2 binary)
MAX_OPEN = 8
_lock = threading.Lock()
_open: dict = {}          # (dir, seq) -> _Seq
_started: set = set()     # sequences this process has started (older files moved away once)
_expected: dict = {}      # (dataset, seq) -> number of reference frames


def _tag() -> str:
    return os.environ.get("BL_INMEM_TAG", "rows")


def store_dir(dataset_dir: Path) -> Path:
    return Path(dataset_dir) / f"ffv1_{_tag()}"


def _expected_frames(dataset: str, seq: str):
    if (dataset, seq) not in _expected:
        try:
            import inmem_score as I  # loaded by common._inmem() before the first frame arrives
            refs = I._score().references(dataset)
            for s, rs in refs.items():
                _expected[(dataset, s)] = len(rs)
        except Exception:  # noqa: BLE001
            _expected[(dataset, seq)] = None
    return _expected.get((dataset, seq))


class _Seq:
    def __init__(self, d: Path, seq: str, h: int, w: int):
        self.d, self.seq, self.h, self.w = d, seq, h, w
        d.mkdir(parents=True, exist_ok=True)
        self.part = d / f"{seq}.mkv.part"
        self.side_part = d / f"{seq}.frames.part"
        self.proc = subprocess.Popen(
            [FF, "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", "25", "-i", "-", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", "-slices", "4", "-g", "1",
             "-f", "matroska", str(self.part)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.side = open(self.side_part, "w")
        self.n, self.stems, self.last = 0, set(), time.time()

    def write(self, stem: str, arr: np.ndarray):
        if arr.shape[:2] != (self.h, self.w):
            raise ValueError(f"ffv1 {self.seq}/{stem}: frame size {arr.shape[:2]} differs from {(self.h, self.w)}")
        b = np.ascontiguousarray(arr).tobytes()
        self.proc.stdin.write(b)
        self.side.write(f"{stem}\t{zlib.crc32(b) & 0xFFFFFFFF:08x}\t{self.h}\t{self.w}\n")
        self.side.flush()
        self.n += 1
        self.stems.add(stem)
        self.last = time.time()

    def close(self):
        self.proc.stdin.close()
        rc = self.proc.wait()
        err = self.proc.stderr.read().decode(errors="ignore")[-500:]
        self.side.close()
        if rc != 0:
            raise RuntimeError(f"ffmpeg rc={rc} for {self.part}: {err}")
        os.replace(self.part, self.d / f"{self.seq}.mkv")
        os.replace(self.side_part, self.d / f"{self.seq}.frames.tsv")


def _move_old(d: Path, seq: str):
    old = [p for p in (d / f"{seq}.mkv", d / f"{seq}.frames.tsv", d / f"{seq}.mkv.part", d / f"{seq}.frames.part") if p.exists()]
    if old:
        dst = d / "superseded" / time.strftime("%Y%m%d_%H%M%S")
        dst.mkdir(parents=True, exist_ok=True)
        for p in old:
            os.replace(p, dst / p.name)


def put(path, arr: np.ndarray) -> None:
    """path = <root>/<method>/<dataset>/<seq>/<stem>.png as given to common.write_png."""
    path = Path(path)
    seq, ds_dir = path.parent.name, path.parent.parent
    d = store_dir(ds_dir)
    key = (str(d), seq)
    with _lock:
        s = _open.get(key)
        if s is None:
            if key not in _started:
                _move_old(d, seq)
                _started.add(key)
            elif (d / f"{seq}.mkv").exists():      # this process closed the sequence and now gets more frames
                k = 2
                while (d / f"{seq}.p{k}.mkv").exists():
                    k += 1
                seq_file = f"{seq}.p{k}"
                s = _Seq(d, seq_file, arr.shape[0], arr.shape[1])
            if s is None:
                s = _Seq(d, seq, arr.shape[0], arr.shape[1])
            _open[key] = s
            if len(_open) > MAX_OPEN:                 # close the least recently written sequence
                k_old = min((k for k in _open if k != key), key=lambda k: _open[k].last)
                _open.pop(k_old).close()
        s.write(path.stem, arr)
        exp = _expected_frames(ds_dir.name, seq)
        if exp is not None and len(s.stems) >= exp:
            _open.pop(key).close()


@atexit.register
def close_all():
    with _lock:
        for k in list(_open):
            try:
                _open.pop(k).close()
            except Exception as e:  # noqa: BLE001
                print(f"ffv1_store: close failed for {k}: {e}", file=sys.stderr)


# ------------------------------------------------------------------------------------------------ reading
def _decode(mkv: Path, h: int, w: int) -> np.ndarray:
    raw = subprocess.run([FF, "-nostdin", "-loglevel", "error", "-i", str(mkv), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, stdin=subprocess.DEVNULL, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def read_sequence(d, seq: str, check: bool = True):
    """All frames of a sequence (main file and continuation parts .p2, .p3, ...), CRC-checked."""
    d = Path(d)
    out = []
    files = [d / f"{seq}.mkv"] + sorted(d.glob(f"{seq}.p*.mkv"))
    for mkv in files:
        if not mkv.exists():
            continue
        side = mkv.with_name(mkv.name[:-4] + ".frames.tsv")
        rows = [ln.split("\t") for ln in side.read_text().splitlines() if ln]
        h, w = int(rows[0][2]), int(rows[0][3])
        v = _decode(mkv, h, w)
        if len(v) != len(rows):
            raise ValueError(f"{mkv}: {len(v)} frames decoded, sidecar lists {len(rows)}")
        for (stem, crc, _, _), fr in zip(rows, v):
            if check and f"{zlib.crc32(fr.tobytes()) & 0xFFFFFFFF:08x}" != crc:
                raise ValueError(f"{mkv}: CRC mismatch at {stem}")
            out.append((stem, fr))
    return out


def cmd_verify(d: Path):
    d = Path(d)
    seqs = sorted({p.name.split(".")[0] for p in d.glob("*.mkv")})
    parts = sorted(p.name for p in d.glob("*.part"))
    n = 0
    for s in seqs:
        fr = read_sequence(d, s)
        n += len(fr)
        print(f"  {s}: {len(fr)} frames ok")
    print(f"{d}: {len(seqs)} sequences, {n} frames, CRC ok" + (f"; leftover parts {parts}" if parts else ""))


def cmd_finalize(d: Path):
    for p in sorted(Path(d).glob("*.mkv.part")):
        seq = p.name[: -len(".mkv.part")]
        side = p.with_name(f"{seq}.frames.part")
        rows = [ln for ln in side.read_text().splitlines() if ln] if side.exists() else []
        h, w = (int(rows[0].split("\t")[2]), int(rows[0].split("\t")[3])) if rows else (0, 0)
        try:
            v = _decode(p, h, w)
        except Exception as e:  # noqa: BLE001
            print(f"  {seq}: not decodable ({e}), left as is")
            continue
        if len(v) < len(rows):
            rows = rows[: len(v)]
            side.write_text("\n".join(rows) + "\n")
        os.replace(p, p.with_name(f"{seq}.mkv"))
        os.replace(side, p.with_name(f"{seq}.frames.tsv"))
        print(f"  {seq}: finalized with {len(v)} frames")


def cmd_png2ffv1(ds_dir: Path, tag: str):
    """Convert <ds_dir>/<seq>/*.png into <ds_dir>/ffv1_<tag>/, then check every frame against its PNG."""
    from PIL import Image
    ds_dir = Path(ds_dir)
    os.environ["BL_INMEM_TAG"] = tag
    d = store_dir(ds_dir)
    for sd in sorted(p for p in ds_dir.iterdir() if p.is_dir() and not p.name.startswith("ffv1_")):
        pngs = sorted(sd.glob("*.png"))
        if not pngs:
            continue
        first = np.asarray(Image.open(pngs[0]).convert("RGB"))
        s = _Seq(d, sd.name, *first.shape[:2])
        for p in pngs:
            s.write(p.stem, np.asarray(Image.open(p).convert("RGB")))
        s.close()
        back = dict(read_sequence(d, sd.name))
        bad = [p.stem for p in pngs if not np.array_equal(back[p.stem], np.asarray(Image.open(p).convert("RGB")))]
        print(f"  {sd.name}: {len(pngs)} PNG -> ffv1, {'identical' if not bad else f'{len(bad)} DIFFERENT'}", flush=True)
        if bad:
            raise SystemExit(1)


if __name__ == "__main__":
    cmd, target = sys.argv[1], Path(sys.argv[2])
    if cmd == "verify":
        cmd_verify(target)
    elif cmd == "finalize":
        cmd_finalize(target)
    elif cmd == "png2ffv1":
        cmd_png2ffv1(target, sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "png")
