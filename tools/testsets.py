"""The compact test sets of huggingface.co/datasets/Illaitar/deblurring-test-sets: download, read, verify.

    python tools/testsets.py download [SET ...] [--dir data/testsets]   # all four sets by default (huggingface_hub)
    python tools/testsets.py verify data/testsets/gopro_test             # decode every file, check every frame's CRC

    from tools.testsets import sequences, read
    for seq in sequences("data/testsets/gopro_test"):
        for blur_name, sharp_name, blur, sharp in read("data/testsets/gopro_test", seq):
            ...                                   # blur, sharp: H x W x 3 uint8 RGB, the original PNG pixels

A set is one folder per sequence (RealBlur: one folder "all") with blur.mkv and sharp.mkv (FFV1, lossless 8-bit RGB)
and frames.tsv: index, the original relative paths of the blurry and sharp frame, the frame's height and width and the
CRC-32 of its H x W x 3 RGB bytes. Decoding needs ffmpeg: on PATH, or pip install imageio-ffmpeg.
"""
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = "Illaitar/deblurring-test-sets"
SETS = ("gopro_test", "reds_val", "realblur_j_test", "realblur_r_test")


def ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("ffmpeg is needed: install it, or pip install imageio-ffmpeg") from None
        return exe


def sequences(set_dir):
    return sorted(p.name for p in Path(set_dir).iterdir() if (p / "frames.tsv").exists())


def frames(set_dir, seq):
    """The rows of frames.tsv: dicts with index, blur, sharp, height, width, crc32_blur, crc32_sharp."""
    lines = (Path(set_dir) / seq / "frames.tsv").read_text().splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:] if ln]


def _decode(mkv, h, w, n):
    raw = subprocess.run([ffmpeg(), "-v", "error", "-i", str(mkv), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    if len(raw) != n * h * w * 3:
        raise ValueError(f"{mkv}: {len(raw) // (h * w * 3)} frames decoded, frames.tsv lists {n}")
    return np.frombuffer(raw, np.uint8).reshape(n, h, w, 3)


def read(set_dir, seq, check=True):
    """Yields (blur name, sharp name, blur, sharp) for every frame of a sequence, in order; check: the CRC of every
    frame against frames.tsv. A file's frames are coded at the largest size of its frames (RealBlur: its images
    differ by a few pixels and are padded with zeros at the bottom and right); each is cropped to its own size."""
    rows = frames(set_dir, seq)
    H, W = max(int(r["height"]) for r in rows), max(int(r["width"]) for r in rows)
    d = Path(set_dir) / seq
    blur, sharp = _decode(d / "blur.mkv", H, W, len(rows)), _decode(d / "sharp.mkv", H, W, len(rows))
    for r, b, s in zip(rows, blur, sharp):
        h, w = int(r["height"]), int(r["width"])
        b, s = np.ascontiguousarray(b[:h, :w]), np.ascontiguousarray(s[:h, :w])
        if check and (zlib.crc32(b.tobytes()) != int(r["crc32_blur"], 16)
                      or zlib.crc32(s.tobytes()) != int(r["crc32_sharp"], 16)):
            raise ValueError(f"{d}: frame {r['index']} differs from its CRC")
        yield r["blur"], r["sharp"], b, s


def download(sets=SETS, dest="data/testsets"):
    from huggingface_hub import snapshot_download
    return snapshot_download(REPO, repo_type="dataset", local_dir=dest,
                             allow_patterns=[f"{s}/*" for s in sets] + ["manifest.json", "README.md", "LICENSE.md"])


def verify(set_dir):
    n = 0
    for seq in sequences(set_dir):
        n += sum(1 for _ in read(set_dir, seq))
    print(f"{set_dir}: {n} pairs, every frame matches its CRC")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "download":
        dest = a[a.index("--dir") + 1] if "--dir" in a else "data/testsets"
        names = [x for x in a[1:] if x in SETS] or list(SETS)
        print(download(names, dest))
    elif a and a[0] == "verify":
        verify(a[1])
    else:
        sys.exit(__doc__)
