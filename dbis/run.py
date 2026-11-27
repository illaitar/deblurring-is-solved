"""python -m dbis.run MODEL INPUT OUTPUT: restore an image, a folder of frames or a video file with one dbis model.

    python -m dbis.run dbisf-NAFNet-64 blurry.png sharp.png
    python -m dbis.run dbisf-DSTNetPlus-L frames/ sharp_frames/       # a folder: one clip, frames in name order
    python -m dbis.run dbisf-Turtle clip.mp4 sharp.mp4                 # a video: read and written with ffmpeg

INPUT   an image file, a folder of images (one clip, in name order) or a video file (any format ffmpeg reads; ffmpeg
        and ffprobe must be on PATH)
OUTPUT  an image file, a folder (PNG frames with the names of the input frames) or a video file (H.264, --crf, the
        frame rate of the input)
The weight file is fetched from Hugging Face on first use (dbis.load). --tile, --tile-overlap, --window, --batch,
--precision and --compile are the options of net.restore (dbis/restore.py); without them the model's own settings.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from . import MODELS, load

IMAGES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def read_image(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def ffmpeg(tool):
    exe = shutil.which(tool)
    if exe is None:
        raise SystemExit(f"{tool} is not on PATH: install ffmpeg to read or write video files")
    return exe


def read_video(path):
    """(frames (T, H, W, 3) uint8, frame rate as a string) of a video file."""
    probe = json.loads(subprocess.run([ffmpeg("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=width,height,r_frame_rate", "-of", "json", str(path)],
                                      check=True, capture_output=True).stdout)["streams"][0]
    w, h = probe["width"], probe["height"]
    raw = subprocess.run([ffmpeg("ffmpeg"), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), probe["r_frame_rate"]


def write_video(path, frames, rate, crf):
    t, h, w, _ = frames.shape
    subprocess.run([ffmpeg("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                    "-r", rate, "-i", "-", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)],
                   input=np.ascontiguousarray(frames).tobytes(), check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dbis.run", description=__doc__.split("\n\n")[0])
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--weights-dir", help="where the weight file is kept (default: see dbis.WEIGHTS)")
    ap.add_argument("--tile", type=int)
    ap.add_argument("--tile-overlap", type=int)
    ap.add_argument("--window", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--precision", choices=("fp32", "fp16"))
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--crf", type=int, default=12, help="H.264 quality of a video output (lower is better)")
    a = ap.parse_args(argv)

    src, dst = Path(a.input), Path(a.output)
    names, rate = None, None
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGES)
        if not files:
            raise SystemExit(f"no images in {src}")
        frames, names = np.stack([read_image(p) for p in files]), [p.stem for p in files]
    elif src.suffix.lower() in IMAGES:
        frames = read_image(src)[None]
    else:
        frames, rate = read_video(src)
    net = load(a.model, weights_dir=a.weights_dir, device=a.device)
    opts = {k: v for k, v in (("tile", a.tile), ("tile_overlap", a.tile_overlap), ("window", a.window),
                              ("batch", a.batch), ("precision", a.precision)) if v is not None}
    clip = torch.from_numpy(frames).permute(0, 3, 1, 2)                 # (T, 3, H, W) uint8, kept on the CPU
    out = net.restore(clip, compile=a.compile, out="uint8", **opts).permute(0, 2, 3, 1).numpy()
    print(f"{a.model}: {len(out)} frame(s) of {out.shape[2]} x {out.shape[1]} -> {dst}")
    from PIL import Image
    if dst.suffix.lower() in IMAGES:
        if len(out) != 1:
            raise SystemExit(f"{len(out)} frames: give a folder or a video file as OUTPUT")
        Image.fromarray(out[0]).save(dst)
    elif dst.suffix == "" or dst.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(out):
            Image.fromarray(f).save(dst / f"{names[i] if names else f'{i:06d}'}.png")
    else:
        write_video(dst, out, rate or "30", a.crf)


if __name__ == "__main__":
    main()
