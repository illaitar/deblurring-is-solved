"""wpack: fast packing of neural network weights, exact or with a bounded rounding error.

    python wpack.py pack   IN OUT.wpk [--tier lossless|f16|q] [--s 128] [--group NAME] [--threads N]
    python wpack.py unpack IN.wpk OUT.safetensors [--threads N]
    python wpack.py info   IN.wpk
    python wpack.py verify IN.wpk ORIGINAL [--group NAME]

IN is a .safetensors file or a PyTorch checkpoint (loaded with weights_only=True, so no code in it is executed). From
a checkpoint only the model tensors are kept: the largest top-level group that is not an optimizer, a scheduler or a
discriminator, or the group given with --group.

Tiers
  lossless   every tensor comes back bit for bit.
  f16        a float32 tensor is stored as float16 when all its values are finite, inside the float16 range and the
             relative L2 error of the cast is at most 1e-3. Other tensors are stored exactly. Unpacks to float32.
  q          a float32 tensor with 2+ dimensions and at least 1024 values is rounded to the step std(tensor) / s
             (so |error| <= step / 2 for every weight). Other tensors are stored exactly. Unpacks to float32.

Coding (see the notes in the README next to this file for the measurements behind each choice)
  float32 exact    bits rotated left by one, so that the eight exponent bits fill one byte. That byte plane goes through
                   a Huffman coder, a mantissa plane is stored raw unless a probe shows that it shrinks.
  float16          two byte planes, the high one (sign, exponent, two mantissa bits) through the Huffman coder.
  quantized        q = 16 * b + r. The four low bits r are packed raw (uniform inside a bucket no wider than
                   std / 4), the bucket index b (one byte) goes through the Huffman coder, rare values outside one
                   byte are listed apart.
The Huffman coder is zstd with match finding switched off in practice (fast strategy, min_match 7): on these streams it
stays within about 0.08 bit per weight of the order-0 entropy and decodes above 1 GB/s per core.

File: b"WPK1", uint64 little-endian header length, UTF-8 JSON header, then the blobs. Every tensor is coded on its own
(random access, threads), with the CRC32 of its unpacked bytes in the header.
Dependencies: numpy and zstandard. torch and safetensors are needed only to read checkpoints and to write safetensors.
"""
from __future__ import annotations

import json
import struct
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zstandard as zstd

MAGIC = b"WPK1"
VERSION = 1
PROBE = 1 << 16          # bytes of a plane that decide whether it is worth coding
MIN_Q = 1024             # smallest tensor that is quantized
_HUF = zstd.ZstdCompressionParameters.from_level(3, min_match=7, strategy=zstd.STRATEGY_FAST)
BAD_GROUPS = ("optim", "sched", "discr", "net_d", "netd", "lr_", "scaler")


# ------------------------------------------------------------------------------------------------ byte streams
def _huf(buf) -> bytes:
    return zstd.ZstdCompressor(compression_params=_HUF).compress(buf)


def _unz(blob, n: int) -> bytes:
    return zstd.ZstdDecompressor().decompress(blob, max_output_size=n)


def _plane(pl: np.ndarray, force: bool = False):
    """(encoding, bytes) for one byte plane: 'h' Huffman, 'r' raw, 'c' constant (one byte)."""
    n = pl.size
    if n and bool((pl == pl[0]).all()):
        return "c", bytes(pl[:1])
    if not force and n > 4 * PROBE:
        mid = (n - PROBE) // 2
        if len(_huf(pl[mid:mid + PROBE])) > 0.985 * PROBE:
            return "r", pl.tobytes()
    c = _huf(pl)
    return ("h", c) if len(c) < 0.985 * n else ("r", pl.tobytes())


def _unplane(enc: str, blob, n: int) -> np.ndarray:
    if enc == "c":
        return np.full(n, blob[0], np.uint8)
    return np.frombuffer(_unz(blob, n) if enc == "h" else blob, np.uint8, n)


# ------------------------------------------------------------------------------------------------ tensor codecs
def _enc_f32(a: np.ndarray):
    u = a.reshape(-1).view(np.uint32)
    r = ((u << np.uint32(1)) | (u >> np.uint32(31))).view(np.uint8).reshape(-1, 4)   # byte 3 = exponent
    parts = [_plane(np.ascontiguousarray(r[:, i]), force=(i == 3)) for i in range(4)]
    return {"codec": "f32x"}, parts


def _dec_f32(t, blobs, n):
    r = np.empty((n, 4), np.uint8)
    for i in range(4):
        r[:, i] = _unplane(t["enc"][i], blobs[i], n)
    v = r.reshape(-1).view(np.uint32)
    return ((v >> np.uint32(1)) | (v << np.uint32(31))).view(np.float32)


def _enc_f16(h: np.ndarray, cast: bool):
    b = h.reshape(-1).view(np.uint8).reshape(-1, 2)
    parts = [_plane(np.ascontiguousarray(b[:, 0])), _plane(np.ascontiguousarray(b[:, 1]), force=True)]
    return {"codec": "f16x", "cast": cast}, parts


def _dec_f16(t, blobs, n):
    b = np.empty((n, 2), np.uint8)
    for i in range(2):
        b[:, i] = _unplane(t["enc"][i], blobs[i], n)
    h = b.reshape(-1).view(np.float16)
    return h.astype(np.float32) if t.get("cast") else h


def _enc_q(a: np.ndarray, s: float):
    w = a.reshape(-1)
    step = np.float32(float(w.std(dtype=np.float64)) / s)
    q = np.rint(w / step).astype(np.int32)
    b = q >> 4
    esc = np.flatnonzero((b < -127) | (b > 127)).astype(np.uint32)
    val = q[esc]
    b8 = b.astype(np.int8)
    b8[esc] = -128
    r = (q & 15).astype(np.uint8)
    if r.size % 2:
        r = np.append(r, np.uint8(0))
    nib = (r[0::2] << 4) | r[1::2]
    parts = [("h", _huf(b8)), ("r", nib.tobytes()), ("r", esc.tobytes()), ("r", val.tobytes())]
    return {"codec": "q4", "step": float(step), "step_bits": int(step.view(np.uint32)), "s": s, "esc": int(esc.size)}, parts


def _dec_q(t, blobs, n):
    b8 = np.frombuffer(_unz(blobs[0], n), np.int8, n)
    nib = np.frombuffer(blobs[1], np.uint8)
    q = b8.astype(np.int32)
    q <<= 4
    q[0::2] |= (nib >> 4)[: (n + 1) // 2]
    q[1::2] |= (nib & 15)[: n // 2]
    if t["esc"]:
        q[np.frombuffer(blobs[2], np.uint32)] = np.frombuffer(blobs[3], np.int32)
    return q.astype(np.float32) * np.uint32(t["step_bits"]).view(np.float32)


def _q_ok(w: np.ndarray, s: float) -> bool:
    """A tensor is quantized only when it is finite, not constant, and its largest value stays far inside int32."""
    if not bool(np.isfinite(w).all()):
        return False
    sd = float(w.std(dtype=np.float64))
    if not sd > 0.0 or float(np.float32(sd / s)) < float(np.finfo(np.float32).tiny) * 2 ** 24:
        return False                       # a step near the subnormal range would lose precision: keep it exact
    return float(np.abs(w).max()) * s / sd < 2.0 ** 30


def _f16_ok(a: np.ndarray) -> bool:
    w = a.reshape(-1)
    if not bool(np.isfinite(w).all()):
        return False
    m = float(np.abs(w).max()) if w.size else 0.0
    if m > 65504.0:
        return False
    den = float((w.astype(np.float64) ** 2).sum())
    if den == 0.0:
        return True
    err = float(((w.astype(np.float16).astype(np.float32) - w).astype(np.float64) ** 2).sum())
    return err <= 1e-6 * den


def _encode_tensor(name: str, a: np.ndarray, tier: str, s: float):
    a = np.asarray(a, order="C")          # not ascontiguousarray: it turns a 0-d tensor into shape (1,)
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    t = {"name": name, "shape": list(a.shape), "dtype": a.dtype.str, "n": int(a.size)}
    if a.dtype == np.float32 and a.size:
        w = a.reshape(-1)
        if tier == "q" and a.ndim >= 2 and a.size >= MIN_Q and _q_ok(w, s):
            info, parts = _enc_q(a, s)
        elif tier == "f16" and _f16_ok(a):
            info, parts = _enc_f16(a.astype(np.float16), cast=True)
        else:
            info, parts = _enc_f32(a)
    elif a.dtype == np.float16 and a.size:
        info, parts = _enc_f16(a, cast=False)
    else:
        raw = a.tobytes()
        c = zstd.ZstdCompressor(level=3).compress(raw) if len(raw) > 64 else raw
        info, parts = ({"codec": "zst"}, [("h", c)]) if len(c) < len(raw) else ({"codec": "raw"}, [("r", raw)])
    t.update(info)
    t["enc"] = "".join(e for e, _ in parts)
    t["len"] = [len(b) for _, b in parts]
    return t, [b for _, b in parts]


def _decode_tensor(t: dict, blobs) -> np.ndarray:
    n, codec = t["n"], t["codec"]
    if codec == "f32x":
        a = _dec_f32(t, blobs, n)
    elif codec == "f16x":
        a = _dec_f16(t, blobs, n)
    elif codec == "q4":
        a = _dec_q(t, blobs, n)
    else:
        dt = np.dtype(t["dtype"])
        raw = _unz(blobs[0], n * dt.itemsize) if codec == "zst" else blobs[0]
        a = np.frombuffer(raw, dt, n)
    return a.reshape(t["shape"])


# ------------------------------------------------------------------------------------------------ files
def pack(tensors: dict, path, tier: str = "lossless", s: float = 128.0, threads: int = 1, meta: dict | None = None,
         selfcheck: bool = True) -> dict:
    """Write {name: numpy array} to `path`. Returns the header (with sizes). With selfcheck every tensor is decoded
    again before it is written, and an exact tensor that does not come back bit for bit raises."""
    if tier not in ("lossless", "f16", "q"):
        raise ValueError("tier must be lossless, f16 or q")
    if tier == "q" and not 8 <= s <= 1024:
        raise ValueError("s must be between 8 and 1024")
    items = list(tensors.items())

    def one(kv):
        a = np.asarray(kv[1], order="C")
        t, blobs = _encode_tensor(kv[0], a, tier, s)
        lossy = t["codec"] == "q4" or bool(t.get("cast"))
        if lossy or selfcheck:
            back = _decode_tensor(t, blobs).tobytes()
            if not lossy and back != a.tobytes():
                raise RuntimeError(f"{kv[0]}: the exact codec {t['codec']} did not return the input")
            t["crc"] = zlib.crc32(back) & 0xFFFFFFFF          # of what unpack will return
        else:
            t["crc"] = zlib.crc32(a.tobytes()) & 0xFFFFFFFF
        return t, blobs

    if threads > 1:
        with ThreadPoolExecutor(threads) as ex:
            done = list(ex.map(one, items))
    else:
        done = [one(kv) for kv in items]
    off, tens = 0, []
    for t, blobs in done:
        t["off"] = off
        off += sum(len(b) for b in blobs)
        tens.append(t)
    head = {"format": "wpack", "version": VERSION, "tier": tier, "s": s if tier == "q" else None, "tensors": tens, "meta": meta or {}}
    hj = json.dumps(head, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(MAGIC + struct.pack("<Q", len(hj)) + hj)
        for _, blobs in done:
            for b in blobs:
                fh.write(b)
    head["file_bytes"] = len(MAGIC) + 8 + len(hj) + off
    return head


def read_header(path) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        if fh.read(4) != MAGIC:
            raise ValueError(f"{path} is not a wpack file")
        (n,) = struct.unpack("<Q", fh.read(8))
        head = json.loads(fh.read(n).decode("utf-8"))
    if head.get("version", 0) > VERSION:
        raise ValueError(f"{path}: version {head['version']} is newer than this reader ({VERSION})")
    return head, 12 + n


def unpack(path, names=None, threads: int = 1, check: bool = True) -> dict:
    """{name: numpy array} of the tensors of `path` (all, or the listed names). Quantized and float16-cast tensors
    come back as float32."""
    head, base = read_header(path)
    want = [t for t in head["tensors"] if names is None or t["name"] in set(names)]
    buf = np.memmap(path, np.uint8, "r")

    def one(t):
        o, blobs = base + t["off"], []
        for ln in t["len"]:
            blobs.append(buf[o:o + ln])
            o += ln
        a = _decode_tensor(t, blobs)
        if check and (zlib.crc32(a.tobytes()) & 0xFFFFFFFF) != t["crc"]:
            raise ValueError(f"{path}: checksum of {t['name']} does not match")
        return t["name"], a

    if threads > 1:
        with ThreadPoolExecutor(threads) as ex:
            return dict(ex.map(one, want))
    return dict(one(t) for t in want)


# ------------------------------------------------------------------------------------------------ checkpoints
def load_model_tensors(path, group: str | None = None) -> tuple[dict, dict]:
    """({name: numpy array}, info) of the model tensors of a .safetensors file or a PyTorch checkpoint."""
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors.numpy import load_file
        return load_file(str(p)), {"source": p.name, "group": None, "dropped": []}
    import argparse

    import torch
    try:        # many training scripts save their argparse options next to the weights: a plain container, safe to allow
        torch.serialization.add_safe_globals([argparse.Namespace])
    except Exception:
        pass
    obj = torch.load(str(p), map_location="cpu", weights_only=True)
    flat = []

    def walk(o, pre, depth=0):
        if torch.is_tensor(o):
            flat.append((pre, o))
        elif isinstance(o, dict) and depth < 8:
            for k, v in o.items():
                walk(v, f"{pre}/{k}" if pre else str(k), depth + 1)

    walk(obj, "")
    nested = isinstance(obj, dict) and any(isinstance(v, dict) for v in obj.values())
    grp = (lambda nm: nm.split("/")[0]) if nested else (lambda nm: "")
    size = {}
    for nm, t in flat:
        size[grp(nm)] = size.get(grp(nm), 0) + t.numel() * t.element_size()
    if group is None:
        cand = {g: b for g, b in size.items() if not (any(x in g.lower() for x in BAD_GROUPS) or g.lower().startswith("opt"))} or size
        group = max(cand, key=cand.get)
    if group not in size:
        raise ValueError(f"{p.name}: no group {group!r}, groups: {sorted(size)}")
    out = {}
    for nm, t in flat:
        if grp(nm) == group:
            key = nm[len(group) + 1:] if group else nm
            out[key.replace("/", ".")] = t.detach().contiguous().numpy()
    return out, {"source": p.name, "group": group or None, "dropped": sorted(g for g in size if g != group)}


def verify(path, original: dict, threads: int = 1) -> dict:
    """Compare a wpack file with the tensors it was made from. Raises on any tensor that breaks its guarantee."""
    head, _ = read_header(path)
    got = unpack(path, threads=threads)
    rep = {"tensors": len(got), "exact": 0, "f16": 0, "quantized": 0, "max_err_in_steps": 0.0}
    for t in head["tensors"]:
        a, b = np.asarray(original[t["name"]]), got[t["name"]]
        if list(a.shape) != t["shape"]:
            raise ValueError(f"{t['name']}: shape differs")
        if t["codec"] == "q4":
            step = float(np.uint32(t["step_bits"]).view(np.float32))
            e = float(np.abs(b.astype(np.float64) - a.astype(np.float64)).max()) / step
            if e > 0.5 + 1e-3:
                raise ValueError(f"{t['name']}: error of {e:.4f} steps")
            rep["quantized"] += 1
            rep["max_err_in_steps"] = max(rep["max_err_in_steps"], e)
        elif t["codec"] == "f16x" and t.get("cast"):
            if not np.array_equal(a.astype(np.float16).astype(np.float32).view(np.uint32), b.view(np.uint32)):
                raise ValueError(f"{t['name']}: differs from the float16 cast")
            rep["f16"] += 1
        else:
            if a.tobytes() != b.tobytes():
                raise ValueError(f"{t['name']}: not bit-exact")
            rep["exact"] += 1
    if set(original) != set(got):
        raise ValueError("tensor names differ")
    return rep


def summary(head: dict) -> dict:
    by, raw = {}, 0
    for t in head["tensors"]:
        nb = t["n"] * np.dtype(t["dtype"]).itemsize
        raw += nb
        c = by.setdefault(t["codec"] + ("+cast" if t.get("cast") else ""), [0, 0, 0])
        c[0] += 1
        c[1] += nb
        c[2] += sum(t["len"])
    packed = sum(sum(t["len"]) for t in head["tensors"])
    return {"tensors": len(head["tensors"]), "raw_bytes": raw, "packed_bytes": packed, "ratio": round(packed / raw, 4) if raw else None,
            "by_codec": {k: {"tensors": v[0], "raw_bytes": v[1], "packed_bytes": v[2]} for k, v in by.items()}}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="wpack", description="Pack and unpack neural network weights.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--tier", default="lossless", choices=["lossless", "f16", "q"])
    p.add_argument("--s", type=float, default=128.0, help="tier q: step = std / s (128 is about 9 bits per weight, 64 about 8)")
    p.add_argument("--group", default=None)
    p.add_argument("--threads", type=int, default=4)
    u = sub.add_parser("unpack")
    u.add_argument("src")
    u.add_argument("dst")
    u.add_argument("--threads", type=int, default=4)
    i = sub.add_parser("info")
    i.add_argument("src")
    v = sub.add_parser("verify")
    v.add_argument("src")
    v.add_argument("original")
    v.add_argument("--group", default=None)
    v.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    if a.cmd == "pack":
        ts, info = load_model_tensors(a.src, a.group)
        head = pack(ts, a.dst, a.tier, a.s, a.threads, meta=info)
        print(json.dumps({**summary(head), "file_bytes": head["file_bytes"], "source_bytes": Path(a.src).stat().st_size, **info}, indent=1))
    elif a.cmd == "unpack":
        from safetensors.numpy import save_file
        save_file({k: np.asarray(x, order="C") for k, x in unpack(a.src, threads=a.threads).items()}, a.dst)
    elif a.cmd == "info":
        head, _ = read_header(a.src)
        print(json.dumps({"tier": head["tier"], "s": head["s"], **summary(head), "meta": head["meta"]}, indent=1))
    else:
        ts, _ = load_model_tensors(a.original, a.group)
        print(json.dumps(verify(a.src, ts, a.threads), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
