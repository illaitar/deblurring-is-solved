"""The training samplers of dbis.train on the CPU: GpuFrames cuts the crops, flips and transposes that Clips' numpy
code would cut for the same draws; shuffle perm visits the clips of an epoch in its permutation; Clips with echo k
reads each clip once for k batches.

    pytest tests/test_sampling.py
"""
import numpy as np
import torch

from dbis.train import Clips, GpuFrames, Stream

RNG = np.random.default_rng(0)
SEQS = {f"s{q}": [(f"s{q}/b/{i}", f"s{q}/g/{i}") for i in range(n)] for q, n in enumerate((13, 9, 11))}
STORE = {p: RNG.integers(0, 256, (40, 48, 3), dtype=np.uint8) for v in SEQS.values() for pr in v for p in pr}
PATHS = [pr for v in SEQS.values() for pr in v]
CPU = torch.device("cpu")


def draws(gf, n, b, crop, seed, rank=0):
    g = torch.Generator(device=CPU)
    g.manual_seed(((seed * 1_000_003 + rank) * 1_000_003 + n) % 2 ** 63)
    s = gf.starts[torch.randint(len(gf.starts), (b,), generator=g)].tolist()
    r0 = torch.randint(40 - crop + 1, (b, 1), generator=g)[:, 0].tolist()
    c0 = torch.randint(48 - crop + 1, (b, 1), generator=g)[:, 0].tolist()
    f = (torch.rand(b, 4, generator=g) < 0.5).tolist()
    return s, r0, c0, f


def reference(s, r0, c0, f, side, frames, crop, reverse):
    a = np.stack([STORE[PATHS[s + t][side]][r0:r0 + crop, c0:c0 + crop] for t in range(frames)])
    a = a[:, :, ::-1] if f[0] else a
    a = a[:, ::-1] if f[1] else a
    a = a.transpose(0, 2, 1, 3) if f[2] else a
    a = a[::-1] if f[3] and reverse and frames > 1 else a
    return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2)


def test_gpu_frames_match_numpy():
    for frames, reverse in ((1, False), (3, True)):
        gf = GpuFrames(SEQS, frames, STORE.__getitem__, CPU)
        for n in range(3):
            x, y = gf.cut(n, 6, 16, True, reverse, 5)
            s, r0, c0, f = draws(gf, n, 6, 16, 5)
            for j in range(6):
                assert torch.equal(x[j], reference(s[j], r0[j], c0[j], f[j], 0, frames, 16, reverse))
                assert torch.equal(y[j], reference(s[j], r0[j], c0[j], f[j], 1, frames, 16, reverse))


def test_perm_crosses_the_epoch():
    gf = GpuFrames(SEQS, 1, STORE.__getitem__, CPU)
    m = len(gf.starts)
    perms = [torch.randperm(m, generator=torch.Generator().manual_seed((3 * 1_000_003 + e) % 2 ** 63)) for e in (0, 1)]
    x, _ = gf.cut(0, 6, 16, False, False, 3, 0, "perm", m - 3, 1)
    _, r0, c0, _ = draws(gf, 0, 6, 16, 3)
    for j in range(6):
        e, k = divmod(m - 3 + j, m)
        want = int(gf.starts[perms[e][k]])
        assert torch.equal(x[j], reference(want, r0[j], c0[j], [False] * 4, 0, 1, 16, False))


def test_echo_reads_each_clip_once_per_group():
    reads = []

    def reader(p):
        reads.append(p)
        return STORE[p]

    b, k = 4, 3
    ds = Clips(SEQS, 1, 16, reader, seed=1, echo=k, batch=b, workers=0)
    for i in range(b * k * 2):                         # two groups of k batches, read by the main process
        ds[i]
    assert len(reads) == 2 * 2 * b                     # 2 groups x b clips x (blurry + sharp)
    one = Clips(SEQS, 1, 16, reader, seed=1)
    reads.clear()
    for i in Stream(0, 0, 1):
        if i == b * k:
            break
        one[i]
    assert len(reads) == 2 * b * k
