# Metrics

All metrics are computed per frame on the 8-bit RGB output against the sharp reference (full-reference) or on the
output alone (no-reference), then averaged over frames (and reported per sequence). They are computed from the stored
FFV1 outputs in one scoring environment, so every method is scored by the same code, and a new metric never needs a
model rerun. Implementation: [pyiqa (IQA-PyTorch)](https://github.com/chaofengc/IQA-PyTorch), one pinned version,
with the names below.

## Requested

| Metric | Type | pyiqa name | Notes |
|---|---|---|---|
| PSNR | FR | `psnr` | RGB, also kept from the in-memory scorer of the run (identical by construction) |
| SSIM | FR | `ssimc` | RGB; `ssim` in pyiqa is the Y channel. The run's own SSIM is kept as `ssim_skimage`: skimage, RGB, Gaussian window sigma 1.5, population covariance, averaged over channels (the scorer of our paper) |
| MS-SSIM | FR | `ms_ssim` | |
| LPIPS | FR | `lpips` | AlexNet, as in most deblurring papers; `lpips-vgg` as an extra column |
| DISTS | FR | `dists` | |
| PaQ-2-PiQ | NR | `paq2piq` | |
| MUSIQ | NR | `musiq` | KonIQ-trained default; `musiq-spaq` optional |

## Added: the most used in 2024-2026 restoration papers

| Metric | Type | pyiqa name | Why |
|---|---|---|---|
| CLIP-IQA+ | NR | `clipiqa+` | the standard NR score of generative restoration papers (SUPIR, DiffBIR, OSEDiff and the like) |
| MANIQA | NR | `maniqa` | reported next to MUSIQ and CLIP-IQA in nearly every recent restoration table |
| Q-Align | NR | `qalign` | large multimodal model (mPLUG-Owl2, 7B) scoring; the current reference for perceptual quality, slow |

Optional, cheap to add later: TOPIQ (`topiq_fr`, `topiq_nr`), NIQE (`niqe`), FID / KID over a whole test set
(set-level, not per frame), Q-ReAlign (`qrealign`, Qwen-based successor of Q-Align).

## Video

- **Temporal consistency**: warping error between consecutive outputs, with optical flow computed on the sharp
  references (MEMFOF), and its LPIPS form (tLPIPS). Reported against the same quantity of the ground truth, so a
  flicker-free method is at the GT level, not at zero.

## Cost (estimate, A100)

About 8 100 test frames per method (720p equivalent about 6 100). FR metrics and PaQ-2-PiQ / MUSIQ / CLIP-IQA+ /
MANIQA: well under 0.1 s per frame together. Q-Align: about 0.3-0.5 s per frame, so about 1 GPU-hour per method and
about 100 GPU-hours for the full table. It is scored last and can be skipped per run.
