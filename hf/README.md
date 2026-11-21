---
pretty_name: deblurring is solved, frames and per-frame scores
license: other
license_name: mixed-source-terms
license_link: LICENSE.md
task_categories:
- image-to-image
tags:
- deblurring
- motion-blur
- video-deblurring
- benchmark
size_categories:
- 1M<n<10M
configs:
- config_name: frames
  data_files: data/frames.parquet
- config_name: methods
  data_files: data/methods.parquet
- config_name: scores
  data_files: data/scores.parquet
---

# deblurring is solved: frames and per-frame scores

Companion data of the **deblurring is solved** benchmark (Nikita Alutis, Egor Chistov): motion deblurring methods with
released weights, run with one protocol on six test sets and scored frame by frame.

## Contents

| Path | What it is |
|---|---|
| `data/frames.parquet` | every test frame of the six sets (GoPro test, REDS val, BSD 2ms/16ms test, DVD test, RealBlur-J test, RealBlur-R test) with its characteristics |
| `data/methods.parquet` | every benchmark row: method, variant, type, training set |
| `data/scores.parquet` | the score of every row on every frame: PSNR, SSIM, LPIPS, MS-SSIM, LPIPS-VGG, DISTS, PaQ-2-PiQ, MUSIQ, CLIP-IQA+, BASED-4 |
| `data/bench.sqlite` | the same three tables in one SQLite file, with a `scored` view joining scores to frames |
| `img/<set>/<n>.bin` | frames for the site's viewer: the sharp frame, the blurry input and several methods' outputs with their error maps |
| `img/index.json` | the list of viewer frames, why each was picked, and each output's PSNR on it |

### Frame characteristics

Computed from the sharp frame and its blurry input: `psnr_in` (PSNR of the blurry input, dB), `grad_ratio` (mean gradient
of the input over that of the sharp frame), `texture`, `brightness`, `contrast`, `highlights` (% of input pixels with a
channel at 250 or more), `noise` (Immerkaer estimate on the input luma), `colorfulness` (Hasler and Suesstrunk).
On the four video sets: `motion`, `motion_p90`, `shift`, `nonuniform` from MEMFOF optical flow between consecutive sharp
frames. `psnr_med` and `psnr_best`: median and maximum PSNR over the GoPro-trained methods scored on every frame.

### Scores

PSNR and SSIM on 8-bit RGB against the sharp frame; on RealBlur by the dataset's official ECC-aligned evaluation.
LPIPS (AlexNet) in memory during the runs. The other metrics are pyiqa 0.1.16 defaults in fp32 on the stored lossless
outputs. BASED-4 scores an output against its own blurry input (based-metric 0.2.0).

### Viewer bundles

`b"DBIS"`, a little-endian uint32 header length, a JSON header `{ds, seq, stem, w, h, items: [{key, kind, off, len}]}`,
then the files: the sharp frame as lossless WebP, each image as WebP (quality 85), and each error map
|output − sharp| (mean over RGB, 8 bit, from the lossless outputs) as lossless grey WebP.

## Terms

The images are frames of public datasets and outputs of published methods on them; each source keeps its own terms,
listed in `LICENSE.md`. Cite the datasets you use (BibTeX on the benchmark site, one card per test set).
