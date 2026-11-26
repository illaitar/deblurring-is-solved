---
license: cc-by-4.0
pretty_name: "deblurring is solved?: compact test sets"
task_categories:
- image-to-image
tags:
- deblurring
- motion-blur
- video-deblurring
- benchmark
size_categories:
- 1K<n<10K
---

# Compact test sets

The test pairs of four public deblurring test sets, exactly the frames the benchmark
[deblurring is solved?](https://illaitar.github.io/deblurring-is-solved/) scores, re-encoded without loss: per sequence
one FFV1 file of blurry frames and one of sharp frames. The pixels are the original PNG pixels; every frame's CRC is
listed and was checked after decoding.

| Folder | Test set | Pairs | Files of | Size here | PNG of these pairs | To get them from the authors |
|---|---|---:|---|---:|---:|---|
| `gopro_test/` | GoPro test (Nah, Kim and Lee, CVPR 2017) | 1,111 | 11 sequences | 1.29 GB | 2.38 GB | GOPRO_Large, 8.9 GiB with the training set |
| `reds_val/` | REDS val (Nah et al., CVPR Workshops 2019) | 3,000 | 30 sequences | 4.45 GB | 7.83 GB | val_blur and val_sharp, 7.3 GiB |
| `realblur_j_test/` | RealBlur-J test (Rim, Lee, Won and Cho, ECCV 2020) | 980 | 50 scenes, one folder | 0.77 GB | 1.15 GB | RealBlur, 12.2 GiB with RealBlur-R and the training set |
| `realblur_r_test/` | RealBlur-R test | 980 | 50 scenes, one folder | 0.38 GB | 0.59 GB | (the same archive) |

- **GoPro**: the `blur` input (the averaged frames without the gamma curve of `blur_gamma`), as the benchmark uses it.
- **REDS**: `val_blur` against `val_sharp`, 100 frames per clip; the test ground truth is not released.
- **RealBlur**: the pairs of the official `RealBlur_J_test_list.txt` and `RealBlur_R_test_list.txt` (the
  `ECC_IMCORR_centroid_itensity_ref` images), in the order of the lists, in one folder `all/`. The aligned images
  differ in size by a few pixels (652 to 675 by 762 to 763), so each file codes them at the largest size, padded with
  zeros at the bottom and right; `frames.tsv` gives every image's own size, to which a reader crops. Score them with
  the dataset's official ECC-aligned evaluation.

## Format

Each folder (`gopro_test/GOPR0384_11_00/`, ...) holds

- `blur.mkv`, `sharp.mkv`: FFV1 level 3 in Matroska, `bgr0` (exact 8-bit RGB), coder 1, context 1, 4 slices, one GOP
  per file;
- `frames.tsv`: `index`, the original relative paths of the blurry and the sharp frame (`blur`, `sharp`), the
  frame's own `height` and `width`, and the CRC-32 of its H x W x 3 RGB bytes (`crc32_blur`, `crc32_sharp`).

`manifest.json` lists every file with its size and sha256. Decoding needs ffmpeg (on PATH, or
`pip install imageio-ffmpeg`):

```bash
ffmpeg -i gopro_test/GOPR0384_11_00/blur.mkv gopro_test/GOPR0384_11_00/blur_%06d.png   # back to PNGs (RealBlur: then crop)
```

or, in Python, `tools/testsets.py` of the [repository](https://github.com/illaitar/deblurring-is-solved):

```python
from tools.testsets import download, sequences, read
root = download()                                  # into data/testsets/
for seq in sequences(f"{root}/gopro_test"):
    for blur_name, sharp_name, blur, sharp in read(f"{root}/gopro_test", seq):
        ...                                        # H x W x 3 uint8 RGB, each frame checked against its CRC
```

## Terms and changes

All four sets are released by their authors under the Creative Commons Attribution 4.0 International license
(CC BY 4.0), and so is this copy; see `LICENSE.md`. Changes made: only the test pairs used by the benchmark are kept,
the PNG files are re-encoded losslessly into FFV1 files, one per sequence, with the original file names and order in
`frames.tsv`. Nothing else is changed. Cite the papers of the sets you use:

```bibtex
@InProceedings{Nah_2017_CVPR,
  author = {Nah, Seungjun and Kim, Tae Hyun and Lee, Kyoung Mu},
  title = {Deep Multi-Scale Convolutional Neural Network for Dynamic Scene Deblurring},
  booktitle = {CVPR},
  month = {July},
  year = {2017}
}
@InProceedings{Nah_2019_CVPR_Workshops_REDS,
  author = {Nah, Seungjun and Baik, Sungyong and Hong, Seokil and Moon, Gyeongsik and Son, Sanghyun and Timofte, Radu and Lee, Kyoung Mu},
  title = {NTIRE 2019 Challenge on Video Deblurring and Super-Resolution: Dataset and Study},
  booktitle = {CVPR Workshops},
  month = {June},
  year = {2019}
}
@inproceedings{rim_2020_ECCV,
  title = {Real-World Blur Dataset for Learning and Benchmarking Deblurring Algorithms},
  author = {Rim, Jaesung and Lee, Haeyun and Won, Jucheol and Cho, Sunghyun},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2020}
}
```

The other test sets of the benchmark (BSD, DVD, GS-Blur, MIORe) state no license for redistribution of their data
and are not copied here; get them from their authors.
