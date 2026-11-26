# Terms

The four test sets in this repository are copies, losslessly re-encoded, of data released by their authors under the
Creative Commons Attribution 4.0 International license (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/):

| Folder | Source | Authors | Terms stated by the authors |
|---|---|---|---|
| `gopro_test/` | GoPro, https://seungjunnah.github.io/Datasets/gopro.html | Seungjun Nah, Tae Hyun Kim, Kyoung Mu Lee | CC BY 4.0 |
| `reds_val/` | REDS, https://seungjunnah.github.io/Datasets/reds.html | Seungjun Nah, Sungyong Baik, Seokil Hong, Gyeongsik Moon, Sanghyun Son, Radu Timofte, Kyoung Mu Lee | CC BY 4.0 |
| `realblur_j_test/`, `realblur_r_test/` | RealBlur, http://cg.postech.ac.kr/research/RealBlur/ | Jaesung Rim, Haeyun Lee, Jucheol Won, Sunghyun Cho | CC BY 4.0 |

Changes: only the test pairs used by the benchmark deblurring is solved? are kept; the PNG images are re-encoded
without loss into FFV1 video files, one per sequence (RealBlur: per scene and image size); the original file names and
order are listed in each folder's `frames.tsv`. The pixel values are unchanged.

This copy is distributed under the same license, CC BY 4.0. The authors do not endorse this copy.
