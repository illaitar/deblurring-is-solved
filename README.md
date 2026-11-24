<h1 align="center">deblurring is solved</h1>

<p align="center">Nikita Alutis &nbsp;&middot;&nbsp; Egor Chistov</p>

<p align="center">
  <b>Motion deblurring methods with released weights, run with one protocol on six test sets (four video sets, plus RealBlur-J and RealBlur-R for image methods) and scored frame by frame.</b>
</p>

<p align="center">
  <a href="https://illaitar.github.io/deblurring-is-solved/"><b>Interactive leaderboard: illaitar.github.io/deblurring-is-solved</b></a>
</p>

<p align="center">
  <a href="#leaderboard">Leaderboard</a> &nbsp;&middot;&nbsp;
  <a href="docs/leaderboard.csv">All results (CSV)</a> &nbsp;&middot;&nbsp;
  <a href="PROTOCOL.md">Protocol</a> &nbsp;&middot;&nbsp;
  <a href="METRICS.md">Metrics</a> &nbsp;&middot;&nbsp;
  <a href="WEIGHTS.md">Weights</a>
</p>

## Leaderboard

<!-- generated:start (tools/build_site.py writes this block; edit the script, not the text) -->
> [!NOTE]
> Numbers as of 2026-11-24: 267 of 268 variants scored on all their sets (six for image methods, four for video methods), 0 on some. GoPro-trained weights (other training sets are on the site), PSNR in dB (RealBlur: official ECC-aligned PSNR), sorted by PSNR. The full table is in [`docs/leaderboard.csv`](docs/leaderboard.csv) and on the [interactive leaderboard](https://illaitar.github.io/deblurring-is-solved/).

### GoPro

| # | Method | Type | Params, M | PSNR | SSIM | LPIPS |
|---:|---|---|---:|---:|---:|---:|
| 1 | **BSSTNet** <sub>clip, CVPR 2024</sub> | video | 46.7 | 35.97 | 0.966 | 0.057 |
| 2 | **DSTNet+ L** <sub>clip, large, arXiv 2024</sub> | video | 14.1 | 35.71 | 0.964 | 0.059 |
| 3 | **Shift-Net+** <sub>clip, CVPR 2023</sub> | video | 13.0 | 35.65 | 0.964 | 0.059 |
| 4 | **STDANet++** <sub>clip, tiled, TCSVT 2025</sub> | video | 11.5 | 35.29 | 0.961 | 0.065 |
| 5 | **MVSSM-L** <sub>clip, large, CVPR-F 2026</sub> | video | 30.2 | 35.27 | 0.960 | 0.065 |
| 6 | **RVRT** <sub>clip, NeurIPS 2022</sub> | video | 13.6 | 34.92 | 0.959 | 0.067 |
| 7 | **VRT** <sub>clip, TIP 2022</sub> | video | 18.3 | 34.81 | 0.957 | 0.069 |
| 8 | **Turtle** <sub>tiled, NeurIPS 2024</sub> | video | 59.1 | 34.56 | 0.955 | 0.072 |
| 9 | **EVSSM** <sub>whole, CVPR 2025</sub> | image | 17.1 | 34.51 | 0.954 | 0.068 |
| 10 | **AdaRevD-B** <sub>tiled, CVPR 2024</sub> | image | 152.4 | 34.50 | 0.954 | 0.073 |

<details>
<summary>All 116 rows on GoPro</summary>

| # | Method | Type | Params, M | PSNR | SSIM | LPIPS |
|---:|---|---|---:|---:|---:|---:|
| 1 | **BSSTNet** <sub>clip, CVPR 2024</sub> | video | 46.7 | 35.97 | 0.966 | 0.057 |
| 2 | **DSTNet+ L** <sub>clip, large, arXiv 2024</sub> | video | 14.1 | 35.71 | 0.964 | 0.059 |
| 3 | **Shift-Net+** <sub>clip, CVPR 2023</sub> | video | 13.0 | 35.65 | 0.964 | 0.059 |
| 4 | **STDANet++** <sub>clip, tiled, TCSVT 2025</sub> | video | 11.5 | 35.29 | 0.961 | 0.065 |
| 5 | **MVSSM-L** <sub>clip, large, CVPR-F 2026</sub> | video | 30.2 | 35.27 | 0.960 | 0.065 |
| 6 | **RVRT** <sub>clip, NeurIPS 2022</sub> | video | 13.6 | 34.92 | 0.959 | 0.067 |
| 7 | **VRT** <sub>clip, TIP 2022</sub> | video | 18.3 | 34.81 | 0.957 | 0.069 |
| 8 | **Turtle** <sub>tiled, NeurIPS 2024</sub> | video | 59.1 | 34.56 | 0.955 | 0.072 |
| 9 | **EVSSM** <sub>whole, CVPR 2025</sub> | image | 17.1 | 34.51 | 0.954 | 0.068 |
| 10 | **AdaRevD-B** <sub>tiled, CVPR 2024</sub> | image | 152.4 | 34.50 | 0.954 | 0.073 |
| 11 | **DSTNet+ base** <sub>clip, arXiv 2024</sub> | video | 3.9 | 34.49 | 0.955 | 0.073 |
| 12 | **Concertormer** <sub>TLC, arXiv 2025</sub> | image | 50.5 | 34.41 | 0.954 | 0.070 |
| 13 | **FFTformer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 16.6 | 34.29 | 0.953 | 0.069 |
| 14 | **StarIR** <sub>TLC, 2025</sub> | image | 18.7 | 34.27 | 0.953 | 0.071 |
| 15 | **MVSSM** <sub>clip, CVPR-F 2026</sub> | video | 12.8 | 34.27 | 0.952 | 0.076 |
| 16 | **Concertormer-lite** <sub>TLC, small, arXiv 2025</sub> | image | 29.0 | 34.18 | 0.952 | 0.072 |
| 17 | **FFTformer** <sub>whole, CVPR 2023</sub> | image | 16.6 | 34.16 | 0.952 | 0.071 |
| 18 | **DSTNet** <sub>clip, CVPR 2023</sub> | video | 7.5 | 34.16 | 0.952 | 0.073 |
| 19 | **UFPNet** <sub>TLC, CVPR 2023</sub> | image | 80.3 | 34.09 | 0.950 | 0.076 |
| 20 | **Concertormer** <sub>whole, arXiv 2025</sub> | image | 50.5 | 34.07 | 0.951 | 0.073 |
| 21 | **BasicVSR++** <sub>clip, CVPR 2022</sub> | video | 9.8 | 34.04 | 0.952 | 0.078 |
| 22 | **GRL-B** <sub>tiled, CVPR 2023</sub> | image | 19.8 | 33.93 | 0.949 | 0.074 |
| 23 | **XYScanNet** <sub>whole, CVPRW 2025</sub> | image | 63.5 | 33.91 | 0.949 | 0.067 |
| 24 | **Concertormer-lite** <sub>whole, small, arXiv 2025</sub> | image | 29.0 | 33.86 | 0.950 | 0.075 |
| 25 | **F-NAFNet-64** <sub>whole, AAAI 2023</sub> | image | 68.0 | 33.85 | 0.948 | 0.075 |
| 26 | **MLWNet-B** <sub>TLC, CVPR 2024</sub> | image | 95.1 | 33.83 | 0.949 | 0.076 |
| 27 | **M3SNet-64** <sub>TLC, VisComp 2024</sub> | image | 66.4 | 33.73 | 0.945 | 0.082 |
| 28 | **LaKDNet-L** <sub>whole, arXiv 2023</sub> | image | 37.5 | 33.72 | 0.947 | 0.076 |
| 29 | **StarIR** <sub>whole, 2025</sub> | image | 18.7 | 33.72 | 0.949 | 0.074 |
| 30 | **NAFNet-64** <sub>TLC, ECCV 2022</sub> | image | 67.9 | 33.71 | 0.947 | 0.078 |
| 31 | **MaIR** <sub>whole, CVPR 2025</sub> | image | 26.3 | 33.69 | 0.947 | 0.079 |
| 32 | **GRL-B** <sub>whole, CVPR 2023</sub> | image | 19.8 | 33.66 | 0.947 | 0.076 |
| 33 | **Stripformer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 19.7 | 33.66 | 0.947 | 0.071 |
| 34 | **DeblurDiNAT-L** <sub>whole, arXiv 2024</sub> | image | 16.1 | 33.63 | 0.947 | 0.071 |
| 35 | **EAMamba** <sub>whole, ICCV 2025</sub> | image | 25.3 | 33.58 | 0.946 | 0.079 |
| 36 | **F-MIMO+** <sub>tiled, large, AAAI 2023</sub> | image | 19.5 | 33.53 | 0.946 | 0.079 |
| 37 | **Restormer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 26.1 | 33.51 | 0.946 | 0.078 |
| 38 | **UFPNet** <sub>whole, CVPR 2023</sub> | image | 80.3 | 33.46 | 0.944 | 0.080 |
| 39 | **AdaRevD-B** <sub>whole, CVPR 2024</sub> | image | 152.4 | 33.44 | 0.944 | 0.090 |
| 40 | **MSFS-Net** <sub>TLC, TCSVT 2023</sub> | image | 21.6 | 33.44 | 0.944 | 0.083 |
| 41 | **ShuffleFormer** <sub>tiled, ICML 2023</sub> | image | 50.6 | 33.38 | 0.945 | 0.083 |
| 42 | **MLWNet-B** <sub>whole, CVPR 2024</sub> | image | 95.1 | 33.37 | 0.945 | 0.079 |
| 43 | **X-Restormer** <sub>whole, ECCV 2024</sub> | image | 26.0 | 33.37 | 0.946 | 0.076 |
| 44 | **RNN-MBP** <sub>clip, AAAI 2022</sub> | video | 16.4 | 33.36 | 0.942 | 0.083 |
| 45 | **LaKDNet-S** <sub>whole, small, arXiv 2023</sub> | image | 17.1 | 33.35 | 0.944 | 0.082 |
| 46 | **LoFormer-L** <sub>whole, ACMMM 2024</sub> | image | 49.0 | 33.35 | 0.943 | 0.083 |
| 47 | **RevD-S** <sub>whole, small, CVPR 2024</sub> | image | 108.2 | 33.34 | 0.943 | 0.088 |
| 48 | **HI-Diff** <sub>whole, NeurIPS 2023</sub> | image | 28.5 | 33.32 | 0.944 | 0.080 |
| 49 | **DiffIR** <sub>whole, ICCV 2023</sub> | image | 26.9 | 33.31 | 0.945 | 0.079 |
| 50 | **MSDI-Net** <sub>tiled, ECCV 2022</sub> | image | 246.9 | 33.28 | 0.942 | 0.080 |
| 51 | **FSNet** <sub>whole, TPAMI 2023</sub> | image | 13.3 | 33.26 | 0.943 | 0.081 |
| 52 | **ChaIR** <sub>whole, KBS 2023</sub> | image | 15.0 | 33.25 | 0.943 | 0.081 |
| 53 | **Stoformer** <sub>tiled, NeurIPS 2022</sub> | image | 50.5 | 33.23 | 0.943 | 0.085 |
| 54 | **DeepRFT+** <sub>TLC, large, arXiv 2021</sub> | image | 23.0 | 33.23 | 0.943 | 0.080 |
| 55 | **SFNet** <sub>TLC, ICLR 2023</sub> | image | 13.3 | 33.23 | 0.943 | 0.080 |
| 56 | **ConvIR-L** <sub>whole, TPAMI 2024</sub> | image | 14.8 | 33.22 | 0.943 | 0.083 |
| 57 | **Stoformer** <sub>whole, NeurIPS 2022</sub> | image | 50.5 | 33.19 | 0.943 | 0.084 |
| 58 | **VDTR** <sub>window, TCSVT 2022</sub> | video | 23.1 | 33.19 | 0.940 | 0.087 |
| 59 | **DGUNet+** <sub>whole, large, CVPR 2022</sub> | image | 69.6 | 33.17 | 0.942 | 0.082 |
| 60 | **F-NAFNet-32** <sub>whole, small, AAAI 2023</sub> | image | 17.8 | 33.12 | 0.941 | 0.085 |
| 61 | **IRNeXt (16 blocks)** <sub>whole, large, ICML 2023</sub> | image | 14.8 | 33.12 | 0.942 | 0.082 |
| 62 | **M3SNet-64** <sub>whole, VisComp 2024</sub> | image | 66.4 | 33.09 | 0.939 | 0.087 |
| 63 | **IRNeXt** <sub>whole, ICML 2023</sub> | image | 13.2 | 33.09 | 0.941 | 0.084 |
| 64 | **Stripformer** <sub>whole, ECCV 2022</sub> | image | 19.7 | 33.08 | 0.941 | 0.077 |
| 65 | **F-MIMO** <sub>tiled, AAAI 2023</sub> | image | 8.2 | 33.08 | 0.942 | 0.085 |
| 66 | **NAFNet-64** <sub>whole, ECCV 2022</sub> | image | 67.9 | 33.08 | 0.942 | 0.082 |
| 67 | **FGST** <sub>window, ICML 2022</sub> | video | 9.7 | 33.03 | 0.947 | 0.081 |
| 68 | **MSSNet** <sub>whole, ECCVW 2022</sub> | image | 15.6 | 33.01 | 0.940 | 0.086 |
| 69 | **CDVD-TSPNL** <sub>clip, TPAMI 2023</sub> | video | 5.5 | 32.93 | 0.942 | 0.090 |
| 70 | **MIMO-UNet+ + ID-Blau** <sub>whole, large, CVPR 2024</sub> | image | 16.1 | 32.93 | 0.939 | 0.088 |
| 71 | **Restormer** <sub>whole, CVPR 2022</sub> | image | 26.1 | 32.92 | 0.940 | 0.084 |
| 72 | **ShuffleFormer** <sub>whole, ICML 2023</sub> | image | 50.6 | 32.90 | 0.940 | 0.086 |
| 73 | **HINet** <sub>whole, CVPRW 2021</sub> | image | 88.7 | 32.90 | 0.938 | 0.086 |
| 74 | **MAXIM-3S** <sub>whole, CVPR 2022</sub> | image | 22.2 | 32.90 | 0.941 | 0.087 |
| 75 | **M3SNet-32** <sub>TLC, small, VisComp 2024</sub> | image | 16.7 | 32.90 | 0.938 | 0.093 |
| 76 | **NAFNet-32** <sub>TLC, small, ECCV 2022</sub> | image | 17.1 | 32.87 | 0.938 | 0.093 |
| 77 | **ShuffleFormer (1 pass)** <sub>whole, 1pass, ICML 2023</sub> | image | 50.6 | 32.86 | 0.940 | 0.087 |
| 78 | **DeepRFT** <sub>TLC, arXiv 2021</sub> | image | 9.6 | 32.82 | 0.939 | 0.085 |
| 79 | **SFNet** <sub>whole, ICLR 2023</sub> | image | 13.3 | 32.77 | 0.938 | 0.084 |
| 80 | **MSFS-Net** <sub>whole, TCSVT 2023</sub> | image | 21.6 | 32.72 | 0.936 | 0.088 |
| 81 | **DGUNet** <sub>whole, CVPR 2022</sub> | image | 17.3 | 32.71 | 0.937 | 0.091 |
| 82 | **MPRNet** <sub>whole, CVPR 2021</sub> | image | 20.1 | 32.66 | 0.936 | 0.089 |
| 83 | **MSDI-Net** <sub>whole, ECCV 2022</sub> | image | 246.9 | 32.59 | 0.937 | 0.080 |
| 84 | **STDAN** <sub>window, ECCV 2022</sub> | video | 13.8 | 32.51 | 0.936 | 0.097 |
| 85 | **F-MIMO-S** <sub>tiled, small, AAAI 2023</sub> | image | 4.4 | 32.50 | 0.935 | 0.093 |
| 86 | **DeepRFT+** <sub>whole, large, arXiv 2021</sub> | image | 23.0 | 32.45 | 0.933 | 0.097 |
| 87 | **MIMO-UNet+** <sub>whole, large, ICCV 2021</sub> | image | 16.1 | 32.44 | 0.933 | 0.090 |
| 88 | **F-MIMO+** <sub>whole, large, AAAI 2023</sub> | image | 19.5 | 32.38 | 0.932 | 0.098 |
| 89 | **M3SNet-32** <sub>whole, small, VisComp 2024</sub> | image | 16.7 | 32.32 | 0.931 | 0.098 |
| 90 | **DeepRFT-S** <sub>TLC, small, arXiv 2021</sub> | image | 5.1 | 32.30 | 0.933 | 0.092 |
| 91 | **NAFNet-32** <sub>whole, small, ECCV 2022</sub> | image | 17.1 | 32.29 | 0.932 | 0.098 |
| 92 | **MSSNet-small** <sub>whole, small, ECCVW 2022</sub> | image | 6.8 | 32.02 | 0.928 | 0.100 |
| 93 | **DeepRFT** <sub>whole, arXiv 2021</sub> | image | 9.6 | 31.91 | 0.926 | 0.105 |
| 94 | **VRNN** <sub>clip, PR 2023</sub> | video | 5.0 | 31.81 | 0.924 | 0.102 |
| 95 | **MemDeblur** <sub>clip, CVPR 2022</sub> | video | 7.0 | 31.78 | 0.923 | 0.115 |
| 96 | **F-MIMO** <sub>whole, AAAI 2023</sub> | image | 8.2 | 31.78 | 0.924 | 0.107 |
| 97 | **MIMO-UNet** <sub>whole, ICCV 2021</sub> | image | 6.8 | 31.72 | 0.924 | 0.101 |
| 98 | **CDVD-TSP** <sub>window, CVPR 2020</sub> | video | 16.2 | 31.61 | 0.927 | 0.114 |
| 99 | **DeepRFT-S** <sub>whole, small, arXiv 2021</sub> | image | 5.1 | 31.36 | 0.918 | 0.115 |
| 100 | **F-MIMO-S** <sub>whole, small, AAAI 2023</sub> | image | 4.4 | 31.20 | 0.915 | 0.119 |
| 101 | **DBGAN** <sub>whole, CVPR 2020</sub> | image | 11.6 | 31.18 | 0.916 | 0.110 |
| 102 | **MT-RNN** <sub>whole, ECCV 2020</sub> | image | 2.6 | 31.13 | 0.915 | 0.123 |
| 103 | **Motion-ETR** <sub>whole, TPAMI 2021</sub> | image | 6.6 | 31.07 | 0.913 | 0.117 |
| 104 | **PSS-NSC** <sub>whole, CVPR 2019</sub> | image | 3.9 | 30.99 | 0.913 | 0.122 |
| 105 | **ESTRNN** <sub>recurrent, ECCV 2020</sub> | video | 2.5 | 30.90 | 0.908 | 0.143 |
| 106 | **SIUN** <sub>whole, Access 2020</sub> | image | 6.3 | 30.27 | 0.902 | 0.127 |
| 107 | **DMPHN-1-2-4** <sub>whole, CVPR 2019</sub> | image | 5.4 | 30.21 | 0.899 | 0.133 |
| 108 | **TSRN-RTVD** <sub>recurrent, arXiv 2026</sub> | video | 4.9 | 30.07 | 0.897 | 0.141 |
| 109 | **MMP-RNN** <sub>recurrent, ECCV 2022</sub> | video | 4.1 | 29.95 | 0.922 | 0.111 |
| 110 | **PVDNet** <sub>window+prev out, TOG 2021</sub> | video | 10.5 | 28.81 | 0.898 | 0.130 |
| 111 | **STDANet++** <sub>clip, whole, TCSVT 2025</sub> | video | 11.5 | 28.36 | 0.859 | 0.197 |
| 112 | **RT-Focuser** <sub>whole, arXiv 2025</sub> | image | 5.9 | 28.30 | 0.880 | 0.161 |
| 113 | **RT-VDBLR** <sub>recurrent, CGF 2022</sub> | video | 1.0 | 28.04 | 0.874 | 0.165 |
| 114 | **IR-SDE** <sub>whole, ICML 2023</sub> | image | 137.1 | 27.98 | 0.867 | 0.089 |
| 115 | **DeepDeblur** <sub>whole, CVPR 2017</sub> | image | 11.7 | 27.83 | 0.871 | 0.157 |
| 116 | **Ghost-DeblurGAN** <sub>whole, IROS 2022</sub> | image | 1.5 | 27.56 | 0.843 | 0.161 |

</details>

### REDS

| # | Method | Type | Params, M | PSNR | SSIM | LPIPS |
|---:|---|---|---:|---:|---:|---:|
| 1 | **VRNN** <sub>clip, PR 2023</sub> | video | 5.0 | 29.10 | 0.857 | 0.179 |
| 2 | **MMP-RNN** <sub>recurrent, ECCV 2022</sub> | video | 4.1 | 28.72 | 0.851 | 0.201 |
| 3 | **STDAN** <sub>window, ECCV 2022</sub> | video | 13.8 | 28.65 | 0.854 | 0.201 |
| 4 | **PVDNet** <sub>window+prev out, TOG 2021</sub> | video | 10.5 | 28.64 | 0.852 | 0.201 |
| 5 | **RNN-MBP** <sub>clip, AAAI 2022</sub> | video | 16.4 | 28.53 | 0.837 | 0.201 |
| 6 | **RT-VDBLR** <sub>recurrent, CGF 2022</sub> | video | 1.0 | 28.30 | 0.830 | 0.233 |
| 7 | **MSFS-Net** <sub>TLC, TCSVT 2023</sub> | image | 21.6 | 28.04 | 0.848 | 0.183 |
| 8 | **DeepDeblur** <sub>whole, CVPR 2017</sub> | image | 11.7 | 27.91 | 0.839 | 0.220 |
| 9 | **VRT** <sub>clip, TIP 2022</sub> | video | 18.3 | 27.74 | 0.879 | 0.144 |
| 10 | **RVRT** <sub>clip, NeurIPS 2022</sub> | video | 13.6 | 27.69 | 0.884 | 0.139 |

<details>
<summary>All 116 rows on REDS</summary>

| # | Method | Type | Params, M | PSNR | SSIM | LPIPS |
|---:|---|---|---:|---:|---:|---:|
| 1 | **VRNN** <sub>clip, PR 2023</sub> | video | 5.0 | 29.10 | 0.857 | 0.179 |
| 2 | **MMP-RNN** <sub>recurrent, ECCV 2022</sub> | video | 4.1 | 28.72 | 0.851 | 0.201 |
| 3 | **STDAN** <sub>window, ECCV 2022</sub> | video | 13.8 | 28.65 | 0.854 | 0.201 |
| 4 | **PVDNet** <sub>window+prev out, TOG 2021</sub> | video | 10.5 | 28.64 | 0.852 | 0.201 |
| 5 | **RNN-MBP** <sub>clip, AAAI 2022</sub> | video | 16.4 | 28.53 | 0.837 | 0.201 |
| 6 | **RT-VDBLR** <sub>recurrent, CGF 2022</sub> | video | 1.0 | 28.30 | 0.830 | 0.233 |
| 7 | **MSFS-Net** <sub>TLC, TCSVT 2023</sub> | image | 21.6 | 28.04 | 0.848 | 0.183 |
| 8 | **DeepDeblur** <sub>whole, CVPR 2017</sub> | image | 11.7 | 27.91 | 0.839 | 0.220 |
| 9 | **VRT** <sub>clip, TIP 2022</sub> | video | 18.3 | 27.74 | 0.879 | 0.144 |
| 10 | **RVRT** <sub>clip, NeurIPS 2022</sub> | video | 13.6 | 27.69 | 0.884 | 0.139 |
| 11 | **UFPNet** <sub>TLC, CVPR 2023</sub> | image | 80.3 | 27.67 | 0.848 | 0.190 |
| 12 | **AdaRevD-B** <sub>tiled, CVPR 2024</sub> | image | 152.4 | 27.66 | 0.850 | 0.194 |
| 13 | **MSFS-Net** <sub>whole, TCSVT 2023</sub> | image | 21.6 | 27.66 | 0.832 | 0.198 |
| 14 | **UFPNet** <sub>whole, CVPR 2023</sub> | image | 80.3 | 27.58 | 0.840 | 0.199 |
| 15 | **CDVD-TSPNL** <sub>clip, TPAMI 2023</sub> | video | 5.5 | 27.55 | 0.867 | 0.169 |
| 16 | **AdaRevD-B** <sub>whole, CVPR 2024</sub> | image | 152.4 | 27.47 | 0.836 | 0.212 |
| 17 | **STDANet++** <sub>clip, tiled, TCSVT 2025</sub> | video | 11.5 | 27.46 | 0.874 | 0.144 |
| 18 | **RevD-S** <sub>whole, small, CVPR 2024</sub> | image | 108.2 | 27.46 | 0.835 | 0.211 |
| 19 | **BasicVSR++** <sub>clip, CVPR 2022</sub> | video | 9.8 | 27.44 | 0.872 | 0.153 |
| 20 | **ShuffleFormer** <sub>tiled, ICML 2023</sub> | image | 50.6 | 27.40 | 0.846 | 0.189 |
| 21 | **DSTNet+ base** <sub>clip, arXiv 2024</sub> | video | 3.9 | 27.38 | 0.858 | 0.168 |
| 22 | **Concertormer** <sub>TLC, arXiv 2025</sub> | image | 50.5 | 27.35 | 0.850 | 0.182 |
| 23 | **MVSSM** <sub>clip, CVPR-F 2026</sub> | video | 12.8 | 27.35 | 0.850 | 0.185 |
| 24 | **Turtle** <sub>tiled, NeurIPS 2024</sub> | video | 59.1 | 27.35 | 0.847 | 0.184 |
| 25 | **MVSSM-L** <sub>clip, large, CVPR-F 2026</sub> | video | 30.2 | 27.29 | 0.856 | 0.172 |
| 26 | **DSTNet** <sub>clip, CVPR 2023</sub> | video | 7.5 | 27.29 | 0.854 | 0.177 |
| 27 | **Stoformer** <sub>whole, NeurIPS 2022</sub> | image | 50.5 | 27.28 | 0.841 | 0.191 |
| 28 | **RT-Focuser** <sub>whole, arXiv 2025</sub> | image | 5.9 | 27.27 | 0.812 | 0.258 |
| 29 | **Stoformer** <sub>tiled, NeurIPS 2022</sub> | image | 50.5 | 27.27 | 0.841 | 0.192 |
| 30 | **Concertormer** <sub>whole, arXiv 2025</sub> | image | 50.5 | 27.25 | 0.844 | 0.188 |
| 31 | **MaIR** <sub>whole, CVPR 2025</sub> | image | 26.3 | 27.25 | 0.839 | 0.192 |
| 32 | **IR-SDE** <sub>whole, ICML 2023</sub> | image | 137.1 | 27.25 | 0.817 | 0.137 |
| 33 | **MemDeblur** <sub>clip, CVPR 2022</sub> | video | 7.0 | 27.24 | 0.838 | 0.210 |
| 34 | **ShuffleFormer** <sub>whole, ICML 2023</sub> | image | 50.6 | 27.24 | 0.837 | 0.197 |
| 35 | **F-MIMO+** <sub>tiled, large, AAAI 2023</sub> | image | 19.5 | 27.23 | 0.844 | 0.186 |
| 36 | **ShuffleFormer (1 pass)** <sub>whole, 1pass, ICML 2023</sub> | image | 50.6 | 27.22 | 0.836 | 0.199 |
| 37 | **Shift-Net+** <sub>clip, CVPR 2023</sub> | video | 13.0 | 27.22 | 0.852 | 0.170 |
| 38 | **Stripformer** <sub>whole, ECCV 2022</sub> | image | 19.7 | 27.18 | 0.838 | 0.176 |
| 39 | **Stripformer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 19.7 | 27.18 | 0.846 | 0.175 |
| 40 | **CDVD-TSP** <sub>window, CVPR 2020</sub> | video | 16.2 | 27.17 | 0.852 | 0.196 |
| 41 | **EAMamba** <sub>whole, ICCV 2025</sub> | image | 25.3 | 27.17 | 0.834 | 0.191 |
| 42 | **DeepRFT** <sub>TLC, arXiv 2021</sub> | image | 9.6 | 27.16 | 0.836 | 0.196 |
| 43 | **DeepRFT+** <sub>TLC, large, arXiv 2021</sub> | image | 23.0 | 27.12 | 0.838 | 0.184 |
| 44 | **MAXIM-3S** <sub>whole, CVPR 2022</sub> | image | 22.2 | 27.11 | 0.841 | 0.196 |
| 45 | **GRL-B** <sub>tiled, CVPR 2023</sub> | image | 19.8 | 27.10 | 0.832 | 0.194 |
| 46 | **X-Restormer** <sub>whole, ECCV 2024</sub> | image | 26.0 | 27.09 | 0.836 | 0.192 |
| 47 | **DSTNet+ L** <sub>clip, large, arXiv 2024</sub> | video | 14.1 | 27.07 | 0.853 | 0.157 |
| 48 | **F-MIMO** <sub>tiled, AAAI 2023</sub> | image | 8.2 | 27.06 | 0.831 | 0.209 |
| 49 | **LaKDNet-S** <sub>whole, small, arXiv 2023</sub> | image | 17.1 | 27.05 | 0.835 | 0.189 |
| 50 | **GRL-B** <sub>whole, CVPR 2023</sub> | image | 19.8 | 27.05 | 0.828 | 0.197 |
| 51 | **VDTR** <sub>window, TCSVT 2022</sub> | video | 23.1 | 27.05 | 0.846 | 0.180 |
| 52 | **LaKDNet-L** <sub>whole, arXiv 2023</sub> | image | 37.5 | 27.03 | 0.839 | 0.187 |
| 53 | **M3SNet-32** <sub>TLC, small, VisComp 2024</sub> | image | 16.7 | 27.03 | 0.830 | 0.214 |
| 54 | **F-MIMO-S** <sub>tiled, small, AAAI 2023</sub> | image | 4.4 | 27.01 | 0.826 | 0.218 |
| 55 | **DeblurDiNAT-L** <sub>whole, arXiv 2024</sub> | image | 16.1 | 27.00 | 0.842 | 0.176 |
| 56 | **TSRN-RTVD** <sub>recurrent, arXiv 2026</sub> | video | 4.9 | 26.99 | 0.830 | 0.218 |
| 57 | **DeepRFT-S** <sub>TLC, small, arXiv 2021</sub> | image | 5.1 | 26.97 | 0.825 | 0.212 |
| 58 | **M3SNet-32** <sub>whole, small, VisComp 2024</sub> | image | 16.7 | 26.96 | 0.822 | 0.220 |
| 59 | **DGUNet** <sub>whole, CVPR 2022</sub> | image | 17.3 | 26.96 | 0.821 | 0.212 |
| 60 | **SFNet** <sub>whole, ICLR 2023</sub> | image | 13.3 | 26.96 | 0.825 | 0.206 |
| 61 | **Concertormer-lite** <sub>TLC, small, arXiv 2025</sub> | image | 29.0 | 26.96 | 0.837 | 0.193 |
| 62 | **HI-Diff** <sub>whole, NeurIPS 2023</sub> | image | 28.5 | 26.96 | 0.829 | 0.203 |
| 63 | **Restormer** <sub>whole, CVPR 2022</sub> | image | 26.1 | 26.93 | 0.820 | 0.211 |
| 64 | **StarIR** <sub>TLC, 2025</sub> | image | 18.7 | 26.93 | 0.839 | 0.192 |
| 65 | **DGUNet+** <sub>whole, large, CVPR 2022</sub> | image | 69.6 | 26.92 | 0.822 | 0.206 |
| 66 | **M3SNet-64** <sub>TLC, VisComp 2024</sub> | image | 66.4 | 26.92 | 0.830 | 0.208 |
| 67 | **Concertormer-lite** <sub>whole, small, arXiv 2025</sub> | image | 29.0 | 26.92 | 0.827 | 0.202 |
| 68 | **EVSSM** <sub>whole, CVPR 2025</sub> | image | 17.1 | 26.91 | 0.848 | 0.178 |
| 69 | **Restormer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 26.1 | 26.90 | 0.824 | 0.206 |
| 70 | **ChaIR** <sub>whole, KBS 2023</sub> | image | 15.0 | 26.90 | 0.825 | 0.196 |
| 71 | **F-MIMO+** <sub>whole, large, AAAI 2023</sub> | image | 19.5 | 26.90 | 0.813 | 0.224 |
| 72 | **XYScanNet** <sub>whole, CVPRW 2025</sub> | image | 63.5 | 26.89 | 0.841 | 0.174 |
| 73 | **DeepRFT** <sub>whole, arXiv 2021</sub> | image | 9.6 | 26.87 | 0.809 | 0.229 |
| 74 | **HINet** <sub>whole, CVPRW 2021</sub> | image | 88.7 | 26.87 | 0.820 | 0.190 |
| 75 | **M3SNet-64** <sub>whole, VisComp 2024</sub> | image | 66.4 | 26.87 | 0.821 | 0.215 |
| 76 | **MPRNet** <sub>whole, CVPR 2021</sub> | image | 20.1 | 26.85 | 0.816 | 0.202 |
| 77 | **MSSNet-small** <sub>whole, small, ECCVW 2022</sub> | image | 6.8 | 26.85 | 0.828 | 0.198 |
| 78 | **NAFNet-64** <sub>whole, ECCV 2022</sub> | image | 67.9 | 26.85 | 0.822 | 0.199 |
| 79 | **LoFormer-L** <sub>whole, ACMMM 2024</sub> | image | 49.0 | 26.84 | 0.824 | 0.202 |
| 80 | **ESTRNN** <sub>recurrent, ECCV 2020</sub> | video | 2.5 | 26.84 | 0.817 | 0.246 |
| 81 | **SFNet** <sub>TLC, ICLR 2023</sub> | image | 13.3 | 26.83 | 0.826 | 0.205 |
| 82 | **MT-RNN** <sub>whole, ECCV 2020</sub> | image | 2.6 | 26.83 | 0.814 | 0.229 |
| 83 | **DeepRFT+** <sub>whole, large, arXiv 2021</sub> | image | 23.0 | 26.82 | 0.809 | 0.226 |
| 84 | **ConvIR-L** <sub>whole, TPAMI 2024</sub> | image | 14.8 | 26.81 | 0.826 | 0.210 |
| 85 | **StarIR** <sub>whole, 2025</sub> | image | 18.7 | 26.81 | 0.827 | 0.203 |
| 86 | **IRNeXt (16 blocks)** <sub>whole, large, ICML 2023</sub> | image | 14.8 | 26.81 | 0.824 | 0.206 |
| 87 | **IRNeXt** <sub>whole, ICML 2023</sub> | image | 13.2 | 26.79 | 0.820 | 0.211 |
| 88 | **PSS-NSC** <sub>whole, CVPR 2019</sub> | image | 3.9 | 26.78 | 0.830 | 0.213 |
| 89 | **DiffIR** <sub>whole, ICCV 2023</sub> | image | 26.9 | 26.78 | 0.816 | 0.206 |
| 90 | **FFTformer** <sub>whole, CVPR 2023</sub> | image | 16.6 | 26.78 | 0.832 | 0.190 |
| 91 | **DeepRFT-S** <sub>whole, small, arXiv 2021</sub> | image | 5.1 | 26.76 | 0.803 | 0.240 |
| 92 | **F-MIMO-S** <sub>whole, small, AAAI 2023</sub> | image | 4.4 | 26.75 | 0.801 | 0.242 |
| 93 | **F-MIMO** <sub>whole, AAAI 2023</sub> | image | 8.2 | 26.75 | 0.803 | 0.238 |
| 94 | **MSDI-Net** <sub>tiled, ECCV 2022</sub> | image | 246.9 | 26.72 | 0.821 | 0.192 |
| 95 | **FSNet** <sub>whole, TPAMI 2023</sub> | image | 13.3 | 26.70 | 0.822 | 0.207 |
| 96 | **MLWNet-B** <sub>whole, CVPR 2024</sub> | image | 95.1 | 26.68 | 0.822 | 0.199 |
| 97 | **NAFNet-64** <sub>TLC, ECCV 2022</sub> | image | 67.9 | 26.66 | 0.829 | 0.197 |
| 98 | **MSSNet** <sub>whole, ECCVW 2022</sub> | image | 15.6 | 26.63 | 0.826 | 0.198 |
| 99 | **SIUN** <sub>whole, Access 2020</sub> | image | 6.3 | 26.61 | 0.830 | 0.187 |
| 100 | **MSDI-Net** <sub>whole, ECCV 2022</sub> | image | 246.9 | 26.61 | 0.808 | 0.196 |
| 101 | **F-NAFNet-32** <sub>whole, small, AAAI 2023</sub> | image | 17.8 | 26.48 | 0.834 | 0.193 |
| 102 | **MIMO-UNet+** <sub>whole, large, ICCV 2021</sub> | image | 16.1 | 26.42 | 0.810 | 0.214 |
| 103 | **MIMO-UNet** <sub>whole, ICCV 2021</sub> | image | 6.8 | 26.31 | 0.802 | 0.218 |
| 104 | **Motion-ETR** <sub>whole, TPAMI 2021</sub> | image | 6.6 | 26.25 | 0.801 | 0.232 |
| 105 | **NAFNet-32** <sub>whole, small, ECCV 2022</sub> | image | 17.1 | 25.97 | 0.796 | 0.242 |
| 106 | **Ghost-DeblurGAN** <sub>whole, IROS 2022</sub> | image | 1.5 | 25.93 | 0.779 | 0.246 |
| 107 | **NAFNet-32** <sub>TLC, small, ECCV 2022</sub> | image | 17.1 | 25.88 | 0.819 | 0.219 |
| 108 | **STDANet++** <sub>clip, whole, TCSVT 2025</sub> | video | 11.5 | 25.85 | 0.810 | 0.226 |
| 109 | **DMPHN-1-2-4** <sub>whole, CVPR 2019</sub> | image | 5.4 | 25.71 | 0.779 | 0.249 |
| 110 | **F-NAFNet-64** <sub>whole, AAAI 2023</sub> | image | 68.0 | 25.38 | 0.833 | 0.191 |
| 111 | **FGST** <sub>window, ICML 2022</sub> | video | 9.7 | 25.33 | 0.838 | 0.184 |
| 112 | **BSSTNet** <sub>clip, CVPR 2024</sub> | video | 46.7 | 24.39 | 0.821 | 0.203 |
| 113 | **FFTformer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 16.6 | 24.37 | 0.782 | 0.261 |
| 114 | **MLWNet-B** <sub>TLC, CVPR 2024</sub> | image | 95.1 | 23.98 | 0.754 | 0.296 |
| 115 | **DBGAN** <sub>whole, CVPR 2020</sub> | image | 11.6 | 22.95 | 0.718 | 0.320 |
| 116 | **MIMO-UNet+ + ID-Blau** <sub>whole, large, CVPR 2024</sub> | image | 16.1 | 19.64 | 0.638 | 0.418 |

</details>

### BSD

| # | Method | Type | Params, M | PSNR | SSIM | LPIPS |
|---:|---|---|---:|---:|---:|---:|
| 1 | **RevD-S** <sub>whole, small, CVPR 2024</sub> | image | 108.2 | 29.50 | 0.887 | 0.133 |
| 2 | **AdaRevD-B** <sub>whole, CVPR 2024</sub> | image | 152.4 | 29.49 | 0.888 | 0.132 |
| 3 | **AdaRevD-B** <sub>tiled, CVPR 2024</sub> | image | 152.4 | 29.40 | 0.886 | 0.131 |
| 4 | **UFPNet** <sub>whole, CVPR 2023</sub> | image | 80.3 | 29.36 | 0.889 | 0.137 |
| 5 | **UFPNet** <sub>TLC, CVPR 2023</sub> | image | 80.3 | 29.26 | 0.888 | 0.137 |
| 6 | **DeepRFT+** <sub>whole, large, arXiv 2021</sub> | image | 23.0 | 28.89 | 0.892 | 0.122 |
| 7 | **DeepRFT+** <sub>TLC, large, arXiv 2021</sub> | image | 23.0 | 28.73 | 0.889 | 0.120 |
| 8 | **ShuffleFormer** <sub>whole, ICML 2023</sub> | image | 50.6 | 28.73 | 0.892 | 0.111 |
| 9 | **ShuffleFormer (1 pass)** <sub>whole, 1pass, ICML 2023</sub> | image | 50.6 | 28.72 | 0.892 | 0.112 |
| 10 | **ShuffleFormer** <sub>tiled, ICML 2023</sub> | image | 50.6 | 28.72 | 0.892 | 0.111 |

<details>
<summary>All 116 rows on BSD</summary>

| # | Method | Type | Params, M | PSNR | SSIM | LPIPS |
|---:|---|---|---:|---:|---:|---:|
| 1 | **RevD-S** <sub>whole, small, CVPR 2024</sub> | image | 108.2 | 29.50 | 0.887 | 0.133 |
| 2 | **AdaRevD-B** <sub>whole, CVPR 2024</sub> | image | 152.4 | 29.49 | 0.888 | 0.132 |
| 3 | **AdaRevD-B** <sub>tiled, CVPR 2024</sub> | image | 152.4 | 29.40 | 0.886 | 0.131 |
| 4 | **UFPNet** <sub>whole, CVPR 2023</sub> | image | 80.3 | 29.36 | 0.889 | 0.137 |
| 5 | **UFPNet** <sub>TLC, CVPR 2023</sub> | image | 80.3 | 29.26 | 0.888 | 0.137 |
| 6 | **DeepRFT+** <sub>whole, large, arXiv 2021</sub> | image | 23.0 | 28.89 | 0.892 | 0.122 |
| 7 | **DeepRFT+** <sub>TLC, large, arXiv 2021</sub> | image | 23.0 | 28.73 | 0.889 | 0.120 |
| 8 | **ShuffleFormer** <sub>whole, ICML 2023</sub> | image | 50.6 | 28.73 | 0.892 | 0.111 |
| 9 | **ShuffleFormer (1 pass)** <sub>whole, 1pass, ICML 2023</sub> | image | 50.6 | 28.72 | 0.892 | 0.112 |
| 10 | **ShuffleFormer** <sub>tiled, ICML 2023</sub> | image | 50.6 | 28.72 | 0.892 | 0.111 |
| 11 | **MSSNet-small** <sub>whole, small, ECCVW 2022</sub> | image | 6.8 | 28.71 | 0.886 | 0.120 |
| 12 | **DeepRFT** <sub>whole, arXiv 2021</sub> | image | 9.6 | 28.69 | 0.888 | 0.131 |
| 13 | **F-MIMO+** <sub>whole, large, AAAI 2023</sub> | image | 19.5 | 28.64 | 0.885 | 0.142 |
| 14 | **MSSNet** <sub>whole, ECCVW 2022</sub> | image | 15.6 | 28.61 | 0.887 | 0.117 |
| 15 | **DeepRFT** <sub>TLC, arXiv 2021</sub> | image | 9.6 | 28.61 | 0.887 | 0.128 |
| 16 | **F-MIMO+** <sub>tiled, large, AAAI 2023</sub> | image | 19.5 | 28.50 | 0.885 | 0.139 |
| 17 | **SIUN** <sub>whole, Access 2020</sub> | image | 6.3 | 28.33 | 0.883 | 0.132 |
| 18 | **MAXIM-3S** <sub>whole, CVPR 2022</sub> | image | 22.2 | 28.33 | 0.882 | 0.131 |
| 19 | **Restormer** <sub>whole, CVPR 2022</sub> | image | 26.1 | 28.32 | 0.883 | 0.125 |
| 20 | **MaIR** <sub>whole, CVPR 2025</sub> | image | 26.3 | 28.20 | 0.876 | 0.141 |
| 21 | **DGUNet+** <sub>whole, large, CVPR 2022</sub> | image | 69.6 | 28.12 | 0.873 | 0.143 |
| 22 | **Stoformer** <sub>whole, NeurIPS 2022</sub> | image | 50.5 | 28.09 | 0.869 | 0.147 |
| 23 | **Stoformer** <sub>tiled, NeurIPS 2022</sub> | image | 50.5 | 28.07 | 0.869 | 0.149 |
| 24 | **DGUNet** <sub>whole, CVPR 2022</sub> | image | 17.3 | 28.05 | 0.873 | 0.146 |
| 25 | **MSFS-Net** <sub>whole, TCSVT 2023</sub> | image | 21.6 | 28.04 | 0.870 | 0.148 |
| 26 | **Stripformer** <sub>whole, ECCV 2022</sub> | image | 19.7 | 28.01 | 0.860 | 0.159 |
| 27 | **MSDI-Net** <sub>whole, ECCV 2022</sub> | image | 246.9 | 27.98 | 0.867 | 0.140 |
| 28 | **MSFS-Net** <sub>TLC, TCSVT 2023</sub> | image | 21.6 | 27.95 | 0.868 | 0.150 |
| 29 | **MSDI-Net** <sub>tiled, ECCV 2022</sub> | image | 246.9 | 27.94 | 0.866 | 0.141 |
| 30 | **DeblurDiNAT-L** <sub>whole, arXiv 2024</sub> | image | 16.1 | 27.92 | 0.867 | 0.141 |
| 31 | **Turtle** <sub>tiled, NeurIPS 2024</sub> | video | 59.1 | 27.84 | 0.866 | 0.160 |
| 32 | **F-MIMO** <sub>whole, AAAI 2023</sub> | image | 8.2 | 27.83 | 0.867 | 0.170 |
| 33 | **LaKDNet-S** <sub>whole, small, arXiv 2023</sub> | image | 17.1 | 27.81 | 0.865 | 0.132 |
| 34 | **Stripformer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 19.7 | 27.75 | 0.857 | 0.155 |
| 35 | **EAMamba** <sub>whole, ICCV 2025</sub> | image | 25.3 | 27.75 | 0.858 | 0.164 |
| 36 | **LoFormer-L** <sub>whole, ACMMM 2024</sub> | image | 49.0 | 27.70 | 0.861 | 0.144 |
| 37 | **HI-Diff** <sub>whole, NeurIPS 2023</sub> | image | 28.5 | 27.68 | 0.875 | 0.138 |
| 38 | **F-MIMO-S** <sub>whole, small, AAAI 2023</sub> | image | 4.4 | 27.67 | 0.862 | 0.181 |
| 39 | **M3SNet-32** <sub>whole, small, VisComp 2024</sub> | image | 16.7 | 27.65 | 0.864 | 0.145 |
| 40 | **F-MIMO** <sub>tiled, AAAI 2023</sub> | image | 8.2 | 27.64 | 0.864 | 0.170 |
| 41 | **MPRNet** <sub>whole, CVPR 2021</sub> | image | 20.1 | 27.55 | 0.856 | 0.167 |
| 42 | **DeepRFT-S** <sub>whole, small, arXiv 2021</sub> | image | 5.1 | 27.55 | 0.868 | 0.174 |
| 43 | **SFNet** <sub>whole, ICLR 2023</sub> | image | 13.3 | 27.50 | 0.856 | 0.150 |
| 44 | **F-MIMO-S** <sub>tiled, small, AAAI 2023</sub> | image | 4.4 | 27.48 | 0.860 | 0.182 |
| 45 | **RT-Focuser** <sub>whole, arXiv 2025</sub> | image | 5.9 | 27.47 | 0.839 | 0.185 |
| 46 | **PSS-NSC** <sub>whole, CVPR 2019</sub> | image | 3.9 | 27.46 | 0.857 | 0.168 |
| 47 | **HINet** <sub>whole, CVPRW 2021</sub> | image | 88.7 | 27.41 | 0.835 | 0.210 |
| 48 | **Restormer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 26.1 | 27.38 | 0.849 | 0.182 |
| 49 | **ChaIR** <sub>whole, KBS 2023</sub> | image | 15.0 | 27.36 | 0.855 | 0.150 |
| 50 | **F-NAFNet-32** <sub>whole, small, AAAI 2023</sub> | image | 17.8 | 27.33 | 0.854 | 0.146 |
| 51 | **IRNeXt** <sub>whole, ICML 2023</sub> | image | 13.2 | 27.33 | 0.850 | 0.171 |
| 52 | **Concertormer** <sub>whole, arXiv 2025</sub> | image | 50.5 | 27.29 | 0.854 | 0.172 |
| 53 | **DiffIR** <sub>whole, ICCV 2023</sub> | image | 26.9 | 27.29 | 0.846 | 0.177 |
| 54 | **Concertormer** <sub>TLC, arXiv 2025</sub> | image | 50.5 | 27.28 | 0.854 | 0.171 |
| 55 | **M3SNet-32** <sub>TLC, small, VisComp 2024</sub> | image | 16.7 | 27.27 | 0.854 | 0.155 |
| 56 | **DeepRFT-S** <sub>TLC, small, arXiv 2021</sub> | image | 5.1 | 27.27 | 0.863 | 0.175 |
| 57 | **SFNet** <sub>TLC, ICLR 2023</sub> | image | 13.3 | 27.27 | 0.846 | 0.157 |
| 58 | **GRL-B** <sub>tiled, CVPR 2023</sub> | image | 19.8 | 27.23 | 0.845 | 0.188 |
| 59 | **GRL-B** <sub>whole, CVPR 2023</sub> | image | 19.8 | 27.22 | 0.845 | 0.187 |
| 60 | **M3SNet-64** <sub>whole, VisComp 2024</sub> | image | 66.4 | 27.14 | 0.853 | 0.166 |
| 61 | **Ghost-DeblurGAN** <sub>whole, IROS 2022</sub> | image | 1.5 | 27.13 | 0.848 | 0.181 |
| 62 | **M3SNet-64** <sub>TLC, VisComp 2024</sub> | image | 66.4 | 27.04 | 0.849 | 0.168 |
| 63 | **MT-RNN** <sub>whole, ECCV 2020</sub> | image | 2.6 | 26.94 | 0.843 | 0.186 |
| 64 | **IRNeXt (16 blocks)** <sub>whole, large, ICML 2023</sub> | image | 14.8 | 26.87 | 0.846 | 0.171 |
| 65 | **FSNet** <sub>whole, TPAMI 2023</sub> | image | 13.3 | 26.87 | 0.833 | 0.167 |
| 66 | **CDVD-TSP** <sub>window, CVPR 2020</sub> | video | 16.2 | 26.81 | 0.837 | 0.189 |
| 67 | **X-Restormer** <sub>whole, ECCV 2024</sub> | image | 26.0 | 26.75 | 0.841 | 0.187 |
| 68 | **Concertormer-lite** <sub>TLC, small, arXiv 2025</sub> | image | 29.0 | 26.74 | 0.846 | 0.168 |
| 69 | **TSRN-RTVD** <sub>recurrent, arXiv 2026</sub> | video | 4.9 | 26.74 | 0.842 | 0.182 |
| 70 | **Concertormer-lite** <sub>whole, small, arXiv 2025</sub> | image | 29.0 | 26.73 | 0.847 | 0.167 |
| 71 | **LaKDNet-L** <sub>whole, arXiv 2023</sub> | image | 37.5 | 26.63 | 0.837 | 0.183 |
| 72 | **StarIR** <sub>whole, 2025</sub> | image | 18.7 | 26.50 | 0.843 | 0.185 |
| 73 | **F-NAFNet-64** <sub>whole, AAAI 2023</sub> | image | 68.0 | 26.25 | 0.840 | 0.170 |
| 74 | **StarIR** <sub>TLC, 2025</sub> | image | 18.7 | 25.95 | 0.835 | 0.189 |
| 75 | **ConvIR-L** <sub>whole, TPAMI 2024</sub> | image | 14.8 | 25.85 | 0.824 | 0.201 |
| 76 | **Motion-ETR** <sub>whole, TPAMI 2021</sub> | image | 6.6 | 25.69 | 0.813 | 0.194 |
| 77 | **NAFNet-32** <sub>whole, small, ECCV 2022</sub> | image | 17.1 | 25.65 | 0.805 | 0.228 |
| 78 | **NAFNet-64** <sub>whole, ECCV 2022</sub> | image | 67.9 | 25.56 | 0.813 | 0.205 |
| 79 | **MLWNet-B** <sub>whole, CVPR 2024</sub> | image | 95.1 | 25.43 | 0.797 | 0.233 |
| 80 | **CDVD-TSPNL** <sub>clip, TPAMI 2023</sub> | video | 5.5 | 25.40 | 0.789 | 0.229 |
| 81 | **NAFNet-64** <sub>TLC, ECCV 2022</sub> | image | 67.9 | 25.34 | 0.805 | 0.213 |
| 82 | **MIMO-UNet** <sub>whole, ICCV 2021</sub> | image | 6.8 | 25.20 | 0.804 | 0.220 |
| 83 | **DMPHN-1-2-4** <sub>whole, CVPR 2019</sub> | image | 5.4 | 25.09 | 0.792 | 0.239 |
| 84 | **DeepDeblur** <sub>whole, CVPR 2017</sub> | image | 11.7 | 25.05 | 0.777 | 0.236 |
| 85 | **MLWNet-B** <sub>TLC, CVPR 2024</sub> | image | 95.1 | 25.00 | 0.782 | 0.254 |
| 86 | **RVRT** <sub>clip, NeurIPS 2022</sub> | video | 13.6 | 24.98 | 0.801 | 0.219 |
| 87 | **NAFNet-32** <sub>TLC, small, ECCV 2022</sub> | image | 17.1 | 24.78 | 0.778 | 0.263 |
| 88 | **MIMO-UNet+** <sub>whole, large, ICCV 2021</sub> | image | 16.1 | 24.41 | 0.793 | 0.225 |
| 89 | **ESTRNN** <sub>recurrent, ECCV 2020</sub> | video | 2.5 | 24.38 | 0.774 | 0.262 |
| 90 | **EVSSM** <sub>whole, CVPR 2025</sub> | image | 17.1 | 24.31 | 0.801 | 0.239 |
| 91 | **FFTformer** <sub>whole, CVPR 2023</sub> | image | 16.6 | 23.98 | 0.793 | 0.249 |
| 92 | **VDTR** <sub>window, TCSVT 2022</sub> | video | 23.1 | 23.87 | 0.777 | 0.237 |
| 93 | **Shift-Net+** <sub>clip, CVPR 2023</sub> | video | 13.0 | 23.79 | 0.771 | 0.246 |
| 94 | **BSSTNet** <sub>clip, CVPR 2024</sub> | video | 46.7 | 23.76 | 0.761 | 0.261 |
| 95 | **MVSSM-L** <sub>clip, large, CVPR-F 2026</sub> | video | 30.2 | 23.71 | 0.761 | 0.266 |
| 96 | **MVSSM** <sub>clip, CVPR-F 2026</sub> | video | 12.8 | 23.41 | 0.754 | 0.278 |
| 97 | **DSTNet+ base** <sub>clip, arXiv 2024</sub> | video | 3.9 | 23.41 | 0.761 | 0.279 |
| 98 | **STDANet++** <sub>clip, tiled, TCSVT 2025</sub> | video | 11.5 | 23.32 | 0.752 | 0.265 |
| 99 | **DSTNet+ L** <sub>clip, large, arXiv 2024</sub> | video | 14.1 | 23.25 | 0.762 | 0.261 |
| 100 | **XYScanNet** <sub>whole, CVPRW 2025</sub> | image | 63.5 | 23.24 | 0.762 | 0.276 |
| 101 | **DSTNet** <sub>clip, CVPR 2023</sub> | video | 7.5 | 23.24 | 0.753 | 0.291 |
| 102 | **PVDNet** <sub>window+prev out, TOG 2021</sub> | video | 10.5 | 23.07 | 0.731 | 0.284 |
| 103 | **MemDeblur** <sub>clip, CVPR 2022</sub> | video | 7.0 | 23.06 | 0.742 | 0.279 |
| 104 | **VRT** <sub>clip, TIP 2022</sub> | video | 18.3 | 22.90 | 0.727 | 0.275 |
| 105 | **BasicVSR++** <sub>clip, CVPR 2022</sub> | video | 9.8 | 22.78 | 0.736 | 0.296 |
| 106 | **STDANet++** <sub>clip, whole, TCSVT 2025</sub> | video | 11.5 | 22.72 | 0.735 | 0.305 |
| 107 | **RT-VDBLR** <sub>recurrent, CGF 2022</sub> | video | 1.0 | 22.59 | 0.728 | 0.304 |
| 108 | **FFTformer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 16.6 | 22.58 | 0.753 | 0.300 |
| 109 | **VRNN** <sub>clip, PR 2023</sub> | video | 5.0 | 21.71 | 0.678 | 0.350 |
| 110 | **FGST** <sub>window, ICML 2022</sub> | video | 9.7 | 21.65 | 0.686 | 0.303 |
| 111 | **DBGAN** <sub>whole, CVPR 2020</sub> | image | 11.6 | 21.54 | 0.697 | 0.310 |
| 112 | **MMP-RNN** <sub>recurrent, ECCV 2022</sub> | video | 4.1 | 21.36 | 0.695 | 0.353 |
| 113 | **STDAN** <sub>window, ECCV 2022</sub> | video | 13.8 | 20.93 | 0.679 | 0.349 |
| 114 | **RNN-MBP** <sub>clip, AAAI 2022</sub> | video | 16.4 | 20.75 | 0.652 | 0.361 |
| 115 | **IR-SDE** <sub>whole, ICML 2023</sub> | image | 137.1 | 19.59 | 0.565 | 0.393 |
| 116 | **MIMO-UNet+ + ID-Blau** <sub>whole, large, CVPR 2024</sub> | image | 16.1 | 19.37 | 0.669 | 0.375 |

</details>

### DVD

| # | Method | Type | Params, M | PSNR | SSIM | LPIPS |
|---:|---|---|---:|---:|---:|---:|
| 1 | **BSSTNet** <sub>clip, CVPR 2024</sub> | video | 46.7 | 32.68 | 0.934 | 0.079 |
| 2 | **RVRT** <sub>clip, NeurIPS 2022</sub> | video | 13.6 | 32.60 | 0.934 | 0.081 |
| 3 | **VRT** <sub>clip, TIP 2022</sub> | video | 18.3 | 32.34 | 0.929 | 0.082 |
| 4 | **STDANet++** <sub>clip, tiled, TCSVT 2025</sub> | video | 11.5 | 32.32 | 0.931 | 0.080 |
| 5 | **BasicVSR++** <sub>clip, CVPR 2022</sub> | video | 9.8 | 32.13 | 0.931 | 0.080 |
| 6 | **DSTNet+ L** <sub>clip, large, arXiv 2024</sub> | video | 14.1 | 32.03 | 0.923 | 0.084 |
| 7 | **Shift-Net+** <sub>clip, CVPR 2023</sub> | video | 13.0 | 31.81 | 0.919 | 0.097 |
| 8 | **CDVD-TSPNL** <sub>clip, TPAMI 2023</sub> | video | 5.5 | 31.81 | 0.923 | 0.097 |
| 9 | **DSTNet+ base** <sub>clip, arXiv 2024</sub> | video | 3.9 | 31.62 | 0.917 | 0.092 |
| 10 | **DSTNet** <sub>clip, CVPR 2023</sub> | video | 7.5 | 31.38 | 0.913 | 0.087 |

<details>
<summary>All 116 rows on DVD</summary>

| # | Method | Type | Params, M | PSNR | SSIM | LPIPS |
|---:|---|---|---:|---:|---:|---:|
| 1 | **BSSTNet** <sub>clip, CVPR 2024</sub> | video | 46.7 | 32.68 | 0.934 | 0.079 |
| 2 | **RVRT** <sub>clip, NeurIPS 2022</sub> | video | 13.6 | 32.60 | 0.934 | 0.081 |
| 3 | **VRT** <sub>clip, TIP 2022</sub> | video | 18.3 | 32.34 | 0.929 | 0.082 |
| 4 | **STDANet++** <sub>clip, tiled, TCSVT 2025</sub> | video | 11.5 | 32.32 | 0.931 | 0.080 |
| 5 | **BasicVSR++** <sub>clip, CVPR 2022</sub> | video | 9.8 | 32.13 | 0.931 | 0.080 |
| 6 | **DSTNet+ L** <sub>clip, large, arXiv 2024</sub> | video | 14.1 | 32.03 | 0.923 | 0.084 |
| 7 | **Shift-Net+** <sub>clip, CVPR 2023</sub> | video | 13.0 | 31.81 | 0.919 | 0.097 |
| 8 | **CDVD-TSPNL** <sub>clip, TPAMI 2023</sub> | video | 5.5 | 31.81 | 0.923 | 0.097 |
| 9 | **DSTNet+ base** <sub>clip, arXiv 2024</sub> | video | 3.9 | 31.62 | 0.917 | 0.092 |
| 10 | **DSTNet** <sub>clip, CVPR 2023</sub> | video | 7.5 | 31.38 | 0.913 | 0.087 |
| 11 | **MVSSM-L** <sub>clip, large, CVPR-F 2026</sub> | video | 30.2 | 31.19 | 0.908 | 0.095 |
| 12 | **Turtle** <sub>tiled, NeurIPS 2024</sub> | video | 59.1 | 31.18 | 0.909 | 0.114 |
| 13 | **VDTR** <sub>window, TCSVT 2022</sub> | video | 23.1 | 30.91 | 0.903 | 0.100 |
| 14 | **MVSSM** <sub>clip, CVPR-F 2026</sub> | video | 12.8 | 30.88 | 0.902 | 0.101 |
| 15 | **CDVD-TSP** <sub>window, CVPR 2020</sub> | video | 16.2 | 30.86 | 0.911 | 0.118 |
| 16 | **ShuffleFormer** <sub>tiled, ICML 2023</sub> | image | 50.6 | 30.45 | 0.898 | 0.127 |
| 17 | **DeblurDiNAT-L** <sub>whole, arXiv 2024</sub> | image | 16.1 | 30.39 | 0.896 | 0.118 |
| 18 | **MSFS-Net** <sub>TLC, TCSVT 2023</sub> | image | 21.6 | 30.38 | 0.896 | 0.123 |
| 19 | **AdaRevD-B** <sub>tiled, CVPR 2024</sub> | image | 152.4 | 30.36 | 0.895 | 0.139 |
| 20 | **F-MIMO+** <sub>tiled, large, AAAI 2023</sub> | image | 19.5 | 30.35 | 0.895 | 0.132 |
| 21 | **MemDeblur** <sub>clip, CVPR 2022</sub> | video | 7.0 | 30.34 | 0.893 | 0.122 |
| 22 | **HI-Diff** <sub>whole, NeurIPS 2023</sub> | image | 28.5 | 30.32 | 0.895 | 0.129 |
| 23 | **DeepRFT+** <sub>TLC, large, arXiv 2021</sub> | image | 23.0 | 30.32 | 0.895 | 0.126 |
| 24 | **MSFS-Net** <sub>whole, TCSVT 2023</sub> | image | 21.6 | 30.27 | 0.893 | 0.128 |
| 25 | **UFPNet** <sub>TLC, CVPR 2023</sub> | image | 80.3 | 30.27 | 0.893 | 0.137 |
| 26 | **M3SNet-32** <sub>TLC, small, VisComp 2024</sub> | image | 16.7 | 30.22 | 0.892 | 0.140 |
| 27 | **ShuffleFormer** <sub>whole, ICML 2023</sub> | image | 50.6 | 30.18 | 0.892 | 0.135 |
| 28 | **ShuffleFormer (1 pass)** <sub>whole, 1pass, ICML 2023</sub> | image | 50.6 | 30.15 | 0.892 | 0.136 |
| 29 | **MLWNet-B** <sub>TLC, CVPR 2024</sub> | image | 95.1 | 30.10 | 0.892 | 0.135 |
| 30 | **Stripformer** <sub>whole, ECCV 2022</sub> | image | 19.7 | 30.09 | 0.889 | 0.125 |
| 31 | **Stripformer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 19.7 | 30.07 | 0.892 | 0.124 |
| 32 | **DeepRFT** <sub>TLC, arXiv 2021</sub> | image | 9.6 | 30.07 | 0.890 | 0.132 |
| 33 | **Stoformer** <sub>whole, NeurIPS 2022</sub> | image | 50.5 | 30.03 | 0.889 | 0.137 |
| 34 | **Stoformer** <sub>tiled, NeurIPS 2022</sub> | image | 50.5 | 30.02 | 0.889 | 0.137 |
| 35 | **FSNet** <sub>whole, TPAMI 2023</sub> | image | 13.3 | 30.01 | 0.890 | 0.133 |
| 36 | **F-MIMO** <sub>tiled, AAAI 2023</sub> | image | 8.2 | 29.99 | 0.889 | 0.142 |
| 37 | **MSSNet** <sub>whole, ECCVW 2022</sub> | image | 15.6 | 29.97 | 0.890 | 0.132 |
| 38 | **UFPNet** <sub>whole, CVPR 2023</sub> | image | 80.3 | 29.94 | 0.887 | 0.146 |
| 39 | **F-MIMO-S** <sub>tiled, small, AAAI 2023</sub> | image | 4.4 | 29.94 | 0.887 | 0.141 |
| 40 | **AdaRevD-B** <sub>whole, CVPR 2024</sub> | image | 152.4 | 29.90 | 0.886 | 0.157 |
| 41 | **ESTRNN** <sub>recurrent, ECCV 2020</sub> | video | 2.5 | 29.90 | 0.885 | 0.148 |
| 42 | **M3SNet-32** <sub>whole, small, VisComp 2024</sub> | image | 16.7 | 29.90 | 0.885 | 0.148 |
| 43 | **MLWNet-B** <sub>whole, CVPR 2024</sub> | image | 95.1 | 29.89 | 0.887 | 0.141 |
| 44 | **RevD-S** <sub>whole, small, CVPR 2024</sub> | image | 108.2 | 29.88 | 0.885 | 0.155 |
| 45 | **TSRN-RTVD** <sub>recurrent, arXiv 2026</sub> | video | 4.9 | 29.88 | 0.885 | 0.131 |
| 46 | **DiffIR** <sub>whole, ICCV 2023</sub> | image | 26.9 | 29.88 | 0.884 | 0.139 |
| 47 | **MSSNet-small** <sub>whole, small, ECCVW 2022</sub> | image | 6.8 | 29.87 | 0.887 | 0.132 |
| 48 | **SFNet** <sub>TLC, ICLR 2023</sub> | image | 13.3 | 29.86 | 0.887 | 0.137 |
| 49 | **DeepRFT-S** <sub>TLC, small, arXiv 2021</sub> | image | 5.1 | 29.85 | 0.885 | 0.142 |
| 50 | **F-MIMO+** <sub>whole, large, AAAI 2023</sub> | image | 19.5 | 29.84 | 0.884 | 0.147 |
| 51 | **ConvIR-L** <sub>whole, TPAMI 2024</sub> | image | 14.8 | 29.83 | 0.886 | 0.139 |
| 52 | **X-Restormer** <sub>whole, ECCV 2024</sub> | image | 26.0 | 29.83 | 0.883 | 0.148 |
| 53 | **NAFNet-32** <sub>whole, small, ECCV 2022</sub> | image | 17.1 | 29.82 | 0.884 | 0.148 |
| 54 | **IRNeXt** <sub>whole, ICML 2023</sub> | image | 13.2 | 29.79 | 0.885 | 0.139 |
| 55 | **MSDI-Net** <sub>tiled, ECCV 2022</sub> | image | 246.9 | 29.79 | 0.883 | 0.140 |
| 56 | **DeepRFT+** <sub>whole, large, arXiv 2021</sub> | image | 23.0 | 29.79 | 0.882 | 0.143 |
| 57 | **GRL-B** <sub>tiled, CVPR 2023</sub> | image | 19.8 | 29.78 | 0.882 | 0.146 |
| 58 | **NAFNet-32** <sub>TLC, small, ECCV 2022</sub> | image | 17.1 | 29.78 | 0.885 | 0.142 |
| 59 | **F-NAFNet-32** <sub>whole, small, AAAI 2023</sub> | image | 17.8 | 29.75 | 0.885 | 0.140 |
| 60 | **PSS-NSC** <sub>whole, CVPR 2019</sub> | image | 3.9 | 29.74 | 0.884 | 0.139 |
| 61 | **IRNeXt (16 blocks)** <sub>whole, large, ICML 2023</sub> | image | 14.8 | 29.72 | 0.884 | 0.141 |
| 62 | **SIUN** <sub>whole, Access 2020</sub> | image | 6.3 | 29.71 | 0.884 | 0.121 |
| 63 | **MaIR** <sub>whole, CVPR 2025</sub> | image | 26.3 | 29.70 | 0.884 | 0.151 |
| 64 | **DGUNet** <sub>whole, CVPR 2022</sub> | image | 17.3 | 29.69 | 0.881 | 0.151 |
| 65 | **STDAN** <sub>window, ECCV 2022</sub> | video | 13.8 | 29.68 | 0.892 | 0.128 |
| 66 | **MPRNet** <sub>whole, CVPR 2021</sub> | image | 20.1 | 29.67 | 0.881 | 0.150 |
| 67 | **ChaIR** <sub>whole, KBS 2023</sub> | image | 15.0 | 29.65 | 0.880 | 0.142 |
| 68 | **Restormer** <sub>whole, CVPR 2022</sub> | image | 26.1 | 29.63 | 0.878 | 0.154 |
| 69 | **M3SNet-64** <sub>TLC, VisComp 2024</sub> | image | 66.4 | 29.63 | 0.873 | 0.166 |
| 70 | **NAFNet-64** <sub>TLC, ECCV 2022</sub> | image | 67.9 | 29.61 | 0.882 | 0.153 |
| 71 | **VRNN** <sub>clip, PR 2023</sub> | video | 5.0 | 29.61 | 0.889 | 0.109 |
| 72 | **GRL-B** <sub>whole, CVPR 2023</sub> | image | 19.8 | 29.61 | 0.878 | 0.150 |
| 73 | **DeepRFT** <sub>whole, arXiv 2021</sub> | image | 9.6 | 29.59 | 0.877 | 0.148 |
| 74 | **SFNet** <sub>whole, ICLR 2023</sub> | image | 13.3 | 29.58 | 0.881 | 0.144 |
| 75 | **MT-RNN** <sub>whole, ECCV 2020</sub> | image | 2.6 | 29.56 | 0.879 | 0.150 |
| 76 | **EAMamba** <sub>whole, ICCV 2025</sub> | image | 25.3 | 29.55 | 0.875 | 0.162 |
| 77 | **Concertormer** <sub>TLC, arXiv 2025</sub> | image | 50.5 | 29.55 | 0.878 | 0.156 |
| 78 | **PVDNet** <sub>window+prev out, TOG 2021</sub> | video | 10.5 | 29.51 | 0.889 | 0.129 |
| 79 | **MAXIM-3S** <sub>whole, CVPR 2022</sub> | image | 22.2 | 29.50 | 0.874 | 0.158 |
| 80 | **HINet** <sub>whole, CVPRW 2021</sub> | image | 88.7 | 29.49 | 0.877 | 0.159 |
| 81 | **LaKDNet-L** <sub>whole, arXiv 2023</sub> | image | 37.5 | 29.48 | 0.879 | 0.159 |
| 82 | **M3SNet-64** <sub>whole, VisComp 2024</sub> | image | 66.4 | 29.47 | 0.869 | 0.169 |
| 83 | **Concertormer-lite** <sub>TLC, small, arXiv 2025</sub> | image | 29.0 | 29.47 | 0.880 | 0.157 |
| 84 | **F-NAFNet-64** <sub>whole, AAAI 2023</sub> | image | 68.0 | 29.46 | 0.883 | 0.153 |
| 85 | **MMP-RNN** <sub>recurrent, ECCV 2022</sub> | video | 4.1 | 29.46 | 0.882 | 0.131 |
| 86 | **LaKDNet-S** <sub>whole, small, arXiv 2023</sub> | image | 17.1 | 29.44 | 0.880 | 0.156 |
| 87 | **LoFormer-L** <sub>whole, ACMMM 2024</sub> | image | 49.0 | 29.44 | 0.874 | 0.166 |
| 88 | **NAFNet-64** <sub>whole, ECCV 2022</sub> | image | 67.9 | 29.44 | 0.876 | 0.160 |
| 89 | **DeepRFT-S** <sub>whole, small, arXiv 2021</sub> | image | 5.1 | 29.43 | 0.874 | 0.161 |
| 90 | **F-MIMO-S** <sub>whole, small, AAAI 2023</sub> | image | 4.4 | 29.42 | 0.873 | 0.161 |
| 91 | **MSDI-Net** <sub>whole, ECCV 2022</sub> | image | 246.9 | 29.40 | 0.873 | 0.147 |
| 92 | **MIMO-UNet+** <sub>whole, large, ICCV 2021</sub> | image | 16.1 | 29.39 | 0.875 | 0.146 |
| 93 | **F-MIMO** <sub>whole, AAAI 2023</sub> | image | 8.2 | 29.35 | 0.872 | 0.163 |
| 94 | **Restormer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 26.1 | 29.33 | 0.868 | 0.161 |
| 95 | **Concertormer** <sub>whole, arXiv 2025</sub> | image | 50.5 | 29.27 | 0.872 | 0.163 |
| 96 | **RNN-MBP** <sub>clip, AAAI 2022</sub> | video | 16.4 | 29.25 | 0.878 | 0.123 |
| 97 | **Concertormer-lite** <sub>whole, small, arXiv 2025</sub> | image | 29.0 | 29.24 | 0.874 | 0.163 |
| 98 | **Motion-ETR** <sub>whole, TPAMI 2021</sub> | image | 6.6 | 29.20 | 0.871 | 0.154 |
| 99 | **DGUNet+** <sub>whole, large, CVPR 2022</sub> | image | 69.6 | 29.05 | 0.873 | 0.164 |
| 100 | **MIMO-UNet** <sub>whole, ICCV 2021</sub> | image | 6.8 | 28.91 | 0.861 | 0.162 |
| 101 | **RT-VDBLR** <sub>recurrent, CGF 2022</sub> | video | 1.0 | 28.85 | 0.868 | 0.155 |
| 102 | **DMPHN-1-2-4** <sub>whole, CVPR 2019</sub> | image | 5.4 | 28.54 | 0.855 | 0.174 |
| 103 | **XYScanNet** <sub>whole, CVPRW 2025</sub> | image | 63.5 | 28.45 | 0.851 | 0.172 |
| 104 | **StarIR** <sub>TLC, 2025</sub> | image | 18.7 | 28.45 | 0.856 | 0.187 |
| 105 | **Ghost-DeblurGAN** <sub>whole, IROS 2022</sub> | image | 1.5 | 28.27 | 0.844 | 0.163 |
| 106 | **RT-Focuser** <sub>whole, arXiv 2025</sub> | image | 5.9 | 28.26 | 0.853 | 0.179 |
| 107 | **DeepDeblur** <sub>whole, CVPR 2017</sub> | image | 11.7 | 28.24 | 0.857 | 0.168 |
| 108 | **EVSSM** <sub>whole, CVPR 2025</sub> | image | 17.1 | 28.19 | 0.864 | 0.171 |
| 109 | **StarIR** <sub>whole, 2025</sub> | image | 18.7 | 28.11 | 0.848 | 0.194 |
| 110 | **FFTformer** <sub>whole, CVPR 2023</sub> | image | 16.6 | 27.96 | 0.840 | 0.191 |
| 111 | **FFTformer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 16.6 | 27.94 | 0.836 | 0.195 |
| 112 | **FGST** <sub>window, ICML 2022</sub> | video | 9.7 | 27.91 | 0.873 | 0.151 |
| 113 | **DBGAN** <sub>whole, CVPR 2020</sub> | image | 11.6 | 27.78 | 0.833 | 0.204 |
| 114 | **IR-SDE** <sub>whole, ICML 2023</sub> | image | 137.1 | 26.77 | 0.799 | 0.190 |
| 115 | **STDANet++** <sub>clip, whole, TCSVT 2025</sub> | video | 11.5 | 25.79 | 0.824 | 0.223 |
| 116 | **MIMO-UNet+ + ID-Blau** <sub>whole, large, CVPR 2024</sub> | image | 16.1 | 24.53 | 0.789 | 0.240 |

</details>

### RealBlur-J

| # | Method | Type | Params, M | PSNR | SSIM | LPIPS |
|---:|---|---|---:|---:|---:|---:|
| 1 | **MSFS-Net** <sub>whole, TCSVT 2023</sub> | image | 21.6 | 30.73 | 0.902 | 0.133 |
| 2 | **MSFS-Net** <sub>TLC, TCSVT 2023</sub> | image | 21.6 | 30.68 | 0.902 | 0.133 |
| 3 | **UFPNet** <sub>whole, CVPR 2023</sub> | image | 80.3 | 30.38 | 0.883 | 0.145 |
| 4 | **UFPNet** <sub>TLC, CVPR 2023</sub> | image | 80.3 | 30.33 | 0.889 | 0.144 |
| 5 | **AdaRevD-B** <sub>whole, CVPR 2024</sub> | image | 152.4 | 30.14 | 0.889 | 0.151 |
| 6 | **AdaRevD-B** <sub>tiled, CVPR 2024</sub> | image | 152.4 | 30.12 | 0.894 | 0.143 |
| 7 | **RevD-S** <sub>whole, small, CVPR 2024</sub> | image | 108.2 | 30.11 | 0.888 | 0.149 |
| 8 | **ShuffleFormer** <sub>tiled, ICML 2023</sub> | image | 50.6 | 29.20 | 0.890 | 0.144 |
| 9 | **HI-Diff** <sub>whole, NeurIPS 2023</sub> | image | 28.5 | 29.15 | 0.890 | 0.147 |
| 10 | **MaIR** <sub>whole, CVPR 2025</sub> | image | 26.3 | 29.15 | 0.888 | 0.147 |

<details>
<summary>All 90 rows on RealBlur-J</summary>

| # | Method | Type | Params, M | PSNR | SSIM | LPIPS |
|---:|---|---|---:|---:|---:|---:|
| 1 | **MSFS-Net** <sub>whole, TCSVT 2023</sub> | image | 21.6 | 30.73 | 0.902 | 0.133 |
| 2 | **MSFS-Net** <sub>TLC, TCSVT 2023</sub> | image | 21.6 | 30.68 | 0.902 | 0.133 |
| 3 | **UFPNet** <sub>whole, CVPR 2023</sub> | image | 80.3 | 30.38 | 0.883 | 0.145 |
| 4 | **UFPNet** <sub>TLC, CVPR 2023</sub> | image | 80.3 | 30.33 | 0.889 | 0.144 |
| 5 | **AdaRevD-B** <sub>whole, CVPR 2024</sub> | image | 152.4 | 30.14 | 0.889 | 0.151 |
| 6 | **AdaRevD-B** <sub>tiled, CVPR 2024</sub> | image | 152.4 | 30.12 | 0.894 | 0.143 |
| 7 | **RevD-S** <sub>whole, small, CVPR 2024</sub> | image | 108.2 | 30.11 | 0.888 | 0.149 |
| 8 | **ShuffleFormer** <sub>tiled, ICML 2023</sub> | image | 50.6 | 29.20 | 0.890 | 0.144 |
| 9 | **HI-Diff** <sub>whole, NeurIPS 2023</sub> | image | 28.5 | 29.15 | 0.890 | 0.147 |
| 10 | **MaIR** <sub>whole, CVPR 2025</sub> | image | 26.3 | 29.15 | 0.888 | 0.147 |
| 11 | **ShuffleFormer** <sub>whole, ICML 2023</sub> | image | 50.6 | 29.14 | 0.888 | 0.146 |
| 12 | **ShuffleFormer (1 pass)** <sub>whole, 1pass, ICML 2023</sub> | image | 50.6 | 29.11 | 0.887 | 0.147 |
| 13 | **M3SNet-32** <sub>whole, small, VisComp 2024</sub> | image | 16.7 | 29.08 | 0.885 | 0.152 |
| 14 | **M3SNet-32** <sub>TLC, small, VisComp 2024</sub> | image | 16.7 | 29.06 | 0.884 | 0.152 |
| 15 | **DiffIR** <sub>whole, ICCV 2023</sub> | image | 26.9 | 29.05 | 0.882 | 0.152 |
| 16 | **DeblurDiNAT-L** <sub>whole, arXiv 2024</sub> | image | 16.1 | 28.98 | 0.885 | 0.142 |
| 17 | **M3SNet-64** <sub>whole, VisComp 2024</sub> | image | 66.4 | 28.98 | 0.881 | 0.151 |
| 18 | **DeepRFT+** <sub>TLC, large, arXiv 2021</sub> | image | 23.0 | 28.97 | 0.884 | 0.143 |
| 19 | **M3SNet-64** <sub>TLC, VisComp 2024</sub> | image | 66.4 | 28.97 | 0.881 | 0.150 |
| 20 | **Stripformer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 19.7 | 28.95 | 0.881 | 0.147 |
| 21 | **Restormer** <sub>whole, CVPR 2022</sub> | image | 26.1 | 28.95 | 0.878 | 0.157 |
| 22 | **DeepRFT** <sub>TLC, arXiv 2021</sub> | image | 9.6 | 28.90 | 0.880 | 0.153 |
| 23 | **X-Restormer** <sub>whole, ECCV 2024</sub> | image | 26.0 | 28.88 | 0.878 | 0.156 |
| 24 | **F-MIMO+** <sub>tiled, large, AAAI 2023</sub> | image | 19.5 | 28.88 | 0.880 | 0.153 |
| 25 | **DeepRFT** <sub>whole, arXiv 2021</sub> | image | 9.6 | 28.88 | 0.875 | 0.161 |
| 26 | **F-MIMO+** <sub>whole, large, AAAI 2023</sub> | image | 19.5 | 28.87 | 0.877 | 0.159 |
| 27 | **DeepRFT+** <sub>whole, large, arXiv 2021</sub> | image | 23.0 | 28.85 | 0.878 | 0.152 |
| 28 | **MAXIM-3S** <sub>whole, CVPR 2022</sub> | image | 22.2 | 28.85 | 0.876 | 0.154 |
| 29 | **Stripformer** <sub>whole, ECCV 2022</sub> | image | 19.7 | 28.82 | 0.876 | 0.146 |
| 30 | **Concertormer** <sub>TLC, arXiv 2025</sub> | image | 50.5 | 28.82 | 0.878 | 0.155 |
| 31 | **Concertormer** <sub>whole, arXiv 2025</sub> | image | 50.5 | 28.81 | 0.876 | 0.155 |
| 32 | **MSSNet** <sub>whole, ECCVW 2022</sub> | image | 15.6 | 28.78 | 0.879 | 0.154 |
| 33 | **F-NAFNet-32** <sub>whole, small, AAAI 2023</sub> | image | 17.8 | 28.77 | 0.877 | 0.156 |
| 34 | **LaKDNet-S** <sub>whole, small, arXiv 2023</sub> | image | 17.1 | 28.77 | 0.878 | 0.153 |
| 35 | **DGUNet** <sub>whole, CVPR 2022</sub> | image | 17.3 | 28.76 | 0.876 | 0.153 |
| 36 | **EAMamba** <sub>whole, ICCV 2025</sub> | image | 25.3 | 28.75 | 0.874 | 0.165 |
| 37 | **NAFNet-32** <sub>whole, small, ECCV 2022</sub> | image | 17.1 | 28.75 | 0.874 | 0.157 |
| 38 | **NAFNet-32** <sub>TLC, small, ECCV 2022</sub> | image | 17.1 | 28.75 | 0.875 | 0.158 |
| 39 | **F-MIMO** <sub>whole, AAAI 2023</sub> | image | 8.2 | 28.73 | 0.866 | 0.174 |
| 40 | **DGUNet+** <sub>whole, large, CVPR 2022</sub> | image | 69.6 | 28.72 | 0.873 | 0.154 |
| 41 | **F-MIMO** <sub>tiled, AAAI 2023</sub> | image | 8.2 | 28.72 | 0.871 | 0.169 |
| 42 | **SIUN** <sub>whole, Access 2020</sub> | image | 6.3 | 28.69 | 0.872 | 0.155 |
| 43 | **MPRNet** <sub>whole, CVPR 2021</sub> | image | 20.1 | 28.69 | 0.872 | 0.159 |
| 44 | **MSSNet-small** <sub>whole, small, ECCVW 2022</sub> | image | 6.8 | 28.68 | 0.874 | 0.158 |
| 45 | **SFNet** <sub>whole, ICLR 2023</sub> | image | 13.3 | 28.64 | 0.870 | 0.159 |
| 46 | **DeepRFT-S** <sub>whole, small, arXiv 2021</sub> | image | 5.1 | 28.64 | 0.866 | 0.175 |
| 47 | **Stoformer** <sub>tiled, NeurIPS 2022</sub> | image | 50.5 | 28.61 | 0.868 | 0.162 |
| 48 | **Stoformer** <sub>whole, NeurIPS 2022</sub> | image | 50.5 | 28.61 | 0.868 | 0.161 |
| 49 | **DeepRFT-S** <sub>TLC, small, arXiv 2021</sub> | image | 5.1 | 28.60 | 0.868 | 0.168 |
| 50 | **MSDI-Net** <sub>tiled, ECCV 2022</sub> | image | 246.9 | 28.59 | 0.870 | 0.158 |
| 51 | **LaKDNet-L** <sub>whole, arXiv 2023</sub> | image | 37.5 | 28.59 | 0.870 | 0.167 |
| 52 | **F-MIMO-S** <sub>tiled, small, AAAI 2023</sub> | image | 4.4 | 28.58 | 0.864 | 0.174 |
| 53 | **F-MIMO-S** <sub>whole, small, AAAI 2023</sub> | image | 4.4 | 28.53 | 0.859 | 0.178 |
| 54 | **LoFormer-L** <sub>whole, ACMMM 2024</sub> | image | 49.0 | 28.53 | 0.863 | 0.168 |
| 55 | **IRNeXt (16 blocks)** <sub>whole, large, ICML 2023</sub> | image | 14.8 | 28.48 | 0.867 | 0.164 |
| 56 | **MSDI-Net** <sub>whole, ECCV 2022</sub> | image | 246.9 | 28.47 | 0.861 | 0.164 |
| 57 | **FSNet** <sub>whole, TPAMI 2023</sub> | image | 13.3 | 28.47 | 0.868 | 0.162 |
| 58 | **IRNeXt** <sub>whole, ICML 2023</sub> | image | 13.2 | 28.46 | 0.865 | 0.166 |
| 59 | **StarIR** <sub>whole, 2025</sub> | image | 18.7 | 28.45 | 0.868 | 0.166 |
| 60 | **MT-RNN** <sub>whole, ECCV 2020</sub> | image | 2.6 | 28.44 | 0.862 | 0.177 |
| 61 | **GRL-B** <sub>whole, CVPR 2023</sub> | image | 19.8 | 28.43 | 0.860 | 0.168 |
| 62 | **GRL-B** <sub>tiled, CVPR 2023</sub> | image | 19.8 | 28.43 | 0.860 | 0.168 |
| 63 | **PSS-NSC** <sub>whole, CVPR 2019</sub> | image | 3.9 | 28.41 | 0.861 | 0.192 |
| 64 | **ConvIR-L** <sub>whole, TPAMI 2024</sub> | image | 14.8 | 28.40 | 0.866 | 0.171 |
| 65 | **SFNet** <sub>TLC, ICLR 2023</sub> | image | 13.3 | 28.39 | 0.863 | 0.167 |
| 66 | **RT-Focuser** <sub>whole, arXiv 2025</sub> | image | 5.9 | 28.38 | 0.836 | 0.200 |
| 67 | **XYScanNet** <sub>whole, CVPRW 2025</sub> | image | 63.5 | 28.33 | 0.866 | 0.160 |
| 68 | **Concertormer-lite** <sub>whole, small, arXiv 2025</sub> | image | 29.0 | 28.33 | 0.860 | 0.172 |
| 69 | **Ghost-DeblurGAN** <sub>whole, IROS 2022</sub> | image | 1.5 | 28.32 | 0.851 | 0.162 |
| 70 | **NAFNet-64** <sub>TLC, ECCV 2022</sub> | image | 67.9 | 28.32 | 0.857 | 0.169 |
| 71 | **StarIR** <sub>TLC, 2025</sub> | image | 18.7 | 28.29 | 0.868 | 0.166 |
| 72 | **Concertormer-lite** <sub>TLC, small, arXiv 2025</sub> | image | 29.0 | 28.29 | 0.860 | 0.171 |
| 73 | **NAFNet-64** <sub>whole, ECCV 2022</sub> | image | 67.9 | 28.18 | 0.849 | 0.174 |
| 74 | **HINet** <sub>whole, CVPRW 2021</sub> | image | 88.7 | 28.18 | 0.848 | 0.175 |
| 75 | **MLWNet-B** <sub>whole, CVPR 2024</sub> | image | 95.1 | 28.15 | 0.850 | 0.186 |
| 76 | **Restormer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 26.1 | 28.11 | 0.854 | 0.192 |
| 77 | **Motion-ETR** <sub>whole, TPAMI 2021</sub> | image | 6.6 | 28.09 | 0.851 | 0.184 |
| 78 | **ChaIR** <sub>whole, KBS 2023</sub> | image | 15.0 | 28.08 | 0.860 | 0.169 |
| 79 | **DeepDeblur** <sub>whole, CVPR 2017</sub> | image | 11.7 | 28.07 | 0.855 | 0.191 |
| 80 | **F-NAFNet-64** <sub>whole, AAAI 2023</sub> | image | 68.0 | 28.05 | 0.856 | 0.170 |
| 81 | **DMPHN-1-2-4** <sub>whole, CVPR 2019</sub> | image | 5.4 | 27.79 | 0.841 | 0.202 |
| 82 | **FFTformer** <sub>whole, CVPR 2023</sub> | image | 16.6 | 27.77 | 0.853 | 0.195 |
| 83 | **MIMO-UNet** <sub>whole, ICCV 2021</sub> | image | 6.8 | 27.76 | 0.837 | 0.192 |
| 84 | **MIMO-UNet+** <sub>whole, large, ICCV 2021</sub> | image | 16.1 | 27.64 | 0.837 | 0.203 |
| 85 | **EVSSM** <sub>whole, CVPR 2025</sub> | image | 17.1 | 27.51 | 0.853 | 0.184 |
| 86 | **MLWNet-B** <sub>TLC, CVPR 2024</sub> | image | 95.1 | 26.36 | 0.782 | 0.285 |
| 87 | **DBGAN** <sub>whole, CVPR 2020</sub> | image | 11.6 | 24.93 | 0.745 | 0.299 |
| 88 | **FFTformer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 16.6 | 24.85 | 0.778 | 0.279 |
| 89 | **IR-SDE** <sub>whole, ICML 2023</sub> | image | 137.1 | 24.15 | 0.727 | 0.271 |
| 90 | **MIMO-UNet+ + ID-Blau** <sub>whole, large, CVPR 2024</sub> | image | 16.1 | 21.41 | 0.647 | 0.413 |

</details>

### RealBlur-R

| # | Method | Type | Params, M | PSNR | SSIM | LPIPS |
|---:|---|---|---:|---:|---:|---:|
| 1 | **MSFS-Net** <sub>TLC, TCSVT 2023</sub> | image | 21.6 | 37.74 | 0.962 | 0.068 |
| 2 | **MSFS-Net** <sub>whole, TCSVT 2023</sub> | image | 21.6 | 37.68 | 0.962 | 0.070 |
| 3 | **UFPNet** <sub>TLC, CVPR 2023</sub> | image | 80.3 | 36.82 | 0.953 | 0.061 |
| 4 | **UFPNet** <sub>whole, CVPR 2023</sub> | image | 80.3 | 36.79 | 0.953 | 0.062 |
| 5 | **RevD-S** <sub>whole, small, CVPR 2024</sub> | image | 108.2 | 36.56 | 0.957 | 0.065 |
| 6 | **AdaRevD-B** <sub>tiled, CVPR 2024</sub> | image | 152.4 | 36.56 | 0.957 | 0.062 |
| 7 | **AdaRevD-B** <sub>whole, CVPR 2024</sub> | image | 152.4 | 36.54 | 0.957 | 0.065 |
| 8 | **ShuffleFormer** <sub>tiled, ICML 2023</sub> | image | 50.6 | 36.34 | 0.958 | 0.058 |
| 9 | **ShuffleFormer** <sub>whole, ICML 2023</sub> | image | 50.6 | 36.32 | 0.958 | 0.058 |
| 10 | **ShuffleFormer (1 pass)** <sub>whole, 1pass, ICML 2023</sub> | image | 50.6 | 36.31 | 0.958 | 0.059 |

<details>
<summary>All 90 rows on RealBlur-R</summary>

| # | Method | Type | Params, M | PSNR | SSIM | LPIPS |
|---:|---|---|---:|---:|---:|---:|
| 1 | **MSFS-Net** <sub>TLC, TCSVT 2023</sub> | image | 21.6 | 37.74 | 0.962 | 0.068 |
| 2 | **MSFS-Net** <sub>whole, TCSVT 2023</sub> | image | 21.6 | 37.68 | 0.962 | 0.070 |
| 3 | **UFPNet** <sub>TLC, CVPR 2023</sub> | image | 80.3 | 36.82 | 0.953 | 0.061 |
| 4 | **UFPNet** <sub>whole, CVPR 2023</sub> | image | 80.3 | 36.79 | 0.953 | 0.062 |
| 5 | **RevD-S** <sub>whole, small, CVPR 2024</sub> | image | 108.2 | 36.56 | 0.957 | 0.065 |
| 6 | **AdaRevD-B** <sub>tiled, CVPR 2024</sub> | image | 152.4 | 36.56 | 0.957 | 0.062 |
| 7 | **AdaRevD-B** <sub>whole, CVPR 2024</sub> | image | 152.4 | 36.54 | 0.957 | 0.065 |
| 8 | **ShuffleFormer** <sub>tiled, ICML 2023</sub> | image | 50.6 | 36.34 | 0.958 | 0.058 |
| 9 | **ShuffleFormer** <sub>whole, ICML 2023</sub> | image | 50.6 | 36.32 | 0.958 | 0.058 |
| 10 | **ShuffleFormer (1 pass)** <sub>whole, 1pass, ICML 2023</sub> | image | 50.6 | 36.31 | 0.958 | 0.059 |
| 11 | **MaIR** <sub>whole, CVPR 2025</sub> | image | 26.3 | 36.28 | 0.957 | 0.058 |
| 12 | **DiffIR** <sub>whole, ICCV 2023</sub> | image | 26.9 | 36.28 | 0.957 | 0.062 |
| 13 | **HI-Diff** <sub>whole, NeurIPS 2023</sub> | image | 28.5 | 36.27 | 0.958 | 0.060 |
| 14 | **X-Restormer** <sub>whole, ECCV 2024</sub> | image | 26.0 | 36.27 | 0.958 | 0.057 |
| 15 | **M3SNet-32** <sub>whole, small, VisComp 2024</sub> | image | 16.7 | 36.21 | 0.957 | 0.062 |
| 16 | **M3SNet-64** <sub>whole, VisComp 2024</sub> | image | 66.4 | 36.21 | 0.956 | 0.063 |
| 17 | **M3SNet-32** <sub>TLC, small, VisComp 2024</sub> | image | 16.7 | 36.20 | 0.956 | 0.061 |
| 18 | **Restormer** <sub>whole, CVPR 2022</sub> | image | 26.1 | 36.19 | 0.957 | 0.061 |
| 19 | **Concertormer** <sub>TLC, arXiv 2025</sub> | image | 50.5 | 36.19 | 0.957 | 0.058 |
| 20 | **M3SNet-64** <sub>TLC, VisComp 2024</sub> | image | 66.4 | 36.17 | 0.956 | 0.063 |
| 21 | **Concertormer** <sub>whole, arXiv 2025</sub> | image | 50.5 | 36.16 | 0.956 | 0.059 |
| 22 | **GRL-B** <sub>whole, CVPR 2023</sub> | image | 19.8 | 36.15 | 0.955 | 0.063 |
| 23 | **LoFormer-L** <sub>whole, ACMMM 2024</sub> | image | 49.0 | 36.15 | 0.955 | 0.066 |
| 24 | **GRL-B** <sub>tiled, CVPR 2023</sub> | image | 19.8 | 36.14 | 0.955 | 0.064 |
| 25 | **MLWNet-B** <sub>whole, CVPR 2024</sub> | image | 95.1 | 36.14 | 0.955 | 0.063 |
| 26 | **EAMamba** <sub>whole, ICCV 2025</sub> | image | 25.3 | 36.13 | 0.956 | 0.069 |
| 27 | **NAFNet-32** <sub>whole, small, ECCV 2022</sub> | image | 17.1 | 36.12 | 0.955 | 0.065 |
| 28 | **F-MIMO+** <sub>whole, large, AAAI 2023</sub> | image | 19.5 | 36.11 | 0.953 | 0.076 |
| 29 | **XYScanNet** <sub>whole, CVPRW 2025</sub> | image | 63.5 | 36.10 | 0.956 | 0.059 |
| 30 | **DeepRFT** <sub>whole, arXiv 2021</sub> | image | 9.6 | 36.10 | 0.953 | 0.077 |
| 31 | **DeepRFT+** <sub>whole, large, arXiv 2021</sub> | image | 23.0 | 36.09 | 0.951 | 0.076 |
| 32 | **Stripformer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 19.7 | 36.09 | 0.950 | 0.058 |
| 33 | **F-MIMO+** <sub>tiled, large, AAAI 2023</sub> | image | 19.5 | 36.09 | 0.955 | 0.067 |
| 34 | **DeblurDiNAT-L** <sub>whole, arXiv 2024</sub> | image | 16.1 | 36.09 | 0.955 | 0.058 |
| 35 | **Stripformer** <sub>whole, ECCV 2022</sub> | image | 19.7 | 36.08 | 0.954 | 0.062 |
| 36 | **LaKDNet-L** <sub>whole, arXiv 2023</sub> | image | 37.5 | 36.08 | 0.955 | 0.063 |
| 37 | **F-NAFNet-32** <sub>whole, small, AAAI 2023</sub> | image | 17.8 | 36.08 | 0.955 | 0.063 |
| 38 | **NAFNet-64** <sub>whole, ECCV 2022</sub> | image | 67.9 | 36.08 | 0.954 | 0.065 |
| 39 | **Stoformer** <sub>tiled, NeurIPS 2022</sub> | image | 50.5 | 36.08 | 0.954 | 0.062 |
| 40 | **Stoformer** <sub>whole, NeurIPS 2022</sub> | image | 50.5 | 36.07 | 0.954 | 0.062 |
| 41 | **DeepRFT** <sub>TLC, arXiv 2021</sub> | image | 9.6 | 36.06 | 0.954 | 0.069 |
| 42 | **DGUNet+** <sub>whole, large, CVPR 2022</sub> | image | 69.6 | 36.02 | 0.951 | 0.067 |
| 43 | **F-MIMO** <sub>whole, AAAI 2023</sub> | image | 8.2 | 36.02 | 0.951 | 0.085 |
| 44 | **NAFNet-32** <sub>TLC, small, ECCV 2022</sub> | image | 17.1 | 35.99 | 0.952 | 0.069 |
| 45 | **MPRNet** <sub>whole, CVPR 2021</sub> | image | 20.1 | 35.99 | 0.952 | 0.076 |
| 46 | **Restormer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 26.1 | 35.98 | 0.953 | 0.067 |
| 47 | **DeepRFT-S** <sub>whole, small, arXiv 2021</sub> | image | 5.1 | 35.97 | 0.951 | 0.087 |
| 48 | **NAFNet-64** <sub>TLC, ECCV 2022</sub> | image | 67.9 | 35.97 | 0.952 | 0.068 |
| 49 | **MSDI-Net** <sub>whole, ECCV 2022</sub> | image | 246.9 | 35.96 | 0.952 | 0.070 |
| 50 | **F-MIMO** <sub>tiled, AAAI 2023</sub> | image | 8.2 | 35.96 | 0.953 | 0.076 |
| 51 | **DGUNet** <sub>whole, CVPR 2022</sub> | image | 17.3 | 35.95 | 0.952 | 0.072 |
| 52 | **Concertormer-lite** <sub>whole, small, arXiv 2025</sub> | image | 29.0 | 35.95 | 0.953 | 0.069 |
| 53 | **Concertormer-lite** <sub>TLC, small, arXiv 2025</sub> | image | 29.0 | 35.94 | 0.954 | 0.067 |
| 54 | **MSSNet** <sub>whole, ECCVW 2022</sub> | image | 15.6 | 35.93 | 0.953 | 0.069 |
| 55 | **F-MIMO-S** <sub>tiled, small, AAAI 2023</sub> | image | 4.4 | 35.92 | 0.952 | 0.077 |
| 56 | **MAXIM-3S** <sub>whole, CVPR 2022</sub> | image | 22.2 | 35.91 | 0.954 | 0.071 |
| 57 | **F-MIMO-S** <sub>whole, small, AAAI 2023</sub> | image | 4.4 | 35.91 | 0.950 | 0.087 |
| 58 | **LaKDNet-S** <sub>whole, small, arXiv 2023</sub> | image | 17.1 | 35.90 | 0.954 | 0.062 |
| 59 | **DeepRFT-S** <sub>TLC, small, arXiv 2021</sub> | image | 5.1 | 35.89 | 0.952 | 0.077 |
| 60 | **SFNet** <sub>whole, ICLR 2023</sub> | image | 13.3 | 35.89 | 0.952 | 0.076 |
| 61 | **MSDI-Net** <sub>tiled, ECCV 2022</sub> | image | 246.9 | 35.88 | 0.952 | 0.070 |
| 62 | **MSSNet-small** <sub>whole, small, ECCVW 2022</sub> | image | 6.8 | 35.87 | 0.952 | 0.070 |
| 63 | **ConvIR-L** <sub>whole, TPAMI 2024</sub> | image | 14.8 | 35.87 | 0.952 | 0.070 |
| 64 | **FFTformer** <sub>whole, CVPR 2023</sub> | image | 16.6 | 35.87 | 0.953 | 0.070 |
| 65 | **DeepRFT+** <sub>TLC, large, arXiv 2021</sub> | image | 23.0 | 35.86 | 0.950 | 0.066 |
| 66 | **F-NAFNet-64** <sub>whole, AAAI 2023</sub> | image | 68.0 | 35.85 | 0.952 | 0.064 |
| 67 | **IRNeXt (16 blocks)** <sub>whole, large, ICML 2023</sub> | image | 14.8 | 35.84 | 0.951 | 0.072 |
| 68 | **MLWNet-B** <sub>TLC, CVPR 2024</sub> | image | 95.1 | 35.82 | 0.943 | 0.080 |
| 69 | **HINet** <sub>whole, CVPRW 2021</sub> | image | 88.7 | 35.80 | 0.950 | 0.079 |
| 70 | **IRNeXt** <sub>whole, ICML 2023</sub> | image | 13.2 | 35.79 | 0.951 | 0.075 |
| 71 | **MT-RNN** <sub>whole, ECCV 2020</sub> | image | 2.6 | 35.79 | 0.951 | 0.082 |
| 72 | **FSNet** <sub>whole, TPAMI 2023</sub> | image | 13.3 | 35.65 | 0.948 | 0.075 |
| 73 | **SFNet** <sub>TLC, ICLR 2023</sub> | image | 13.3 | 35.61 | 0.943 | 0.083 |
| 74 | **PSS-NSC** <sub>whole, CVPR 2019</sub> | image | 3.9 | 35.56 | 0.950 | 0.082 |
| 75 | **StarIR** <sub>whole, 2025</sub> | image | 18.7 | 35.55 | 0.954 | 0.066 |
| 76 | **MIMO-UNet+** <sub>whole, large, ICCV 2021</sub> | image | 16.1 | 35.54 | 0.947 | 0.084 |
| 77 | **StarIR** <sub>TLC, 2025</sub> | image | 18.7 | 35.48 | 0.952 | 0.067 |
| 78 | **MIMO-UNet** <sub>whole, ICCV 2021</sub> | image | 6.8 | 35.47 | 0.946 | 0.089 |
| 79 | **Motion-ETR** <sub>whole, TPAMI 2021</sub> | image | 6.6 | 35.43 | 0.945 | 0.089 |
| 80 | **FFTformer + ID-Blau** <sub>whole, CVPR 2024</sub> | image | 16.6 | 35.40 | 0.946 | 0.080 |
| 81 | **SIUN** <sub>whole, Access 2020</sub> | image | 6.3 | 35.38 | 0.942 | 0.077 |
| 82 | **DMPHN-1-2-4** <sub>whole, CVPR 2019</sub> | image | 5.4 | 35.14 | 0.940 | 0.102 |
| 83 | **Ghost-DeblurGAN** <sub>whole, IROS 2022</sub> | image | 1.5 | 34.93 | 0.935 | 0.084 |
| 84 | **DeepDeblur** <sub>whole, CVPR 2017</sub> | image | 11.7 | 34.73 | 0.940 | 0.103 |
| 85 | **ChaIR** <sub>whole, KBS 2023</sub> | image | 15.0 | 34.72 | 0.930 | 0.081 |
| 86 | **EVSSM** <sub>whole, CVPR 2025</sub> | image | 17.1 | 34.42 | 0.908 | 0.105 |
| 87 | **IR-SDE** <sub>whole, ICML 2023</sub> | image | 137.1 | 33.95 | 0.918 | 0.113 |
| 88 | **DBGAN** <sub>whole, CVPR 2020</sub> | image | 11.6 | 33.78 | 0.909 | 0.137 |
| 89 | **RT-Focuser** <sub>whole, arXiv 2025</sub> | image | 5.9 | 33.59 | 0.810 | 0.143 |
| 90 | **MIMO-UNet+ + ID-Blau** <sub>whole, large, CVPR 2024</sub> | image | 16.1 | 31.96 | 0.876 | 0.198 |

</details>
<!-- generated:end -->

## What is in here

- **Methods**, image and video, from DeepDeblur (2017) to 2026, ported to as few environments as possible. One shared
  PyTorch environment serves most of them. The few that need special kernels (Mamba, NATTEN, mmcv ops, JAX) get their
  own, pinned. See [ENVS.md](ENVS.md).
- **Weights** in one format: `safetensors` plus a JSON card per model (architecture config, source, original md5,
  license, conversion notes). See [WEIGHTS.md](WEIGHTS.md).
- **Runs** of every method on every test set with its GoPro-trained weights, and with other released weights as extra
  rows. Every restored frame is kept losslessly (FFV1), so any metric can be recomputed without running a model again.
- **Metrics** per frame: PSNR, SSIM and LPIPS now. MS-SSIM, DISTS, PaQ-2-PiQ, MUSIQ, CLIP-IQA+, MANIQA, Q-Align and a
  temporal-consistency score for video come from the stored outputs. See [METRICS.md](METRICS.md).
- **Options**: one vocabulary for all methods (tiling, TLC, test-time augmentation, precision, temporal scheme,
  sampling) and two presets per method, the uniform `protocol` and the `authors` own setting. Both are reported. See
  [OPTIONS.md](OPTIONS.md).

```mermaid
flowchart LR
  R["registry/methods.yaml<br>paper, code commit, weights, license"] --> A["methods/&lt;name&gt;<br>thin adapter"]
  W["weights/&lt;name&gt;<br>safetensors + card"] --> A
  A --> X["run on GoPro, REDS, BSD, DVD<br>and RealBlur-J/R"]
  X --> O["outputs/<br>every frame, FFV1"]
  O --> M["metrics per frame"]
  M --> T["results/<br>master.json, per_sequence.tsv"]
  T --> S["tools/build_site.py"]
  S --> P["docs/ site, CSV,<br>this README"]
```

## Test sets

| Set | Sequences | Frames | Size | Blurry input PSNR | Notes |
|---|---:|---:|---|---:|---|
| GoPro test | 11 | 1111 | 1280x720 | 25.64 | the standard `blur` input, not `blur_gamma` |
| REDS val | 30 | 3000 | 1280x720 | 26.27 | |
| BSD 2ms16ms test | 20 | 3000 | 640x480 | 26.63 | real blur from a beam-splitter camera rig |
| DVD test | 10 | 1000 | 1280x720 | 27.29 | one portrait video |
| RealBlur-J test | 50 | 980 | about 670x760 | 27.68 | real blur, camera JPEG, 50 scenes, every image scored on its own; image methods only; PSNR and SSIM after the official ECC alignment |
| RealBlur-R test | 50 | 980 | about 680x773 | 35.35 | the same scenes developed from raw images, same evaluation |

Candidates to add: BSD 1ms8ms and 3ms24ms, HIDE.

## The site

The interactive leaderboard lives in [`docs/`](docs/), a single static page with the data inside it, ready for GitHub
Pages (Settings, Pages, deploy from the `docs/` folder of the default branch). One tab per test set, filters by method
type and family, sorting by any column, each method's score per test sequence, and the Pareto front of score against
parameters, compute or latency.

```bash
python tools/build_site.py              # docs/index.html, docs/leaderboard.csv and the tables above
python -m http.server 8765 --directory docs   # then open http://localhost:8765
```

The tables above and the page are generated from `results/` by the same script. Page design and code:
[`site/template.html`](site/template.html).

## Layout

```
deblurring-is-solved/
  README.md  METRICS.md  WEIGHTS.md  ENVS.md  PROTOCOL.md  OPTIONS.md
  registry/methods.yaml        one entry per method and variant: paper, venue, code (repo + commit), license,
                               weights (source + md5), environment, protocol notes, status
  registry/blocked.yaml        methods that cannot run, with the reason
  envs/                        environment specs and lock files (base + special)
  methods/<name>/              thin adapter: build() and forward(x) for images, reset()/step() for video
  weights/<name>/              <variant>.safetensors + <variant>.json   (large files: cluster / HF, not git)
  datasets/                    test-set manifests: sequences, frame lists, md5 of every frame
  results/                     master.json, per_sequence.tsv, input_per_sequence.tsv
  outputs/                     FFV1 videos of all restored frames (cluster only, not git)
  site/template.html           the page
  docs/                        the built site (GitHub Pages)
  tools/                       build_site, run, score, convert_weights, verify, ffv1 store
```

## Status

- Code and weights collected for about 110 methods and variants (a survey of about 230 entries, three search passes).
- Scored: see the note at the top of the leaderboard. Rows still running show in the site under *unfinished runs*.
- The other metrics will be computed from the FFV1 outputs in a separate scoring pass.
- Not runnable (weights only on Baidu, dead links, no weights, out of scope): listed in `registry/blocked.yaml`.

## Where things live

- Compute cluster (primary): environments, weights, FFV1 outputs and run logs.
- This folder is the git part: code, specs, registry, result tables and the site.
