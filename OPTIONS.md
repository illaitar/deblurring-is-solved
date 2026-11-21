# Inference options

Every adapter accepts the same option vocabulary. A method declares which options it supports and two presets:

- `protocol`: the uniform setting of this repository (PROTOCOL.md), the same for every method;
- `authors`: exactly what the authors used for their reported numbers.

The tables give both numbers per method. Any other combination can be run by passing options explicitly; a result
always stores the full resolved option set next to its metrics.

## Options that change the output

### Spatial scheme

| Option | Values | Used by (examples) |
|---|---|---|
| `tiling` | `none` \| `tile:<size>:<overlap>:<blend>` with blend `mean`, `gauss`, `hann`, `center` | Stoformer 512/32, STDANet++ 256, MSDI-Net and HINet 256 grids |
| `tlc` | `off` \| `on:<base>` (test-time local converter: global pooling and attention statistics over a local window) | NAFNet, UFPNet, MLWNet, SFNet, M3SNet, MSFS-Net-Local |
| `window` | `native` \| `<size>` (window of windowed attention or windowed inference) | DeepRFT `win_size 256` |
| `pad` | `<mode>:<multiple>` with mode `replicate`, `reflect`, `zero`; or `fixed:<H>x<W>` | all; DIVD crops to 720x1248 |
| `scale` | `1` \| `<factor>` (downscale in, upscale out) | none by default |

### Test-time augmentation

| Option | Values | Used by |
|---|---|---|
| `tta` | `none` \| `flip2` \| `x4` \| `x8` (flips and 90-degree rotations, averaged) | MVSSM, STDANet++ test configs |
| `tta_temporal` | `off` \| `on` (forward and reversed sequence, averaged) | CDVD-TSPNL `flip_seq` |
| `passes` | `1` \| `<n>` random passes averaged | ShuffleFormer `repeat_num 16`, Stoformer |

### Numerics

| Option | Values | Notes |
|---|---|---|
| `precision` | `fp32` \| `fp16` (autocast) \| `half` (`net.half()`) \| `bf16` | protocol: fp16 if within 0.05 dB of fp32 on 20 GoPro frames |
| `tf32` | `on` \| `off` | |
| `norm` | `eval` \| `train` (batch statistics at test time) | DeblurGAN-v2, Ghost-DeblurGAN infer in train mode |
| `quant` | `round` \| `trunc` | protocol: round, clamp to [0, 1] |

### Input conventions (fixed per method, recorded)

| Option | Values | Used by |
|---|---|---|
| `input` | `blur` \| `blur_gamma` (GoPro only) | DeepDeblur, RNN-MBP trained and tested on `blur_gamma` |
| `channels` | `rgb` \| `bgr` | RNN-MBP, DeblurGAN-v2 |
| `range` | `0_1` \| `0_255` \| `m1_1` \| `shift:<value>` | MT-RNN 0..255, Stripformer -0.5 |

### Output and weights

| Option | Values | Used by |
|---|---|---|
| `head` | `last` \| `<index>` | MPRNet, MIMO-UNet, HINet (several outputs) |
| `weights` | `<variant>`: training set (`GoPro`, `DVD`, `BSD2`, `REDS`, ...), size (`S`, `B`, `L`, width), extra data (`alldata`), `ema` | every method |
| `exit` | `off` \| `<threshold>` (adaptive patch exit) | AdaRevD |

### Video, temporal scheme

| Option | Values | Used by |
|---|---|---|
| `temporal` | `window:<n>` \| `clip:<n>:<overlap>` \| `sequence` \| `stream` | VDTR/DBN/EDVR window 5, VRNN 12/4, RNN-MBP 20/4, MVSSM 35/0, CDVD-TSPNL 10 |
| `lookahead` | `<k>` frames (streaming) | TSRN-RTVD 1 |
| `last_clip` | `aligned` \| `short` | MVSSM |
| `edges` | `mirror` \| `replicate` \| `drop` | dropped edge frames give 1067 of 1111 on GoPro |
| `init` | `blur` \| `zeros` \| `gt` (start of a recurrence; `gt` is never allowed) | CATF uses GT in its own test |
| `reset` | `off` \| `watchdog:<threshold>` \| `every:<n>` | |
| `motion` | `builtin` \| `<estimator>` \| `teacher` | |
| `stride` | `1` \| `<n>` (temporal subsampling) | |

### Generative and all-in-one

| Option | Values |
|---|---|
| `steps` | number of sampling steps |
| `sampler` | `ddpm` \| `ddim` \| `ode` \| method-specific |
| `seed` | integer (protocol: 0) |
| `guidance` | guidance scale |
| `samples` | `1` \| `<n>` averaged |
| `vae_tiling` | `off` \| `tile:<size>:<overlap>` |
| `task` | text instruction, task id or prompt for all-in-one models; `none` for blind ones |
| `base_model` | e.g. SD 1.4 / 2.1, SDXL, DiT version |

### Adaptation to the test data

| Option | Values | Used by |
|---|---|---|
| `adapt` | `off` \| `tta:<domain>` \| `self:<video>` | DADeblur, SelfHVD (never in `protocol`) |
| `calib` | `off` \| `<parameter>:<value or rule>` | |

## Evaluation options (the output stays the same, the number changes)

| Option | Values | Protocol |
|---|---|---|
| `color` | `rgb` \| `y` | rgb |
| `ssim_impl` | `skimage_gauss` \| `matlab` \| `pyiqa` | skimage_gauss (sigma 1.5) |
| `border` | `0` \| `<pixels>` cropped before scoring | 0 |
| `align` | `none` \| `ecc` (RealBlur) | none |
| `frames` | `all` \| `emitted` \| `common` (the set every method emits) | `emitted` and `common` both reported |
| `average` | `frames` \| `sequences` | frames, per-sequence means stored |

## Runtime only (output unchanged or within floating-point noise)

`compile` (`off`, `default`, `max-autotune`), `cuda_graphs`, `channels_last`, `batch`, `cudnn_benchmark`. Used for the
timing tables; quality runs record them too.
