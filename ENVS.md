# Environments

Goal: as few as possible. One base environment runs every method whose code is plain PyTorch.

| Env | Contents | Methods |
|---|---|---|
| `base` | Python 3.11, torch 2.10 + cu128, torchvision 0.25, numpy, pillow, einops, timm, scipy, scikit-image, opencv-headless, safetensors | most image and video methods |
| `mamba` | base + mamba-ssm / causal-conv1d built for sm_80 | EVSSM, MVSSM, XYScanNet, MaIR, EAMamba, ALGNet (if SSM) |
| `natten` | base + NATTEN with key names matching the released checkpoints | DeblurDiNAT |
| `mmcv` | base + mmcv ops built from source (deformable attention, MSDA) | STDAN, STDANet++, FGST, BasicVSR++ |
| `jax` | JAX + Flax (CUDA 12) | MAXIM |
| `diffusion` | base + diffusers / transformers pinned | HI-Diff, DiffIR, IR-SDE, DiffUIR, DA-CLIP, Diff-Plugin, AutoDIR, PixWizard |
| `metrics` | base + pyiqa pinned (+ Q-Align dependencies) | scoring only |

Rules:

- Every env is created from a spec in `envs/<name>.txt` (exact versions) with `uv`; nothing is installed from a
  method's own requirements file without reading it.
- A method moves out of `base` only for a compiled kernel or a hard version pin, and its card says why.
- Custom CUDA kernels are built from the method's own source and recorded with the commit.
- Each adapter is checked in its env: strict weight load, and the GoPro number against the authors' reported value.

Today on the cluster the methods still run in their per-method envs (about 50). Folding them into
the table above is the first consolidation step.
