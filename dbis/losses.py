"""dbis.losses: the training losses of the benchmark's methods, each computed as its authors' code computes it.

    from dbis.losses import LOSSES, make
    loss = make("charb_e_1e-12+0.1*fft_cmod_rfft")   # ids of LOSSES joined by +, each optionally weighted: "w*id"
    value = loss(pred, target, **ctx)                 # pred, target: (B, C, H, W) or (B, T, C, H, W) in [0, 1]

Conventions
- Every loss is LOSSES[id](pred, target, **ctx) -> a scalar tensor, with pred and target in [0, 1] (RGB). A loss whose
  authors worked in another range (0..255 BGR, [-1, 1], [-0.5, 0.5], ImageNet mean/std) converts inside, so the value
  is the authors' value for the same images and the gradient is taken w.r.t. the [0, 1] prediction.
- Plain reductions (means, sums) see the whole tensor, whatever its shape. Per-image losses (PSNR, colour
  Charbonnier, edge, gradient, HOG, Pearson, SSIM, VGG/LPIPS) see a clip (B, T, C, H, W) as B*T images.
- ctx carries what a loss needs beyond pred and target (blur=, mask=, kernel=, ...); every loss ignores keys it does not
  use, so one ctx serves a whole make() spec. Weights live in the spec ("0.1*fft_cmod_rfft"), never inside a loss,
  except where the authors hard-coded them inside one formula (content_dgv2's 0.006 / 0.5).
- Scale factors the authors' range adds to a plain loss also go into the spec: DeepDeblur and VRNN train L1 on 0..255
  ("255*l1_mean"), PVDNet MSE on [-1, 1] ("4*mse_mean"); HINet and MSDI-Net use "0.5*psnr_naf"; CDVD-TSP "2*hem_cdvd".
- Multi-output and multi-scale compositions (per-scale targets, which frames) are the caller's: apply a term per output.
- FFT losses run in float32 (pred.float()); the others follow the dtype they are given (autocast applies to the VGGs).
- Stateful losses (fusion_dec_slair: running statistics; feat_l1_msdi_enc: spectral-norm power iteration) are objects
  with .fresh(); make() gives each composed loss its own state. LOSSES[id] itself is one shared default instance.

Implemented (id = the inventory id in lower case; reference = the authors' code the cluster test compares against)
  pixel
  l1_mean            mean|x-y|                                  MIMO-UNet/train.py:13, Turtle/basicsr/loss/__init__.py:7-16,
                     BSSTNet/basicsr/losses/basic_loss.py:91-115 (and 48 more: Restormer, FFTformer, STDAN, ...)
  mse_mean           mean (x-y)^2                               DMPHN, RT-Focuser, Motion-ETR, PVDNet, RT-VDBLR
  psnr_naf           (10/ln10) mean_b ln(mean_chw (x-y)^2 + 1e-8)  NAFNet/basicsr/models/losses/losses.py:90-115;
                     HINet, MSDI-Net, M3SNet, MLWNet, UFPNet
  smoothl1_b0.01     mean smooth_l1(beta 0.01) on ImageNet-normalised images      ProRes/ProRes/models_ours.py:410-430
  masked_l1_ram      sum(|x-y| M)/(sum M + 1e-5)/3 on ImageNet-normalised images; ctx mask=M (broadcast to x)
                                                                RAM/ram/losses/losses.py:117-131 (RAM++ pre-training)
  charb_e2_1e-3      mean sqrt(d^2 + eps^2), eps 1e-3           MPRNet/Deblurring/losses.py:5-16; DGUNet, DeepRFT,
                     Stripformer, DeblurDiNAT, XYScanNet, Stoformer, ShuffleFormer, GRL, MISCFilter, TSRN-RTVD, ID-Blau
  charb_e_1e-12      mean sqrt(d^2 + 1e-12)                     DSTNet-plus/basicsr/losses/basic_loss.py:23-24,98;
                     DSTNet, MVSSM, BasicVSR++, CDVD-TSPNL, FGST
  charb_e_1e-9       mean sqrt(d*d + 1e-9)                      KAIR models/loss.py CharbonnierLoss (VRT, RVRT)
  charb_e_1e-6_sumps sum sqrt(d^2 + 1e-6) / target.shape[0]     SimDeblur/simdeblur/model/loss/charbonnier_loss.py:10-23
                     (VDTR, CATF)
  charb_sum_e_1e-6   sum sqrt(d*d + 1e-6)                       EDVR old_version codes/models/loss.py:5-15 (EDVR)
  charb_color_e2_1e-3  mean_bhw sqrt(mean_c d^2 + 1e-6)         ESTRNN/train/loss.py:91-108; MMP-RNN, MemDeblur, RNN-MBP
  frequency
  fft_l1_full        L1 mean of stacked (Re, Im) of fft2         FFTformer/basicsr/models/losses/losses.py:85-114,
                     MIMO-UNet/train.py:59-69 (torch.rfft onesided=False); SFNet, ConvIR, FSNet, IRNeXt, ChaIR, StarIR,
                     MSSNet, EVSSM, ALGNet, BioIR, DATPRL-IR, R2R
  fft_l1_rfft        L1 mean of stacked (Re, Im) of rfft2        MoCE-IR/src/utils/loss_utils.py:90-103; MIRAGE
  fft_cmod_full      mean |fft2(x) - fft2(y)| (complex modulus)  DeepRFT/losses.py:44-51; MISCFilter
  fft_cmod_rfft      mean |rfft2(x) - rfft2(y)| (complex modulus) DSTNet-plus/basicsr/losses/basic_loss.py:117-126;
                     DSTNet, MVSSM, F-MIMO, F-NAFNet, Concertormer, LoFormer, AdaRevD (DeepRFT-AAAI2023 FreqLoss)
  fft1d_h_charb_idblau  charb_e2_1e-3 of stacked (Re, Im) of a 1-D fft along H   ID-Blau/MIMO_UNet/deblur_train.py:30-36,
                     111-121 (MIMO-UNet+ + ID-Blau: the torch>=1.8 fallback of torch.rfft)
  structure
  edge_mpr           charb_e2_1e-3 of Laplacian-pyramid residuals (5x5 Gaussian, replicate pad)
                     MPRNet/Deblurring/losses.py:18-43; DGUNet, DeepRFT, Stripformer, DeblurDiNAT, XYScanNet, MISCFilter
  grad_l1_estrnn     L1 mean of per-channel gradient magnitude sqrt(gv^2 + gh^2 + 1e-6), 3x3 central differences,
                     zero padding 2 (output H+2 x W+2)          MMP-RNN/MMP-RNN/train/loss.py:23-72 (any channel count)
  hog_l1_hogformer   L1 mean of 9-bin HOG (8x8 cells, 2x2 blocks)  HOGformer/settingIII&IV/train.py:17-133
  pearson_corr_hogformer  mean over finite samples of (1 - corr)/2   HOGformer/settingIII&IV/train.py:143-163
  msssim_pms         1 - MS-SSIM of jorge-pessoa/pytorch-msssim master (prod(mcs[:4]^w) * ssim5^w5)
  msssim_pms_v2018   1 - the same package's MS-SSIM of 2018-07 .. 2021-04-03 (prod over 4 of mcs_i^w_i * ssim5^w5),
                     the version Motion-ETR's stage 1 most likely ran (Motion-ETR/models/losses.py:11,23-31)
  tv_l1_aniso_mean   two L1 means of neighbour differences of pred along W and H  Motion-ETR/models/
                     bluring_offset_quadratic.py:97-99 (pred = the offset field)
  l2_act_reg_mean    mean pred^2                                Motion-ETR/models/bluring_offset_quadratic.py:102-103
  wavelet_pr_ac_mlw  sum over learnable wavelet blocks of perfect-reconstruction + alias-cancellation errors of their
                     filters; ctx model= (every submodule with dec_lo, dec_hi, rec_lo, rec_hi) or wavelet_filters=
                     [(dec_lo, dec_hi, rec_lo, rec_hi), ...]; pred and target unused
                                                                MLWNet/basicsr/models/archs/wavelet_block.py:256-317
  fixed pretrained networks
  content_dgv2       0.006 MSE(VGG19 conv3_3) + 0.5 MSE(pixels), batch element 0 ImageNet-normalised, the others not
                                                                DeblurGANv2/models/losses.py:26-56; Ghost-DeblurGAN
  perc_vgg19_conv5_4_l1_raw255_bgr  L1 mean of VGG19 conv5_4 (pre-ReLU) on BGR 0..255, no normalisation
                                                                VRNN/utils.py:9-26
  perc_vgg19_r12_r22_c33_msesum  sum_i w_i sum (phi_i(x)-phi_i(y))^2, phi = relu1_2, relu2_2, conv3_3 (torchvision's
                     in-place ReLU turns the stored conv1_2 and conv2_2 into relu1_2 and relu2_2), w = 1, 0.2, 0.04,
                     RGB [0, 1] without normalisation      SimDeblur/simdeblur/model/loss/perceptual_loss.py:20-52 (VDTR)
  contrast_vgg19_relu31_masked  sum(d_ap/(d_an + 1e-7) m)/sum(m) on VGG19 relu3_1 of [-0.5, 0.5] images; anchor pred,
                     positive target, negative ctx blur=; m = bilinear 1/4 of 1[mean_c|target - blur| > 0.01]
                                                                Stripformer/models/losses.py:6-46; ID-Blau, DeblurDiNAT,
                                                                XYScanNet
  lpips_vgg          mean LPIPS v0.1 (VGG16) on [-1, 1]          lpips 0.1.4 (as SLAIR/train_decoder.py:89 calls it)
  haar_hf_l1_slair   L1 mean of the LH, HL, HH bands of a 2x2 stride-2 Haar on [-1, 1]  SLAIR/train_decoder.py:43-66
  fusion_dec_slair   w_p lpips_vgg + w_m MSE + w_h haar_hf_l1_slair on [-1, 1] with SLAIR's running z-score weights
                     (base 1.0 / 0.5 / 0; base weights for the first 1000 calls)   SLAIR/train_decoder.py:68-215,403-410
  feat_l1_msdi_enc   sum_k L1 mean(E_k(target), E_k(pred)), E = MSDI-Net's frozen degradation encoder (5 taps,
                     spectral norm in train mode, InstanceNorm); ctx msdi_encoder= path to the authors' net_E.pth or a
                     state dict (not shipped: MSDI-Net's README asks to download it)
                                                                MSDI-Net/basicsr/models/archs/encoder1.py:118-165
  masks, mining and network outputs
  hem_cdvd           L1 mean of x M, y M; M = top-50 % residual pixels per sample + 10 % random pixels (numpy shuffle,
                     ctx hem_rng= a numpy RandomState/Generator, default the global np.random as the authors); a clip
                     (B, T, C, H, W) is the channel concat (B, T*C, H, W) as CDVD-TSP concatenates its outputs
                                                                CDVD-TSP/code/loss/hard_example_mining.py:6-47
  bim_masked_mse_pvdnet  mean over the 4 pairs of sum((W_p - y M_p)^2)/sum(M_p); pred = warped frames (B, 4, C, H, W)
                     in the order bb, bs, sb, ss, ctx warp_mask= (B, 4, C or 1, H, W)
                                                                PVDNet/models/trainers/trainer.py:187-194 (BIMNet)
  reblur_sv_l1_ufp   L1 mean(B_K(pred), blur), B_K = per-pixel 19x19 kernels, replicate pad; ctx blur=, kernel=
                     (B, H*W, 19, 19); for B > 1 the authors' unfold pairs channel c of sample b with the kernel of sample
                     (b*3+c) % B, kept               UFPDeblur/basicsr/utils/sv_blur.py:9-47 + image_restoration_model.py
  warp_l1_stdan      (0.5 F + 0.5 B)/K of L1s between GT frames (bilinear 1/4, align_corners) and their neighbours warped
                     by each of the K stage flows, for all four triplets; target = GT clip (B, 5, C, H, W), ctx
                     flows_forward=, flows_backward= lists of (B, 2, 2, H/4, W/4); STDAN's weight 0.05 goes in the spec
                                                                STDAN/core/train.py:21-35 + models/submodules.py:221-251
  tur_unc_l1_5t      per degradation class: L1(s gt, s pred) + 2 ln mean u, s = 1/(mean_class u + 1e-6), with the
                     authors' accumulation of earlier classes' samples; ctx uncertainty=, de_id=
                                                     TUR/basicsr/losses/airnet_uncertainty_loss.py:67-125 (candidate for TUR)

Aliases (the short names dbis/train.py used before): l1 -> l1_mean, mse -> mse_mean, psnr -> psnr_naf,
charbonnier -> charb_e_1e-12 (DSTNet+'s), fft -> fft_cmod_rfft (DSTNet+'s). The train.py recipes, in full:
NAFNet-64 "psnr_naf", BSSTNet "l1_mean", DSTNetPlus-L "charb_e_1e-12+0.1*fft_cmod_rfft", Turtle "l1_mean".

External weights (downloaded on first use into the torch hub cache, $TORCH_HOME/hub/checkpoints): torchvision VGG19
ImageNet (vgg19-dcbb9e9d.pth, the file of vgg19(pretrained=True)) for the VGG19 losses; torchvision VGG16 ImageNet
(vgg16-397923af.pth) and the LPIPS v0.1 linear layers (vgg.pth of the lpips package, from its GitHub repository) for
lpips_vgg / fusion_dec_slair. feat_l1_msdi_enc needs MSDI-Net's own net_E.pth (ctx msdi_encoder=).

Tests: tests/test_losses.py (pytest, CPU, no downloads: shapes, finite values and gradients, closed forms, specs) and
tests/cluster_compare_losses.py (value and gradient against the authors' code, float32, then time and peak memory of
value + backward at (8, 3, 256, 256); results in diag/train2711/losses/test_results.json on the cluster).

Not in this library (inventory id: why)
  CHARB_UNSPEC: LaKDNet published neither eps nor reduction and no training code (nearest: charb_e2_1e-3)
  FFT_L1_MAXIM_PAPER: MAXIM's paper leaves FFT kind, norm and reduction open, training code not released (nearest:
    fft_l1_full)
  PERC_PREACT_DBGAN: DBGAN names neither network, layer nor distance, no training code
  VAE_DEC_RESTOREVAR: paper only; SSIM and perceptual variants unstated, and it has a PatchGAN term
  GAN_RAGAN_LS_DOUBLE_DGV2, GAN_WGANGP_DOUBLE_DGV2, GAN_RA_BCE_DBGAN: need discriminators trained along with the net
  VAE_LPIPS_GAN_SLAIR: VAE training with KL, learned log-variance and a hinge PatchGAN discriminator
  OT_RCOT: DA-RCOT's optimal-transport dual: a critic network trained in alternation (with gradient penalty)
  BARY_WB_BARYIR: BaryIR's Wasserstein-barycenter dual: per-domain potential networks trained in alternation
  CLIP_INFONCE_SYM: DA-CLIP's controller training, contrastive over image/text batches of a CLIP model
  CE_RAWDOT_VLU: VLU-Net's degradation-adapter pre-training: classifier on CLIP features
  CONTRAST_NTXENT_SPD_MIRAGE: MIRAGE's NT-Xent between learnable projection heads inside the network
  MSFS_HF_CONTRAST, MSFS_LF_CONSIST: MSFS-Net's terms on the network's internal frequency features, no code released
  CE_CORR_RTVDBLR: RT-VDBLR's cross-entropy on the network's own correlation volume, LiteFlowNet targets
  CE_LS0.1, CE_MEAN, FOCAL_CE_G2: degradation-classifier cross-entropies (R2R, AdaRevD, InstructIR, DCPT)
  CE_TOKENS_VAR: RestoreVAR's next-scale token prediction
  MASK_POLICY_LOGP_RAM: RAM++'s REINFORCE-like term that trains the mask generator
  MOE_BALANCE_CV2: MoCE-IR's mixture-of-experts load-balancing term on the router
  DIFFCHAIN_L1_DIFFIR, DIFFCHAIN_L1_HIDIFF: L1 on diffusion-chain priors from a stage-1 network of the method
  DIFF_IRSDE_MATCH, DIFF_RESDIFF_PREDRES_L1: diffusion / SDE objectives (IR-SDE, DA-CLIP, DiffUIR): noise-level sampling
  FM_BRIDGE_V_MSE_COS: SLAIR's latent flow matching
  TINYDELTA_HETERO_SMOOTHL1: TSRN-RTVD's separately trained trajectory predictor (displacements, log-variance,
    dataset statistics), not an image loss
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LPIPS_URL = "https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/vgg.pth"


def _img(x):
    """(..., C, H, W) -> (N, C, H, W): a per-image loss sees every frame of a clip as one image."""
    return x.reshape(-1, *x.shape[-3:])


_CONSTS = {}


def _cached(key, like, make):
    """make() (a CPU float32 tensor) moved to like's device and dtype once per (key, device, dtype)."""
    k = (key, like.device, like.dtype)
    if k not in _CONSTS:
        _CONSTS[k] = make().to(like.device, like.dtype)
    return _CONSTS[k]


def _const(values, like):
    """A (C, 1, 1) float32 constant on like's device and dtype (broadcasts over the channel axis, dim -3)."""
    return _cached(("const", values), like, lambda: torch.tensor(values, dtype=torch.float32).view(-1, 1, 1))


def _imagenet(x):
    """[0, 1] RGB -> ImageNet mean/std normalised, as torchvision.transforms.Normalize."""
    return (x - _const(IMAGENET_MEAN, x)) / _const(IMAGENET_STD, x)


# ------------------------------------------------------------------------------------------------------------ pixel
def l1_mean(pred, target, **ctx):
    """mean |x - y| (nn.L1Loss())."""
    return F.l1_loss(pred, target)


def mse_mean(pred, target, **ctx):
    """mean (x - y)^2 (nn.MSELoss())."""
    return F.mse_loss(pred, target)


def psnr_naf(pred, target, **ctx):
    """NAFNet PSNRLoss: (10 / ln 10) * mean over images of ln(mean_chw (x - y)^2 + 1e-8) (= -PSNR in dB, roughly)."""
    p, y = _img(pred), _img(target)
    return 10 / np.log(10) * torch.log(((p - y) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


def smoothl1_b0_01(pred, target, **ctx):
    """ProRes: mean smooth_l1(beta=0.01) between ImageNet-normalised images."""
    return F.smooth_l1_loss(_imagenet(pred), _imagenet(target), reduction="none", beta=0.01).mean()


def masked_l1_ram(pred, target, *, mask, **ctx):
    """RAM++ AdaptiveMaskL1Loss: sum(|x - y| * M) / (sum(M) + 1e-5) / 3 on ImageNet-normalised images."""
    l1 = F.l1_loss(_imagenet(pred), _imagenet(target), reduction="none")
    return (l1 * mask).sum() / (mask.sum() + 1e-5) / 3


def charb_e2_1e_3(pred, target, eps=1e-3, **ctx):
    """MPRNet CharbonnierLoss: mean sqrt(d*d + eps*eps), eps 1e-3."""
    diff = pred - target
    return torch.mean(torch.sqrt((diff * diff) + (eps * eps)))


def charb_e_1e_12(pred, target, **ctx):
    """BasicSR CharbonnierLoss (DSTNet+): mean sqrt(d^2 + 1e-12)."""
    return torch.sqrt((pred - target) ** 2 + 1e-12).mean()


def charb_e_1e_9(pred, target, **ctx):
    """KAIR CharbonnierLoss with VRT's G_charbonnier_eps 1e-9: mean sqrt(d*d + 1e-9)."""
    diff = pred - target
    return torch.mean(torch.sqrt((diff * diff) + 1e-9))


def charb_e_1e_6_sumps(pred, target, **ctx):
    """SimDeblur CharbonnierLoss: sum sqrt(d^2 + 1e-6) / target.shape[0]."""
    return torch.sum(torch.sqrt((pred - target) ** 2 + 1e-6)) / target.shape[0]


def charb_sum_e_1e_6(pred, target, **ctx):
    """EDVR (old_version) CharbonnierLoss: sum sqrt(d*d + 1e-6), no division."""
    diff = pred - target
    return torch.sum(torch.sqrt(diff * diff + 1e-6))


def charb_color_e2_1e_3(pred, target, **ctx):
    """ESTRNN L1_Charbonnier_loss_color: mean over pixels of sqrt(mean_c d^2 + 1e-3 * 1e-3)."""
    p, y = _img(pred), _img(target)
    diff = torch.add(p, -y)
    error = torch.sqrt(torch.mean(diff * diff, 1, True) + 1e-3 * 1e-3)
    return torch.mean(error)


# -------------------------------------------------------------------------------------------------------- frequency
def _ri(z):
    return torch.stack([z.real, z.imag], dim=-1)


def fft_l1_full(pred, target, **ctx):
    """L1 mean between stacked (Re, Im) of the full fft2 (norm backward) of output and target."""
    return F.l1_loss(_ri(torch.fft.fft2(pred.float(), dim=(-2, -1))), _ri(torch.fft.fft2(target.float(), dim=(-2, -1))))


def fft_l1_rfft(pred, target, **ctx):
    """L1 mean between stacked (Re, Im) of the one-sided rfft2 (norm backward)."""
    return F.l1_loss(_ri(torch.fft.rfft2(pred.float())), _ri(torch.fft.rfft2(target.float())))


def fft_cmod_full(pred, target, **ctx):
    """DeepRFT fftLoss: mean complex modulus |fft2(x) - fft2(y)| over the full spectrum."""
    return torch.mean(torch.abs(torch.fft.fft2(pred.float()) - torch.fft.fft2(target.float())))


def fft_cmod_rfft(pred, target, **ctx):
    """DSTNet+ FFT: mean complex modulus |rfft2(x) - rfft2(y)| over the half spectrum (their F.l1_loss on complex)."""
    return torch.abs(torch.fft.rfft2(pred.float(), norm="backward") -
                     torch.fft.rfft2(target.float(), norm="backward")).mean()


def fft1d_h_charb_idblau(pred, target, **ctx):
    """ID-Blau MIMO-UNet+: charb_e2_1e-3 between stacked (Re, Im) of a 1-D fft along H (dim -2)."""
    tp, ty = torch.fft.fft(pred.float(), dim=-2), torch.fft.fft(target.float(), dim=-2)
    return charb_e2_1e_3(torch.stack((tp.real, tp.imag), -1), torch.stack((ty.real, ty.imag), -1))


# -------------------------------------------------------------------------------------------------------- structure
def _gauss5(c, like):
    def make():
        k = torch.tensor([[.05, .25, .4, .25, .05]], dtype=torch.float32)
        return torch.matmul(k.t(), k).unsqueeze(0).repeat(c, 1, 1, 1)
    return _cached(("gauss5", c), like, make)


def _laplacian_mpr(x, kernel):
    def conv_gauss(img):
        return F.conv2d(F.pad(img, (2, 2, 2, 2), mode="replicate"), kernel, groups=img.shape[1])

    filtered = conv_gauss(x)
    new_filter = torch.zeros_like(filtered)
    new_filter[:, :, ::2, ::2] = filtered[:, :, ::2, ::2] * 4
    return x - conv_gauss(new_filter)


def edge_mpr(pred, target, **ctx):
    """MPRNet EdgeLoss: charb_e2_1e-3 between x - G(up(down(G(x)))) of output and target."""
    p, y = _img(pred), _img(target)
    k = _gauss5(p.shape[1], p)
    return charb_e2_1e_3(_laplacian_mpr(p, k), _laplacian_mpr(y, k))


def _grad_mag_estrnn(x):
    """ESTRNN Gradient: per channel sqrt(gv^2 + gh^2 + 1e-6) of [[0,-1,0],[0,0,0],[0,1,0]] and its transpose applied
    as a cross-correlation with zero padding 2 (so the map is (H+2) x (W+2)); differences taken directly."""
    h, w = x.shape[-2:]
    xp = F.pad(x, (2, 2, 2, 2))
    gv = xp[..., 2:h + 4, 1:w + 3] - xp[..., 0:h + 2, 1:w + 3]
    gh = xp[..., 1:h + 3, 2:w + 4] - xp[..., 1:h + 3, 0:w + 2]
    return torch.sqrt(torch.pow(gv, 2) + torch.pow(gh, 2) + 1e-6)


def grad_l1_estrnn(pred, target, **ctx):
    """MMP-RNN L1GradientLoss: L1 mean between gradient-magnitude maps."""
    return F.l1_loss(_grad_mag_estrnn(_img(pred)), _grad_mag_estrnn(_img(target)))


def _hog(x, nbins=9, cell=8, block=2, eps=1e-8):
    b, c, h, w = x.shape
    gray = (0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]).unsqueeze(1) if c == 3 else x
    dxf = _cached("sobel_x", x, lambda: torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3))
    dyf = _cached("sobel_y", x, lambda: torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3))
    angles = _cached(("hog_angles", nbins), torch.empty(0, device=x.device), lambda: torch.tensor(
        [(i * np.pi / nbins) for i in range(nbins)]).view(1, -1, 1, 1))
    dx, dy = F.conv2d(gray, dxf, padding=1), F.conv2d(gray, dyf, padding=1)
    magnitude = torch.sqrt(dx ** 2 + dy ** 2 + eps)
    orientation = torch.abs(torch.atan2(dy, dx + eps))
    delta = torch.abs(orientation - angles)
    delta = torch.min(delta, np.pi - delta)
    weights = torch.relu(1.0 - delta / (np.pi / nbins))
    nh, nw = (h // cell) * cell, (w // cell) * cell
    if h % cell != 0 or w % cell != 0:
        magnitude, weights = magnitude[:, :, :nh, :nw], weights[:, :, :nh, :nw]
    hist = F.avg_pool2d(weights * magnitude, kernel_size=cell, stride=cell)
    bh, ch, hc, wc = hist.shape
    if hc >= block and wc >= block:
        blocks = F.unfold(hist, kernel_size=block, stride=1).permute(0, 2, 1).reshape(-1, ch * block ** 2)
        blocks = blocks / (torch.norm(blocks, p=2, dim=1, keepdim=True) + eps)
        return blocks.reshape(bh, (hc - block + 1) * (wc - block + 1), -1).reshape(bh, -1)
    return hist.reshape(bh, -1)


def hog_l1_hogformer(pred, target, **ctx):
    """HOGformer HOGLoss: L1 mean between HOG descriptors (gray, Sobel, 9 unsigned bins, 8x8 cells, 2x2 blocks)."""
    return F.l1_loss(_hog(_img(pred)), _hog(_img(target)))


def pearson_corr_hogformer(pred, target, **ctx):
    """HOGformer correlation loss: mean over images with a finite value of (1 - Pearson r) / 2."""
    x1, x2 = _img(pred), _img(target)
    b = x1.shape[0]
    x1, x2 = x1.reshape(b, -1), x2.reshape(b, -1)
    m1, m2 = x1.mean(dim=-1, keepdim=True), x2.mean(dim=-1, keepdim=True)
    numerator = ((x1 - m1) * (x2 - m2)).sum(dim=-1, keepdim=True)
    std1 = (x1 - m1).pow(2).sum(dim=-1, keepdim=True).sqrt()
    std2 = (x2 - m2).pow(2).sum(dim=-1, keepdim=True).sqrt()
    pearson = (1. - numerator.div(std1 * std2 + 1e-6)) / 2.
    return pearson[~pearson.isnan() * ~pearson.isinf()].mean()


def _ssim_pms(img1, img2, window_size=11):
    """pytorch_msssim.ssim(full=True) of jorge-pessoa: value range from img1 (255 if max > 128 else 1, minus -1 if
    min < -0.5 else 0), Gaussian window min(11, H, W) with sigma 1.5, no padding; returns (mean SSIM, mean cs)."""
    max_val = 255 if torch.max(img1) > 128 else 1
    min_val = -1 if torch.min(img1) < -0.5 else 0
    big_l = max_val - min_val
    _, channel, height, width = img1.size()
    real = min(window_size, height, width)

    def make():
        g = torch.tensor([math.exp(-(x - real // 2) ** 2 / float(2 * 1.5 ** 2)) for x in range(real)],
                         dtype=torch.float32)
        g = (g / g.sum()).unsqueeze(1)
        return g.mm(g.t()).float().unsqueeze(0).unsqueeze(0).expand(channel, 1, real, real).contiguous()

    window = _cached(("pms_window", real, channel), torch.empty(0, device=img1.device), make)
    mu1 = F.conv2d(img1, window, padding=0, groups=channel)
    mu2 = F.conv2d(img2, window, padding=0, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=0, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=0, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=0, groups=channel) - mu1_mu2
    c1, c2 = (0.01 * big_l) ** 2, (0.03 * big_l) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    cs = (v1 / v2).mean()
    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
    return ssim_map.mean(), cs


def _msssim_pms(img1, img2, v2018):
    weights = _cached("pms_weights", torch.empty(0, device=img1.device),
                      lambda: torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float32))
    ssims, mcs = [], []
    for _ in range(5):
        sim, cs = _ssim_pms(img1, img2)
        ssims.append(sim)
        mcs.append(cs)
        img1, img2 = F.avg_pool2d(img1, (2, 2)), F.avg_pool2d(img2, (2, 2))
    pow1, pow2 = torch.stack(mcs) ** weights, torch.stack(ssims) ** weights
    if v2018:
        return torch.prod(pow1[:-1] * pow2[-1])
    return torch.prod(pow1[:-1]) * pow2[-1]


def msssim_pms(pred, target, **ctx):
    """Motion-ETR stage 1: 1 - pytorch_msssim.msssim(pred, target) of jorge-pessoa master (since 2021-04-03).
    Like the package it returns NaN when a level's cs is negative (fractional power)."""
    return 1 - _msssim_pms(_img(pred), _img(target), v2018=False)


def msssim_pms_v2018(pred, target, **ctx):
    """1 - the same package's msssim of 2018-07 .. 2021-04-03: torch.prod(pow1[:-1] * pow2[-1]), i.e. the last
    level's SSIM enters with exponent 4 * 0.1333 (fixed by the package's PR #20)."""
    return 1 - _msssim_pms(_img(pred), _img(target), v2018=True)


def tv_l1_aniso_mean(pred, target=None, **ctx):
    """Motion-ETR tv_loss on the offset field (pred): L1 mean of neighbours along W + L1 mean along H."""
    return F.l1_loss(pred[..., :-1], pred[..., 1:]) + F.l1_loss(pred[..., :-1, :], pred[..., 1:, :])


def l2_act_reg_mean(pred, target=None, **ctx):
    """Motion-ETR reg_loss on the offset field (pred): mean pred^2."""
    return torch.mean(pred ** 2)


def _wavelet_pr_ac(dec_lo, dec_hi, rec_lo, rec_hi):
    """MLWNet LWN: perfect-reconstruction error + alias-cancellation error of one block's (1, L) filters."""
    pad = dec_lo.shape[-1] - 1
    p_lo = F.conv1d(dec_lo.flip(-1).unsqueeze(0), rec_lo.flip(-1).unsqueeze(0), padding=pad)
    pad = dec_hi.shape[-1] - 1
    p_hi = F.conv1d(dec_hi.flip(-1).unsqueeze(0), rec_hi.flip(-1).unsqueeze(0), padding=pad)
    p = p_lo + p_hi
    two = torch.zeros_like(p)
    two[..., p.shape[-1] // 2] = 2
    pr = torch.sum((p - two) * (p - two))
    length = dec_lo.shape[-1]
    mask = torch.tensor([(-1.0) ** n for n in range(length)][::-1], device=dec_lo.device, dtype=dec_lo.dtype)
    pad = dec_lo.shape[-1] - 1
    a_lo = F.conv1d(dec_lo.flip(-1).unsqueeze(0) * mask, rec_lo.flip(-1).unsqueeze(0), padding=pad)
    pad = dec_hi.shape[-1] - 1
    a_hi = F.conv1d(dec_hi.flip(-1).unsqueeze(0) * mask, rec_hi.flip(-1).unsqueeze(0), padding=pad)
    a = a_lo + a_hi
    return pr + torch.sum(a * a)


def wavelet_pr_ac_mlw(pred=None, target=None, *, model=None, wavelet_filters=None, **ctx):
    """MLWNet wavelet loss: sum over the learnable wavelet blocks of PR + AC (pred and target are not used)."""
    if wavelet_filters is None and model is not None:
        names = ("dec_lo", "dec_hi", "rec_lo", "rec_hi")
        wavelet_filters = [tuple(getattr(m, a) for a in names) for m in model.modules()
                           if all(isinstance(getattr(m, a, None), torch.Tensor) for a in names)]
    if not wavelet_filters:
        raise ValueError("wavelet_pr_ac_mlw needs ctx model= (with LWN blocks) or wavelet_filters=")
    total = 0.
    for f in wavelet_filters:
        total = total + _wavelet_pr_ac(*f)
    return total


# ----------------------------------------------------------------------------------- fixed pretrained networks
def _tv():
    import torchvision
    return torchvision


def _lpips_lin():
    sd = torch.hub.load_state_dict_from_url(LPIPS_URL, map_location="cpu", progress=False,
                                            file_name="lpips_v0.1_vgg.pth")
    return [sd[f"lin{k}.model.1.weight"] for k in range(5)]


# What the fixed networks are built from; tests replace entries with random weights.
BUILDERS = {
    "vgg19": lambda: _tv().models.vgg19(weights=_tv().models.VGG19_Weights.IMAGENET1K_V1).features,
    "vgg16": lambda: _tv().models.vgg16(weights=_tv().models.VGG16_Weights.IMAGENET1K_V1).features,
    "lpips_vgg_lin": _lpips_lin,
}
_NETS = {}


def _net(name, like):
    """A fixed network (eval, frozen, ReLUs not in place) or weight list on like's device and dtype, cached."""
    key = (name, like.device, like.dtype)
    if key not in _NETS:
        net = BUILDERS[name]()
        if isinstance(net, nn.Module):
            for m in net.modules():
                if isinstance(m, nn.ReLU):
                    m.inplace = False
            net = net.eval().requires_grad_(False).to(like.device, like.dtype)
        else:
            net = [t.to(like.device, like.dtype) for t in net]
        _NETS[key] = net
    return _NETS[key]


def _vgg_taps(x, taps, name="vgg19"):
    """Outputs of features[i] for each i in taps (torchvision indices, ReLUs applied out of place)."""
    net, out, h = _net(name, x), [], x
    for i in range(max(taps) + 1):
        h = net[i](h)
        if i in taps:
            out.append(h)
    return out


def content_dgv2(pred, target, **ctx):
    """DeblurGAN-v2 PerceptualLoss on its [-1, 1] outputs: x' = (x + 1) / 2, image 0 of the batch (only) replaced by
    its ImageNet normalisation; 0.006 * MSE(VGG19 features[:15] (conv3_3, pre-ReLU)) + 0.5 * MSE(x', y')."""
    fake, real = ((_img(pred) * 2 - 1) + 1) / 2.0, ((_img(target) * 2 - 1) + 1) / 2.0
    fake = torch.cat([_imagenet(fake[:1]), fake[1:]])
    real = torch.cat([_imagenet(real[:1]), real[1:]])
    f_fake = _vgg_taps(fake, [14])[0]
    with torch.no_grad():
        f_real = _vgg_taps(real, [14])[0]
    return 0.006 * torch.mean(F.mse_loss(f_fake, f_real)) + 0.5 * F.mse_loss(fake, real)


def perc_vgg19_conv5_4_l1_raw255_bgr(pred, target, **ctx):
    """VRNN PerceptualLoss: L1 mean of VGG19 features[:35] (conv5_4, pre-ReLU) of BGR images in 0..255."""
    p, y = _img(pred).flip(-3) * 255, _img(target).flip(-3) * 255
    fp = _vgg_taps(p, [34])[0]
    with torch.no_grad():
        fy = _vgg_taps(y, [34])[0]
    return F.l1_loss(fp, fy)


def perc_vgg19_r12_r22_c33_msesum(pred, target, **ctx):
    """SimDeblur PerceptualLossVGG19: 1 * SSE(relu1_2) + 0.2 * SSE(relu2_2) + 0.04 * SSE(conv3_3), RGB [0, 1].
    The authors store features[2] and features[7] (conv1_2, conv2_2), which the next in-place ReLU overwrites."""
    fp = _vgg_taps(_img(pred), [3, 8, 14])
    with torch.no_grad():
        fy = _vgg_taps(_img(target), [3, 8, 14])
    loss = 0
    for a, b, w in zip(fp, fy, (1, 0.2, 0.04)):
        loss += F.mse_loss(a, b, reduction="sum") * w
    return loss


def contrast_vgg19_relu31_masked(pred, target, *, blur, **ctx):
    """Stripformer ContrastLoss on [-0.5, 0.5] images: per-pixel ratio of channel-mean L1 distances of VGG19 relu3_1
    features, anchor pred, positive target, negative blur, averaged over the 1/4-scale mask of blurred pixels."""
    restore, sharp, blur = _img(pred) - 0.5, _img(target) - 0.5, _img(blur) - 0.5
    b, c, h, w = restore.size()
    restore_vgg = _vgg_taps(restore, [11])[0]
    with torch.no_grad():
        sharp_vgg, blur_vgg = _vgg_taps(sharp, [11])[0], _vgg_taps(blur, [11])[0]
        mask = torch.mean(torch.abs(sharp - blur), dim=1).view(b, 1, h, w)
        mask = (mask > 0.01).to(mask.dtype)
        mask = F.interpolate(mask, scale_factor=0.25, mode="bilinear", align_corners=False)
    d_ap = torch.mean(torch.abs(restore_vgg - sharp_vgg), dim=1).view(b, 1, h // 4, w // 4)
    d_an = torch.mean(torch.abs(restore_vgg - blur_vgg), dim=1).view(b, 1, h // 4, w // 4)
    return torch.sum((d_ap / (d_an + 1e-7)) * mask) / torch.sum(mask)


_LPIPS_SLICES = ((0, 4), (4, 9), (9, 16), (16, 23), (23, 30))


def _lpips_vgg(in0, in1):
    """lpips.LPIPS(net='vgg', version '0.1')(in0, in1) for [-1, 1] images: (N, 1, 1, 1). in0 runs without grad."""
    net, lins = _net("vgg16", in0), _net("lpips_vgg_lin", in0)
    shift, scale = _const((-.030, -.088, -.188), in0), _const((.458, .448, .450), in0)
    h0, h1 = (in0 - shift) / scale, (in1 - shift) / scale
    val = 0
    for k, (a, b) in enumerate(_LPIPS_SLICES):
        with torch.no_grad():
            for i in range(a, b):
                h0 = net[i](h0)
            n0 = h0 / (torch.sqrt(torch.sum(h0 ** 2, dim=1, keepdim=True)) + 1e-10)
        for i in range(a, b):
            h1 = net[i](h1)
        n1 = h1 / (torch.sqrt(torch.sum(h1 ** 2, dim=1, keepdim=True)) + 1e-10)
        val += F.conv2d((n0 - n1) ** 2, lins[k]).mean([2, 3], keepdim=True)
    return val


def lpips_vgg(pred, target, **ctx):
    """Mean LPIPS v0.1 (VGG16 trunk) between target and pred mapped to [-1, 1] (lpips.LPIPS(net='vgg'))."""
    return _lpips_vgg(_img(target) * 2 - 1, _img(pred) * 2 - 1).mean()


def _haar_slair(x):
    k = _cached("haar_slair", x, lambda: torch.tensor(
        [[[0.5, 0.5], [0.5, 0.5]], [[0.5, -0.5], [0.5, -0.5]], [[0.5, 0.5], [-0.5, -0.5]],
         [[0.5, -0.5], [-0.5, 0.5]]], dtype=torch.float32).unsqueeze(1))
    b, c, h, w = x.shape
    return F.conv2d(x.reshape(b * c, 1, h, w), k, stride=2).view(b, c, 4, h // 2, w // 2)


def haar_hf_l1_slair(pred, target, **ctx):
    """SLAIR WaveletLoss on [-1, 1]: L1 mean of the three high-frequency bands of a 2x2 stride-2 Haar transform."""
    return F.l1_loss(_haar_slair(_img(pred) * 2 - 1)[:, :, 1:], _haar_slair(_img(target) * 2 - 1)[:, :, 1:])


class FusionDecSLAIR:
    """SLAIR FusionLoss (decoder fine-tune, train_decoder.py:68-215 with the settings of :403-410): on [-1, 1],
    w_p * LPIPS_vgg(gt, rec) + w_m * MSE(rec, gt) + w_h * Haar-HF L1; the first `warmup` calls use the base weights
    (1.0, 0.5, 0.0), later calls scale each by its loss's running z-score (1 / (1 + 0.15 z) above the mean,
    1 + 0.075 |z| below), clamped to [0.3, 2] x base. Running means/deviations are kept as the authors' float32."""

    def __init__(self, mse_weight=1.0, perceptual_weight=0.5, high_freq_weight=0.0, momentum=0.99, warmup=1000):
        self.base = (mse_weight, perceptual_weight, high_freq_weight)
        self.momentum, self.warmup = momentum, warmup
        self.mean = [torch.tensor(0.0), torch.tensor(0.1), torch.tensor(0.01)]   # mse, lpips, hf
        self.std = [torch.tensor(1.0), torch.tensor(0.05), torch.tensor(0.01)]
        self.step_count = torch.tensor(0)
        self.weights = self.base

    def fresh(self):
        return FusionDecSLAIR(*self.base, momentum=self.momentum, warmup=self.warmup)

    def __call__(self, pred, target, **ctx):
        inputs, rec = (_img(target) * 2 - 1).float(), (_img(pred) * 2 - 1).float()
        mse = F.mse_loss(rec, inputs, reduction="mean")
        hf = F.l1_loss(_haar_slair(rec)[:, :, 1:], _haar_slair(inputs)[:, :, 1:])
        p_loss = _lpips_vgg(inputs, rec).mean()
        with torch.no_grad():
            self.step_count += 1
            n = self.step_count.item()
            vals = (mse.item(), p_loss.item(), hf.item())
            warm = self.step_count <= self.warmup
            alpha = (0.5 if n == 1 else 1.0 / n) if warm else None
            for i in range(3):
                if warm:
                    self.mean[i] = (1 - alpha) * self.mean[i] + alpha * vals[i]
                else:
                    self.mean[i] = self.momentum * self.mean[i] + (1 - self.momentum) * vals[i]
            diffs = [abs(vals[i] - self.mean[i].item()) for i in range(3)]
            for i in range(3):
                if warm:
                    self.std[i] = (1 - alpha) * self.std[i] + alpha * diffs[i]
                else:
                    self.std[i] = self.momentum * self.std[i] + (1 - self.momentum) * diffs[i]
        if warm:
            w = self.base
        else:
            w = []
            for i in range(3):
                z = (vals[i] - self.mean[i].item()) / (self.std[i].item() + 1e-6)
                adj = self.base[i] * (1.0 / (1.0 + 0.15 * z) if z > 0 else 1.0 + 0.15 * abs(z) * 0.5)
                w.append(max(self.base[i] * 0.3, min(adj, self.base[i] * 2.0)))
        self.weights = tuple(w)
        return w[1] * p_loss + w[0] * mse + w[2] * hf


class _MSDIEncoder(nn.Module):
    """The used part of MSDI-Net's ConvEncoderLoss: spectral-norm convs without bias + InstanceNorm, LeakyReLU 0.2.
    Attribute names follow the authors' so their net_E.pth loads (strict=False, as they load it)."""

    def __init__(self):
        super().__init__()

        def sn_in(cin, cout):
            conv = nn.utils.spectral_norm(nn.Conv2d(cin, cout, 3, stride=2, padding=1))
            delattr(conv, "bias")
            conv.register_parameter("bias", None)
            return nn.Sequential(conv, nn.InstanceNorm2d(cout, affine=False))

        self.layer1, self.layer2 = sn_in(3, 64), sn_in(64, 128)
        self.layer3, self.layer4 = sn_in(128, 256), sn_in(256, 512)
        self.conv_7x7 = nn.Conv2d(64, 64, kernel_size=7, padding=0, bias=True)

    def forward(self, x):
        def act(t):
            return F.leaky_relu(t, 0.2)

        x1 = self.layer1(x)
        x2 = self.conv_7x7(F.pad(act(x1), (3, 3, 3, 3), mode="reflect"))
        x3 = self.layer2(act(x2))
        x4 = self.layer3(act(x3))
        x5 = self.layer4(act(x4))
        return [x1, x2, x3, x4, x5]


class FeatL1MSDIEnc:
    """MSDI-Net l_blur: sum over the 5 encoder taps of L1 mean(E_k(target), E_k(pred)). As in the authors' training
    the encoder stays in train mode, so every call runs one spectral-norm power iteration per layer (target first,
    then pred); the state is kept between calls. ctx msdi_encoder= the authors' net_E.pth (path) or a state dict."""

    def __init__(self):
        self.enc, self.src = None, None

    def fresh(self):
        return FeatL1MSDIEnc()

    def __call__(self, pred, target, *, msdi_encoder=None, **ctx):
        same = msdi_encoder == self.src if isinstance(msdi_encoder, str) else msdi_encoder is self.src
        if self.enc is None or (msdi_encoder is not None and not same):
            if msdi_encoder is None:
                raise ValueError("feat_l1_msdi_enc needs ctx msdi_encoder= MSDI-Net's net_E.pth (path or state dict)")
            sd = torch.load(msdi_encoder, map_location="cpu") if isinstance(msdi_encoder, str) else msdi_encoder
            enc = _MSDIEncoder()
            enc.load_state_dict(sd, strict=False)
            self.enc, self.src = enc.requires_grad_(False).train(), msdi_encoder
        self.enc.to(pred.device)
        p, y = _img(pred), _img(target)
        with torch.no_grad():
            fy = self.enc(y)
        fp = self.enc(p)
        loss = 0.
        for a, b in zip(fy, fp):
            loss += F.l1_loss(a, b)
        return loss


# ------------------------------------------------------------------------------- masks, mining, network outputs
def hem_cdvd(pred, target, *, hem_rng=None, **ctx):
    """CDVD-TSP HEM: L1 mean of (x M, y M) over all elements; M = (residual > its value at rank 50 %) or one of
    exactly 10 % random pixels per sample (numpy shuffle of hem_rng, default np.random as the authors)."""
    x = pred.reshape(pred.shape[0], -1, *pred.shape[-2:])
    y = target.reshape(target.shape[0], -1, *target.shape[-2:])
    b, _, h, w = x.shape
    rng = np.random if hem_rng is None else hem_rng
    with torch.no_grad():
        res = torch.sum(torch.abs(x - y), dim=1, keepdim=True)
        thr = res.view(b, -1).sort(dim=1, descending=True)[0][:, int(0.5 * w * h)]
        hard = (res > thr.view(b, 1, 1, 1)).to(torch.float32)
        rand = np.zeros(shape=(b, h * w))
        for i in range(b):
            rand[i, :int(0.1 * w * h)] = 1.
            rng.shuffle(rand[i])
        rand = torch.from_numpy(rand.reshape(b, 1, h, w)).to(x.device, torch.float32)
        mask = ((hard + rand) > 0.).to(x.dtype)
    return F.l1_loss(x * mask, y * mask)


def bim_masked_mse_pvdnet(pred, target, *, warp_mask, **ctx):
    """PVDNet blur-invariant motion loss (BIMNet pre-training): mean over the four warps p of
    sum((W_p - y M_p)^2) / sum(M_p); pred (B, 4, C, H, W) = W_bb, W_bs, W_sb, W_ss, target y = the current GT."""
    errs = [F.mse_loss(pred[:, k], target * warp_mask[:, k], reduction="sum") / warp_mask[:, k].sum()
            for k in range(4)]
    return (errs[0] + errs[1] + errs[2] + errs[3]) / 4.


def reblur_sv_l1_ufp(pred, target, *, blur, kernel, **ctx):
    """UFPNet reblur loss: L1 mean(BatchBlur_SV(pred, kernel), blur), BatchBlur_SV = replicate pad l // 2, unfold
    l x l patches, multiply-sum with the per-pixel kernels (B, H*W, l, l) of the network's kernel branch."""
    b, c, h, w = pred.size()
    l = kernel.shape[-1]
    pad = F.pad(pred, (l // 2, l // 2, l // 2, l // 2), mode="replicate")
    hp, wp = pad.size()[-2:]
    pad = F.unfold(pad.view(c * b, 1, hp, wp), l).transpose(1, 2)
    k = kernel.flatten(2).unsqueeze(0).expand(3, -1, -1, -1)
    out_unf = (pad * k.contiguous().view(-1, k.size(2), k.size(3))).sum(2).unsqueeze(1)
    out = F.fold(out_unf, (h, w), 1).view(b, c, h, w)
    return F.l1_loss(out, blur)


def _warp_stdan(x, flo):
    b, _, h, w = x.size()
    xx = torch.arange(0, w).reshape(1, -1).repeat(h, 1).reshape(1, 1, h, w).repeat(b, 1, 1, 1)
    yy = torch.arange(0, h).reshape(-1, 1).repeat(1, w).reshape(1, 1, h, w).repeat(b, 1, 1, 1)
    vgrid = torch.cat((xx, yy), 1).float().to(x.device) + flo
    gx = 2.0 * vgrid[:, 0, :, :] / max(w - 1, 1) - 1.0
    gy = 2.0 * vgrid[:, 1, :, :] / max(h - 1, 1) - 1.0
    return F.grid_sample(x, torch.stack((gx, gy), -1), padding_mode="border", align_corners=True)


def warp_l1_stdan(pred, target, *, flows_forward, flows_backward, **ctx):
    """STDAN warp_loss (without its x0.05): GT clip (B, 5, C, H, W) bilinear to (H//4, W//4) (align_corners=True);
    for every triplet [0,1,2], [1,2,3], [2,3,4], [1,2,3] and every stage flow (B, 2, 2, H/4, W/4):
    F += L1(warp(f2, fwd), f1), B += L1(warp(f1, bwd), f2); returns (0.5 F + 0.5 B) / K. pred is not used."""
    b, t, c, h, w = target.shape
    g = F.interpolate(target.reshape(-1, c, h, w), size=(h // 4, w // 4), mode="bilinear",
                      align_corners=True).reshape(b, t, c, h // 4, w // 4)
    h, w = g.shape[-2:]
    forward_loss, backward_loss = 0, 0
    for idx in ([0, 1, 2], [1, 2, 3], [2, 3, 4], [1, 2, 3]):
        frames = g[:, idx, :, :, :]
        for flow_forward, flow_backward in zip(flows_forward, flows_backward):
            frames_1 = frames[:, :-1, :, :, :].reshape(-1, c, h, w)
            frames_2 = frames[:, 1:, :, :, :].reshape(-1, c, h, w)
            backward_frames = _warp_stdan(frames_1, flow_backward.reshape(-1, 2, h, w))
            forward_frames = _warp_stdan(frames_2, flow_forward.reshape(-1, 2, h, w))
            forward_loss += F.l1_loss(forward_frames, frames_1)
            backward_loss += F.l1_loss(backward_frames, frames_2)
    return (0.5 * forward_loss + 0.5 * backward_loss) / len(flows_forward)


def tur_unc_l1_5t(pred, target, *, uncertainty, de_id, **ctx):
    """TUR airnet_uncertainty_loss_5_types: classes 0..4, 6, 7; for each class present, s = 1 / (mean over its
    samples of the uncertainty map + 1e-6), L += L1(stack(s * gt), stack(s * pred)) + 2 ln(mean(mean map + 1e-6));
    the scaled lists are not reset between classes (as released); returns L / number of classes present."""
    cats = {k: ([], [], []) for k in (0, 1, 2, 3, 4, 6, 7)}
    for p, t, u, i in zip(_img(pred), _img(target), uncertainty.reshape(-1, *uncertainty.shape[-3:]), de_id):
        cats[int(i)][0].append(t)
        cats[int(i)][1].append(p)
        cats[int(i)][2].append(u)
    total, num, un_hq, un_lq = 0, 0, [], []
    for idx in range(7):
        lq_list, list_, un_list = cats[idx if idx < 5 else idx + 1]
        if lq_list:
            un_map = torch.mean(torch.stack(un_list), dim=0) + 1e-6
            s = 1.0 / un_map
            num += 1
            for u, v in zip(lq_list, list_):
                un_hq.append(torch.mul(v, s))
                un_lq.append(torch.mul(u, s))
            total += F.l1_loss(torch.stack(un_lq), torch.stack(un_hq), reduction="mean") + 2 * torch.log(torch.mean(un_map))
    return total / num


# ------------------------------------------------------------------------------------------------------ registry
LOSSES = {
    "l1_mean": l1_mean,
    "mse_mean": mse_mean,
    "psnr_naf": psnr_naf,
    "smoothl1_b0.01": smoothl1_b0_01,
    "masked_l1_ram": masked_l1_ram,
    "charb_e2_1e-3": charb_e2_1e_3,
    "charb_e_1e-12": charb_e_1e_12,
    "charb_e_1e-9": charb_e_1e_9,
    "charb_e_1e-6_sumps": charb_e_1e_6_sumps,
    "charb_sum_e_1e-6": charb_sum_e_1e_6,
    "charb_color_e2_1e-3": charb_color_e2_1e_3,
    "fft_l1_full": fft_l1_full,
    "fft_l1_rfft": fft_l1_rfft,
    "fft_cmod_full": fft_cmod_full,
    "fft_cmod_rfft": fft_cmod_rfft,
    "fft1d_h_charb_idblau": fft1d_h_charb_idblau,
    "edge_mpr": edge_mpr,
    "grad_l1_estrnn": grad_l1_estrnn,
    "hog_l1_hogformer": hog_l1_hogformer,
    "pearson_corr_hogformer": pearson_corr_hogformer,
    "msssim_pms": msssim_pms,
    "msssim_pms_v2018": msssim_pms_v2018,
    "tv_l1_aniso_mean": tv_l1_aniso_mean,
    "l2_act_reg_mean": l2_act_reg_mean,
    "wavelet_pr_ac_mlw": wavelet_pr_ac_mlw,
    "content_dgv2": content_dgv2,
    "perc_vgg19_conv5_4_l1_raw255_bgr": perc_vgg19_conv5_4_l1_raw255_bgr,
    "perc_vgg19_r12_r22_c33_msesum": perc_vgg19_r12_r22_c33_msesum,
    "contrast_vgg19_relu31_masked": contrast_vgg19_relu31_masked,
    "lpips_vgg": lpips_vgg,
    "haar_hf_l1_slair": haar_hf_l1_slair,
    "fusion_dec_slair": FusionDecSLAIR(),
    "feat_l1_msdi_enc": FeatL1MSDIEnc(),
    "hem_cdvd": hem_cdvd,
    "bim_masked_mse_pvdnet": bim_masked_mse_pvdnet,
    "reblur_sv_l1_ufp": reblur_sv_l1_ufp,
    "warp_l1_stdan": warp_l1_stdan,
    "tur_unc_l1_5t": tur_unc_l1_5t,
}
# components and variants that are not inventory ids of their own
EXTRAS = {"msssim_pms_v2018", "lpips_vgg", "haar_hf_l1_slair"}
ALIASES = {"l1": "l1_mean", "mse": "mse_mean", "psnr": "psnr_naf", "charbonnier": "charb_e_1e-12",
           "fft": "fft_cmod_rfft"}
LOSSES.update({a: LOSSES[i] for a, i in ALIASES.items()})

NOT_IN_LIBRARY = {
    "CHARB_UNSPEC": "LaKDNet: eps and reduction unpublished, no training code (nearest charb_e2_1e-3)",
    "FFT_L1_MAXIM_PAPER": "MAXIM: FFT kind, norm and reduction unpublished, no training code (nearest fft_l1_full)",
    "PERC_PREACT_DBGAN": "DBGAN: network, layer and distance unpublished, no training code",
    "VAE_DEC_RESTOREVAR": "RestoreVAR decoder: paper only, SSIM / perceptual variants unstated, PatchGAN term",
    "GAN_RAGAN_LS_DOUBLE_DGV2": "discriminators trained along with the network",
    "GAN_WGANGP_DOUBLE_DGV2": "critics trained along with the network (gradient penalty)",
    "GAN_RA_BCE_DBGAN": "discriminator trained along with the network",
    "VAE_LPIPS_GAN_SLAIR": "VAE training: KL, learned log-variance, hinge PatchGAN discriminator",
    "OT_RCOT": "optimal-transport dual with a critic trained in alternation",
    "BARY_WB_BARYIR": "barycenter dual with per-domain potential networks trained in alternation",
    "CLIP_INFONCE_SYM": "CLIP controller training, contrastive over image/text batches",
    "CE_RAWDOT_VLU": "degradation-adapter classifier on CLIP features",
    "CONTRAST_NTXENT_SPD_MIRAGE": "NT-Xent between learnable projection heads inside the network",
    "MSFS_HF_CONTRAST": "network-internal frequency features, no code released",
    "MSFS_LF_CONSIST": "network-internal frequency features, no code released",
    "CE_CORR_RTVDBLR": "cross-entropy on the network's correlation volume with LiteFlowNet targets",
    "CE_LS0.1": "degradation classifier cross-entropy",
    "CE_MEAN": "classifier cross-entropy",
    "FOCAL_CE_G2": "degradation classifier focal loss",
    "CE_TOKENS_VAR": "next-scale token prediction",
    "MASK_POLICY_LOGP_RAM": "policy term that trains the mask generator",
    "MOE_BALANCE_CV2": "mixture-of-experts router balance",
    "DIFFCHAIN_L1_DIFFIR": "diffusion-chain prior from a stage-1 network",
    "DIFFCHAIN_L1_HIDIFF": "diffusion-chain prior from a stage-1 network",
    "DIFF_IRSDE_MATCH": "SDE matching objective (noise-level sampling)",
    "DIFF_RESDIFF_PREDRES_L1": "residual diffusion objective (noise-level sampling)",
    "FM_BRIDGE_V_MSE_COS": "latent flow matching",
    "TINYDELTA_HETERO_SMOOTHL1": "separately trained trajectory predictor, not an image loss",
}


def make(spec):
    """A loss from a spec: ids of LOSSES (or aliases) joined by +, each with an optional weight: "l1_mean+0.1*fft_l1_full".
    The result is loss(pred, target, **ctx) -> sum of weight * term; stateful terms get their own state."""
    terms = []
    for part in spec.split("+"):
        w, _, name = part.strip().rpartition("*")
        name = name.strip().lower()
        if name not in LOSSES:
            raise ValueError(f"unknown loss {name!r}; known: {', '.join(sorted(LOSSES))}")
        f = LOSSES[name]
        terms.append((float(w) if w else 1.0, f.fresh() if hasattr(f, "fresh") else f))

    def loss(pred, target, **ctx):
        return sum(w * f(pred, target, **ctx) for w, f in terms)

    loss.spec, loss.terms = spec, terms
    return loss
