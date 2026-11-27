"""Checks of dbis.losses that need neither the cluster nor downloaded weights (pytest, CPU).

Every loss: a finite scalar and a finite gradient on (B, C, H, W) and, where it applies, (B, T, C, H, W); known
closed-form values; make() specs, aliases and the train.py recipes. The fixed networks (VGG19, VGG16, LPIPS linear
layers) are replaced by randomly initialised ones, so no weights are downloaded. The comparison with the authors' code
is tests/cluster_compare_losses.py.
"""
import math

import pytest
import torch

from dbis import losses as L

tv = pytest.importorskip("torchvision")

IDS = sorted(set(L.LOSSES) - set(L.ALIASES))


@pytest.fixture(autouse=True)
def random_nets(monkeypatch):
    """Random-weight stand-ins for the pretrained networks (deterministic, no download)."""
    def vgg(make):
        def build():
            torch.manual_seed(0)
            return make(weights=None).features
        return build

    def lins():
        g = torch.Generator().manual_seed(0)
        return [torch.rand(1, c, 1, 1, generator=g) for c in (64, 128, 256, 512, 512)]

    monkeypatch.setitem(L.BUILDERS, "vgg19", vgg(tv.models.vgg19))
    monkeypatch.setitem(L.BUILDERS, "vgg16", vgg(tv.models.vgg16))
    monkeypatch.setitem(L.BUILDERS, "lpips_vgg_lin", lins)
    monkeypatch.setattr(L, "_NETS", {})
    yield


def pair(shape, seed=0, noise=0.05):
    g = torch.Generator().manual_seed(seed)
    y = torch.rand(shape, generator=g)
    p = (y + noise * torch.randn(shape, generator=g)).clamp(0, 1)
    return p, y


def case(name, video=False):
    """(pred, target, ctx, tensors whose gradient is checked) for one loss."""
    g = torch.Generator().manual_seed(1)
    shape = (2, 2, 3, 32, 32) if video else (2, 3, 32, 32)
    p, y = pair(shape)
    ctx = {}
    if name == "masked_l1_ram":
        ctx["mask"] = (torch.rand(*shape[:-3], 1, *shape[-2:], generator=g) > 0.5).float()
    elif name == "contrast_vgg19_relu31_masked":
        ctx["blur"] = (y + 0.2 * torch.randn(shape, generator=g)).clamp(0, 1)
    elif name in ("tv_l1_aniso_mean", "l2_act_reg_mean"):
        p, y = torch.randn(2, 15, 2, 16, 16, generator=g), None
    elif name == "wavelet_pr_ac_mlw":
        f = [torch.randn(1, 2, generator=g).requires_grad_() for _ in range(4)]
        ctx["wavelet_filters"] = [tuple(f)]
        return p, y, ctx, f
    elif name == "feat_l1_msdi_enc":
        torch.manual_seed(0)
        ctx["msdi_encoder"] = L._MSDIEncoder().state_dict()
    elif name == "bim_masked_mse_pvdnet":
        p, y = torch.rand(2, 4, 3, 32, 32, generator=g), torch.rand(2, 3, 32, 32, generator=g)
        ctx["warp_mask"] = (torch.rand(2, 4, 1, 32, 32, generator=g) > 0.2).float()
    elif name == "reblur_sv_l1_ufp":
        p, y = pair((2, 3, 16, 16))
        ctx["blur"] = torch.rand(2, 3, 16, 16, generator=g)
        ctx["kernel"] = torch.softmax(torch.randn(2, 256, 19 * 19, generator=g), -1).view(2, 256, 19, 19)
        ctx["kernel"].requires_grad_()
    elif name == "warp_l1_stdan":
        y = torch.rand(2, 5, 3, 32, 32, generator=g)
        fl = [torch.randn(2, 2, 2, 8, 8, generator=g).requires_grad_() for _ in range(8)]
        ctx["flows_forward"], ctx["flows_backward"] = fl[:4], fl[4:]
        return p, y, ctx, fl
    elif name == "tur_unc_l1_5t":
        p, y = pair((6, 3, 16, 16))
        ctx["uncertainty"] = (torch.rand(6, 3, 16, 16, generator=g) + 0.5).requires_grad_()
        ctx["de_id"] = torch.tensor([0, 3, 3, 6, 7, 0])
        p.requires_grad_()
        return p, y, ctx, [p, ctx["uncertainty"]]
    p.requires_grad_()
    return p, y, ctx, [p] + ([ctx["kernel"]] if "kernel" in ctx else [])


VIDEO = set(IDS) - {"tv_l1_aniso_mean", "l2_act_reg_mean", "wavelet_pr_ac_mlw", "bim_masked_mse_pvdnet",
                    "reblur_sv_l1_ufp", "warp_l1_stdan", "tur_unc_l1_5t"}


@pytest.mark.parametrize("video", [False, True])
@pytest.mark.parametrize("name", IDS)
def test_finite_value_and_gradient(name, video):
    if video and name not in VIDEO:
        pytest.skip("loss defined on its own tensor layout")
    p, y, ctx, leaves = case(name, video)
    f = L.LOSSES[name].fresh() if hasattr(L.LOSSES[name], "fresh") else L.LOSSES[name]
    v = f(p, y, **ctx)
    assert v.dim() == 0 and torch.isfinite(v), (name, v)
    v.backward()
    for t in leaves:
        assert t.grad is not None and torch.isfinite(t.grad).all(), name


def test_inventory_covered():
    implemented = set(IDS) - L.EXTRAS
    assert len(implemented) == 35 and len(L.NOT_IN_LIBRARY) == 28
    assert len(implemented) + len(L.NOT_IN_LIBRARY) == 63            # the inventory's 63 distinct losses
    assert not {i.lower() for i in L.NOT_IN_LIBRARY} & implemented


def test_pixel_closed_forms():
    y = torch.rand(2, 3, 8, 8)
    p = y + 0.5
    assert math.isclose(L.LOSSES["l1_mean"](p, y).item(), 0.5, rel_tol=1e-5)
    assert math.isclose(L.LOSSES["mse_mean"](p, y).item(), 0.25, rel_tol=1e-5)
    assert math.isclose(L.LOSSES["psnr_naf"](p, y).item(), 10 * math.log10(0.25 + 1e-8), rel_tol=1e-5)
    assert math.isclose(L.LOSSES["charb_e2_1e-3"](y, y).item(), 1e-3, rel_tol=1e-4)
    assert math.isclose(L.LOSSES["charb_e_1e-12"](y, y).item(), 1e-6, rel_tol=1e-4)
    assert math.isclose(L.LOSSES["charb_e_1e-9"](y, y).item(), math.sqrt(1e-9), rel_tol=1e-4)
    assert math.isclose(L.LOSSES["charb_e_1e-6_sumps"](y, y).item(), 1e-3 * 3 * 64, rel_tol=1e-4)
    assert math.isclose(L.LOSSES["charb_sum_e_1e-6"](y, y).item(), 1e-3 * 2 * 3 * 64, rel_tol=1e-4)
    assert math.isclose(L.LOSSES["charb_color_e2_1e-3"](p, y).item(), math.sqrt(0.25 + 1e-6), rel_tol=1e-5)
    # ImageNet space: a [0, 1] offset of 0.5 is 0.5 / std_c after normalisation
    want = sum(0.5 / s for s in L.IMAGENET_STD) / 3
    assert math.isclose(L.LOSSES["smoothl1_b0.01"](p, y).item(), want - 0.005, rel_tol=1e-5)
    m = torch.zeros(2, 1, 8, 8)
    m[:, :, :4] = 1
    assert math.isclose(L.LOSSES["masked_l1_ram"](p, y, mask=m).item(), want * 64 / (64 + 1e-5), rel_tol=1e-5)


def test_fft_closed_forms():
    """A constant offset d only moves the DC bin, d * H * W."""
    y = torch.rand(1, 3, 8, 10)
    p, d, n = y + 0.5, 0.5, 8 * 10
    assert math.isclose(L.LOSSES["fft_cmod_full"](p, y).item(), d, rel_tol=1e-4)
    assert math.isclose(L.LOSSES["fft_l1_full"](p, y).item(), d / 2, rel_tol=1e-4)
    assert math.isclose(L.LOSSES["fft_cmod_rfft"](p, y).item(), d * n / (8 * 6), rel_tol=1e-4)
    assert math.isclose(L.LOSSES["fft_l1_rfft"](p, y).item(), d * n / (2 * 8 * 6), rel_tol=1e-4)
    # 1-D fft along H: each column's DC (one of 8 bins x 2 parts) is d * H; Charbonnier of 0 elsewhere is eps
    v = (d * 8) / 16 + (15 / 16) * 1e-3
    assert math.isclose(L.LOSSES["fft1d_h_charb_idblau"](p, y).item(), v, rel_tol=1e-3)


def test_structure_closed_forms():
    y = torch.rand(2, 3, 32, 32)
    assert math.isclose(L.LOSSES["edge_mpr"](y, y).item(), 1e-3, rel_tol=1e-4)
    assert L.LOSSES["grad_l1_estrnn"](y, y).item() == 0
    # a constant image: zero gradient inside, the zero padding makes the 2-pixel border ring |c| or sqrt(2)|c|
    c = torch.full((1, 1, 6, 6), 0.5)
    g = L._grad_mag_estrnn(c)
    assert g.shape == (1, 1, 8, 8) and math.isclose(g[0, 0, 4, 4].item(), 1e-3, rel_tol=1e-4)
    assert math.isclose(g[0, 0, 0, 4].item(), math.sqrt(0.25 + 1e-6), rel_tol=1e-5)
    assert L.LOSSES["hog_l1_hogformer"](y, y).item() == 0
    assert L.LOSSES["pearson_corr_hogformer"](2 * y + 1, y).item() < 1e-6
    assert math.isclose(L.LOSSES["pearson_corr_hogformer"](-y, y).item(), 1.0, rel_tol=1e-5)
    assert abs(L.LOSSES["msssim_pms"](y, y).item()) < 1e-6
    assert abs(L.LOSSES["msssim_pms_v2018"](y, y).item()) < 1e-6
    z = torch.full((2, 15, 2, 8, 8), 2.0)
    assert L.LOSSES["tv_l1_aniso_mean"](z, None).item() == 0
    assert L.LOSSES["l2_act_reg_mean"](z, None).item() == 4.0


def test_wavelet_haar_is_perfect():
    r = 1 / math.sqrt(2)
    haar = [torch.tensor([[r, r]]), torch.tensor([[-r, r]]), torch.tensor([[r, r]]), torch.tensor([[r, -r]])]
    # MLWNet stores dec flipped and rec flipped twice (wavelet_block.py:186-194)
    dec_lo, dec_hi, rec_lo, rec_hi = haar[0].flip(-1), haar[1].flip(-1), haar[2], haar[3]
    v = L.LOSSES["wavelet_pr_ac_mlw"](wavelet_filters=[(dec_lo, dec_hi, rec_lo, rec_hi)])
    assert v.item() < 1e-10
    lwn = torch.nn.Module()
    for n, t in zip(("dec_lo", "dec_hi", "rec_lo", "rec_hi"), (dec_lo, dec_hi, rec_lo, rec_hi)):
        setattr(lwn, n, torch.nn.Parameter(t))
    assert L.LOSSES["wavelet_pr_ac_mlw"](None, None, model=torch.nn.Sequential(lwn, lwn)).item() < 1e-10


def test_network_losses_zero_on_identical():
    y = torch.rand(3, 3, 32, 32)
    for name in ("perc_vgg19_conv5_4_l1_raw255_bgr", "perc_vgg19_r12_r22_c33_msesum", "lpips_vgg",
                 "haar_hf_l1_slair"):
        assert L.LOSSES[name](y, y).item() == 0, name
    assert L.LOSSES["content_dgv2"](y, y).item() == 0
    blur = (y + 0.3).clamp(0, 1)
    assert L.LOSSES["contrast_vgg19_relu31_masked"](y, y, blur=blur).item() == 0


def test_content_dgv2_normalises_only_image_0():
    """The pixel term: image 0 in ImageNet units, the other images in [0, 1] units."""
    y = torch.rand(2, 3, 16, 16)
    p = y.clone()
    p[1] += 0.2
    want = 0.5 * 0.04 / 2                                        # 0.5 * mean over both images of 0.2^2
    feat = L.LOSSES["content_dgv2"](p, y).item() - want
    assert feat > 0
    p = y.clone()
    p[0] += 0.2
    want0 = 0.5 * sum((0.2 / s) ** 2 for s in L.IMAGENET_STD) / 3 / 2
    assert L.LOSSES["content_dgv2"](p, y).item() > want0 * 0.999


def test_fusion_dec_slair_base_weights():
    p, y = pair((2, 3, 32, 32))
    f = L.make("fusion_dec_slair")
    want = 4 * L.LOSSES["mse_mean"](p, y) + 0.5 * L.LOSSES["lpips_vgg"](p, y)
    assert torch.allclose(f(p, y), want, rtol=1e-5)
    assert f.terms[0][1].step_count.item() == 1 and L.LOSSES["fusion_dec_slair"].step_count.item() == 0


def test_feat_l1_msdi_enc_needs_weights():
    with pytest.raises(ValueError):
        L.make("feat_l1_msdi_enc")(*pair((1, 3, 16, 16)))


def test_hem_cdvd():
    y = torch.randint(0, 256, (2, 3, 16, 16)) / 256
    assert L.LOSSES["hem_cdvd"](y, y).item() == 0
    p = y + 1.0                                                  # every residual exactly 3: only the random 10 % count
    rng = __import__("numpy").random.RandomState(0)
    assert math.isclose(L.LOSSES["hem_cdvd"](p, y, hem_rng=rng).item(), int(0.1 * 256) / 256, rel_tol=1e-6)


def test_bim_reblur_warp_tur_closed_forms():
    y = torch.rand(2, 3, 16, 16)
    w = y.unsqueeze(1).repeat(1, 4, 1, 1, 1)
    assert L.LOSSES["bim_masked_mse_pvdnet"](w, y, warp_mask=torch.ones(2, 4, 1, 16, 16)).item() == 0
    k = torch.zeros(2, 256, 19, 19)
    k[:, :, 9, 9] = 1                                            # delta kernels: B_K(x) = x
    blur = torch.rand(2, 3, 16, 16)
    assert math.isclose(L.LOSSES["reblur_sv_l1_ufp"](y, None, blur=blur, kernel=k).item(),
                        (y - blur).abs().mean().item(), rel_tol=1e-5)
    clip = torch.rand(1, 1, 3, 32, 32).repeat(1, 5, 1, 1, 1)     # static clip, zero flow: nothing to warp
    zero = [torch.zeros(1, 2, 2, 8, 8)] * 4
    assert L.LOSSES["warp_l1_stdan"](None, clip, flows_forward=zero, flows_backward=zero).item() < 1e-6
    p, y = pair((3, 3, 8, 8))
    v = L.LOSSES["tur_unc_l1_5t"](p, y, uncertainty=torch.ones(3, 3, 8, 8), de_id=torch.tensor([3, 3, 3]))
    s = 1 / (1 + 1e-6)
    assert math.isclose(v.item(), s * (p - y).abs().mean().item() + 2 * math.log(1 + 1e-6), rel_tol=1e-5)


def test_make_specs_and_aliases():
    p, y = pair((2, 3, 16, 16))
    assert torch.equal(L.make("l1")(p, y), L.l1_mean(p, y))
    assert torch.equal(L.make("psnr")(p, y), L.psnr_naf(p, y))
    dst = L.make("charbonnier+0.1*fft")(p, y)
    assert torch.allclose(dst, L.charb_e_1e_12(p, y) + 0.1 * L.fft_cmod_rfft(p, y))
    assert torch.equal(dst, L.make("charb_e_1e-12+0.1*fft_cmod_rfft")(p, y))
    assert torch.allclose(L.make("l1_mean + 0.1*fft_l1_full")(p, y), L.l1_mean(p, y) + 0.1 * L.fft_l1_full(p, y))
    blur = torch.rand(2, 3, 16, 16)                              # one ctx serves every term
    v = L.make("charb_e2_1e-3+0.05*edge_mpr+0.0005*contrast_vgg19_relu31_masked")(p, y, blur=blur)
    assert torch.isfinite(v)
    with pytest.raises(ValueError):
        L.make("l1+nope")


def test_train_recipes_give_the_authors_losses():
    """The loss specs of dbis/train.py RECIPES resolve to the authors' losses."""
    want = {"psnr": L.psnr_naf, "l1": L.l1_mean}
    for spec, f in want.items():
        assert L.make(spec).terms == [(1.0, f)]
    assert L.make("charbonnier+0.1*fft").terms == [(1.0, L.charb_e_1e_12), (0.1, L.fft_cmod_rfft)]
    for spec in ("psnr_naf", "l1_mean", "charb_e_1e-12+0.1*fft_cmod_rfft"):
        L.make(spec)
