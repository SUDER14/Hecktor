"""Phase 2: shapes at every level, parameter counts, FiLM identity, gradients, B2 structure."""
import pytest
import torch

from src.models.acf_shared import ACF
from src.models.dual_encoder_shared import DualEncoderUNet
from src.models.evidential import EvidentialHead, dirichlet_summary
from src.models.hypernet import FiLMHypernet, HyperFuse1x1, film
from src.models.unet3d import UNet3D

CH, ST = [8, 16, 32, 64, 128], [2, 2, 2, 2]  # narrower than the 16..256 production setting: CPU-friendly
SP = (1.0, 1.0, 1.0)


def _batch(b=2, size=32):
    torch.manual_seed(0)
    return dict(image=torch.randn(b, 2, size, size, size), cond=torch.randn(b, 3), spacing=SP)


def _acf(**kw):
    kw.setdefault("channels", CH)
    return ACF(**kw)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_unet3d_shapes():
    m = UNet3D(CH, ST, out_channels=2)
    assert m(torch.randn(1, 1, 32, 32, 32)).shape == (1, 2, 32, 32, 32)
    with pytest.raises(ValueError):
        m(torch.randn(1, 1, 30, 32, 32))
    print("B1 UNet3D params", n_params(m))


def test_acf_output_shape_and_level_table():
    m = _acf()
    out = m(**_batch())
    assert out.shape == (2, 2, 32, 32, 32)
    rows = m.level_shapes((96, 96, 96))
    assert [r["shape"][0] for r in rows] == [96, 48, 24, 12, 6]  # one grid: CT and PET levels coincide
    print("ACF(small) params", n_params(m))
    print("ACF(production 16-256) params", n_params(ACF()), "| B2 production",
          n_params(DualEncoderUNet([16, 32, 64, 128, 256], ST, 2)), "| B1 production",
          n_params(UNet3D()))


def test_anisotropic_patch_runs():
    m = _acf()
    out = m(torch.randn(1, 2, 32, 32, 16), torch.randn(1, 3), (1.0, 1.0, 2.0))
    assert out.shape == (1, 2, 32, 32, 16)


def test_absurd_patch_extent_fails_loudly():
    with pytest.raises(ValueError, match="outside the allowed physical range"):
        _acf()(**{**_batch(), "spacing": (0.1, 0.1, 0.1)})  # a 3.2 mm box


def test_bad_input_shapes_fail_loudly():
    m = _acf()
    with pytest.raises(ValueError, match="channel 0 = CT"):
        m(torch.randn(2, 1, 32, 32, 32), torch.randn(2, 3))
    with pytest.raises(ValueError, match="cond must be"):
        m(torch.randn(2, 2, 32, 32, 32), torch.randn(2, 4))   # old 4-d vector (with modality flag)


def test_film_identity_at_init_and_exact_formula():
    h = FiLMHypernet(3, [8, 16])
    (g0, b0), (g1, b1) = h(torch.randn(3, 3))
    assert torch.equal(g0, torch.ones(3, 8)) and torch.equal(b0, torch.zeros(3, 8))
    assert g1.shape == (3, 16)
    x = torch.randn(3, 8, 4, 4, 4)
    assert torch.equal(film(x, g0, b0), x)
    g, b = torch.rand(3, 8) + 0.5, torch.randn(3, 8)
    assert torch.allclose(film(x, g, b)[1, 2, 0, 0, 0], g[1, 2] * x[1, 2, 0, 0, 0] + b[1, 2])


def test_conditioning_has_no_effect_at_init_but_zeroed_flag_matches():
    torch.manual_seed(1)
    m = _acf().eval()
    bt = _batch()
    with torch.no_grad():
        a = m(**bt)
        m.conditioning_zeroed = True
        z = m(**bt)
        m.conditioning_zeroed = False
        no_film = _acf(use_film=False)
        no_film.load_state_dict({k: v for k, v in m.state_dict().items() if "hyper" not in k})
        n = no_film.eval()(**bt)
    assert torch.allclose(a, z, atol=1e-6) and torch.allclose(a, n, atol=1e-6)  # identity FiLM == no FiLM


def test_zeroed_differs_from_conditioned_after_training_step():
    m = _acf()
    bt = _batch()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    loss = m(**bt).square().mean()
    loss.backward()
    opt.step()
    m.eval()
    with torch.no_grad():
        a = m(**bt)
        m.conditioning_zeroed = True
        z = m(**bt)
    assert not torch.allclose(a, z, atol=1e-7)  # the hypernet now actually uses the conditioning vector


def test_gradient_reaches_both_hypernetworks():
    m = _acf()
    m(**_batch()).square().mean().backward()
    for hyper in (m.ct_hyper, m.pet_hyper):
        assert hyper.head.weight.grad is not None and hyper.head.weight.grad.abs().sum() > 0
    # second step: after the head moves off zero, the hidden layers receive gradient too
    with torch.no_grad():
        for p in m.parameters():
            p -= 0.1 * (p.grad if p.grad is not None else 0)
    m.zero_grad()
    m(**_batch()).square().mean().backward()
    assert m.ct_hyper.mlp[0].weight.grad.abs().sum() > 0 and m.pet_hyper.mlp[0].weight.grad.abs().sum() > 0


def test_ct_and_pet_hypernetworks_have_independent_weights():
    m = _acf()
    with torch.no_grad():
        for hy in (m.ct_hyper, m.pet_hyper):
            hy.head.weight.normal_(0, 0.1)
    bt = _batch()
    base = m(**bt)
    changed = m(**{**bt, "cond": bt["cond"] + 3.0})
    assert not torch.allclose(base, changed)  # the spacing vector matters
    # same input, independent parameters: one hypernetwork per modality stream
    assert m.pet_hyper is not m.ct_hyper and m.pet_hyper.head.weight.data_ptr() != m.ct_hyper.head.weight.data_ptr()


def test_structure_matches_b2_without_the_hypernets():
    """ACF without the hypernets has exactly B2's parameters."""
    acf = ACF(channels=CH, use_film=False)
    b2 = DualEncoderUNet(CH, ST, 2)
    a, b = acf.state_dict(), b2.state_dict()
    assert {k: tuple(v.shape) for k, v in a.items()} == {k: tuple(v.shape) for k, v in b.items()}
    assert n_params(acf) == n_params(b2)
    extra = n_params(ACF(channels=CH)) - n_params(b2)
    print("FiLM hypernets add", extra, "parameters on the small model")
    assert extra > 0


def test_evidential_head_values():
    head = EvidentialHead(8, 2)
    ev = head(torch.randn(2, 8, 4, 4, 4))
    assert ev.shape == (2, 2, 4, 4, 4) and (ev >= 0).all()
    d = dirichlet_summary(torch.zeros(1, 2, 3, 3, 3))  # no evidence -> uniform p, u = 1
    assert torch.allclose(d["u"], torch.ones(1, 1, 3, 3, 3)) and torch.allclose(d["p"], torch.full((1, 2, 3, 3, 3), 0.5))
    d = dirichlet_summary(torch.tensor([8.0, 1.0]).reshape(1, 2, 1, 1, 1))  # alpha = [9, 2], S = 11
    assert d["S"].item() == 11.0 and d["u"].item() == pytest.approx(2 / 11)


def test_evidential_acf_and_k3_configurable():
    m = _acf(head="evidential")
    ev = m(**_batch())
    assert (ev >= 0).all() and ev.shape == (2, 2, 32, 32, 32)
    s = m.predict(**_batch())
    assert torch.allclose(s["p"].sum(1), torch.ones(2, 32, 32, 32), atol=1e-5)
    assert _acf(head="evidential", out_channels=3)(**_batch()).shape[1] == 3  # K is configurable
    assert _acf(out_channels=3)(**_batch()).shape[1] == 3


def test_tier2_kernels_identity_at_init_and_trainable():
    h = HyperFuse1x1(16, 8, 3)
    x = torch.randn(2, 16, 3, 3, 3)
    c = torch.randn(2, 3)
    assert torch.allclose(h(x, c), h.base(x))
    h(x, c).square().mean().backward()
    assert h.gen[-1].weight.grad.abs().sum() > 0
    m = _acf(tier2_kernels=True)
    assert m(**_batch()).shape == (2, 2, 32, 32, 32)
    assert _acf(tier2_kernels=False).fuse[2].__class__.__name__ == "Conv3d"  # off by default
