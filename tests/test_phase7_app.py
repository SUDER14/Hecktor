"""Viewer core + Streamlit page, on synthetic data, no GPU."""
from pathlib import Path

import pytest

from app import viewer_core as vc
from src.data.synthetic import SynthConfig, write_synthetic_hecktor2026


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = tmp_path_factory.mktemp("app") / "d"
    write_synthetic_hecktor2026(r, SynthConfig(fov_mm=(48, 48, 48), n_per_centre=1, seed=5, tumour_prob=1.0))
    return r


def test_load_case_puts_pet_on_ct_grid_and_reports_spacing(root):
    c = vc.load_case_from_dir(root, "CHUP-001")
    assert c.pet.shape == c.ct.shape and c.label.shape == c.ct.shape
    assert c.ct_spacing == pytest.approx((1, 1, 3)) and c.pet_spacing == pytest.approx((4, 4, 4))


def test_demo_prediction_metrics_and_render_all_planes(root, tmp_path):
    c = vc.demo_prediction(vc.load_case_from_dir(root, "CHUM-001"))
    assert c.source == "demo" and c.notes and c.unc.min() >= 0 and c.unc.max() <= 1
    m = vc.case_metrics(c)
    assert set(m) >= {"Dice", "HD95 (mm)", "Surface Dice @2mm"}
    for plane in vc.PLANES:
        fig = vc.render(c, plane, vc.axis_length(c, plane) // 2)
        p = tmp_path / f"{plane}.png"
        fig.savefig(p)
        assert p.stat().st_size > 2000


def test_case_without_label_has_no_metrics(root):
    c = vc.demo_prediction(vc.load_case_from_dir(root, "CHUM-001"))
    c.label = None
    assert vc.case_metrics(c) is None
    assert vc.render(c, "axial", 5) is not None


def test_streamlit_page_runs_headless_on_synthetic_data():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("DEMO" in w.value for w in at.warning)          # demo prediction is labelled as such
    labels = [m.label for m in at.metric]
    assert "CT spacing (mm)" in labels and "PET spacing (mm)" in labels
    at.sidebar.radio[1].set_value("coronal").run()              # switch plane
    assert not at.exception
