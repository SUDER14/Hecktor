"""ACF Studio, stage 1: licence rules, the three endpoints, the display volume, cold start.

Tests that need the real cache use TRAINING cases only (never a test case) and are skipped where the cache is
absent. No test writes a patient image anywhere.
"""
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import tracemalloc
import urllib.request
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="ACF Studio needs fastapi (app/studio/requirements.txt)")
from fastapi.testclient import TestClient

from app.studio import cases, server
from app.studio.settings import HOST, Settings, load_settings
from src.data import crop_cache
from src.data.acf_dataset import (
    PET_CLIP_MAX,
    AcfVolumeDataset,
    _normalise,
    load_sidecar,
    pet_scale,
)
from src.data.conditioning import load_stats
from src.data.splits import load_split

ROOT = Path(__file__).resolve().parents[1]
REAL = Path("D:/hecktor_cache")
real = pytest.mark.skipif(not (REAL / "CHUM-001.acf").exists(), reason="real cache not built")
BASE = f"http://{HOST}:8765"
WATERMARK = "Research use only. HECKTOR data. Not for publication."
# z spread and the PET-scale block: median slice spacing, thickest slices, one MDA-402..447 case. All training cases.
THREE = ["MDA-071", "MDA-284", "MDA-410"]


# ---------------------------------------------------------------- a tiny synthetic cache (no patient data)

def _write_case(cache: Path, pid: str, shape, spacing, signs=(1, 1, 1), seed=0):
    rng = np.random.default_rng(seed)
    ct = rng.integers(-1000, 1500, size=shape).astype(np.int16)
    pet = (rng.random(shape) * 12).astype(np.float16)
    lab = np.zeros(shape, np.uint8)
    lab[3, 5, 2] = 1                      # single voxels: stride sampling would lose them, block maximum must not
    lab[shape[0] - 1, shape[1] - 1, 1] = 2
    aff = np.diag([signs[0] * spacing[0], signs[1] * spacing[1], signs[2] * spacing[2], 1.0])
    header = {"patient_id": pid, "centre": pid.split("-")[0], "orig_shape": list(shape), "box": [[0, n] for n in shape],
              "shape": list(shape), "affine": aff.tolist(), "spacing": list(spacing), "orientation": "RAS", "sig": {},
              "pet_max": float(pet.max()), "label_values": [0, 1, 2]}
    crop_cache.write_case(cache / f"{pid}.acf", header, {"ct": ct, "pet": pet, "label": lab})
    return ct, pet, lab


@pytest.fixture()
def toy(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    arrs = {"AAA-001": _write_case(cache, "AAA-001", (301, 280, 9), (0.5, 0.5, 3.0), seed=1),          # f = 2
            "BBB-002": _write_case(cache, "BBB-002", (40, 64, 7), (1.2, 1.2, 5.0), signs=(-1, 1, -1), seed=2)}
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["AAA-001"], "val": [], "test": ["BBB-002"], "_meta": {}}), encoding="utf-8")
    meta = tmp_path / "scan.csv"
    meta.write_text("patient_id,pet_max\nAAA-001,18.5\nBBB-002,52000\n", encoding="utf-8")
    return Settings(cache_dir=cache, split_path=split, meta_csv=meta), arrs


def _client(settings) -> TestClient:
    return TestClient(server.create_app(settings), base_url=BASE)


# ---------------------------------------------------------------- licence rules

def test_server_binds_to_loopback_only():
    assert HOST == "127.0.0.1"
    src = (ROOT / "app/studio/server.py").read_text(encoding="utf-8")
    assert "uvicorn.run(create_app(s), host=HOST" in src
    assert "0.0.0.0" not in src and "ACF_STUDIO_HOST" not in (ROOT / "app/studio/settings.py").read_text(encoding="utf-8")


def test_no_served_file_points_at_another_host():
    files = server.served_files()
    assert any(f.name == "index.html" for f in files) and any(f.suffix == ".woff2" for f in files)
    bad = []
    for f in files:
        for m in re.finditer(rb"https?://[^\s\"'<>)]*", f.read_bytes()):
            url = m.group(0).decode("latin-1")
            if not re.match(r"https?://(127\.0\.0\.1|localhost)([:/]|$)", url):
                bad.append(f"{f.relative_to(ROOT)}: {url}")
    assert not bad, bad
    page = (server.STATIC / "index.html").read_text(encoding="utf-8")
    assert "fonts.googleapis.com" not in page and "<script>" not in page    # no CDN, no inline script (CSP 'self')


def test_fonts_are_vendored_with_licence_and_pinned_version():
    v = server.STATIC / "vendor"
    assert len(list(v.glob("saira-semi-condensed/*.woff2"))) == 4 and len(list(v.glob("ibm-plex-mono/*.woff2"))) == 2
    for d in ("saira-semi-condensed", "ibm-plex-mono"):
        assert "SIL Open Font License" in (v / d / "LICENSE").read_text(encoding="utf-8")
    versions = (v / "VERSIONS.txt").read_text(encoding="utf-8")
    assert "@fontsource/saira-semi-condensed, version 5.3.0" in versions and "@fontsource/ibm-plex-mono, version 5.3.0" in versions


PATIENT_FILE = re.compile(r"\.(nii|nii\.gz|acf|npy|npz|dcm|mha|mhd|nrrd|png|jpe?g|gif|bmp|tiff?|webp|pt|pth)$", re.IGNORECASE)


def test_no_patient_image_cache_prediction_or_screenshot_is_tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    assert out, "git ls-files returned nothing"
    assert not [f for f in out if PATIENT_FILE.search(f)]
    assert not [f for f in out if f.startswith(("results/", "data/")) or "studio_shots" in f or "test_predictions" in f]
    assert not [f for f in out if f.lower().endswith(".csv")]            # no clinical or per-case table
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for rule in ("*.nii.gz", "*.acf", "studio_shots/", "test_predictions/", "app/studio/**/*.png"):
        assert rule in ignore, rule


def test_only_local_read_only_requests_are_answered(toy):
    c = _client(toy[0])
    r = c.get("/")
    assert r.status_code == 200 and "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["cache-control"] == "no-store" and "access-control-allow-origin" not in r.headers
    assert c.get("/api/index", headers={"host": "evil.example"}).status_code == 403     # DNS rebinding
    assert c.get("/api/index", headers={"host": "192.168.1.20:8765"}).status_code == 403
    assert c.get("/api/index", headers={"host": "localhost:8765"}).status_code == 200
    for method in ("post", "put", "delete", "patch"):
        assert getattr(c, method)("/api/index").status_code == 405
    for path in ("/docs", "/redoc", "/openapi.json", "/static/vendor/ibm-plex-mono/LICENSE", "/static/vendor/VERSIONS.txt",
                 "/static/../server.py", "/static/..%2f..%2fserver.py", "/reference/acf_studio.html",
                 "/static/reference/acf_studio.html", "/api/cases/..%2f..%2fsecret/volume", "/api/cases/NOPE-1/volume"):
        assert c.get(path).status_code == 404, path


def test_no_export_control_and_real_views_carry_the_watermark_and_do_not_print():
    page = (server.STATIC / "index.html").read_text(encoding="utf-8")
    js = (server.STATIC / "js/studio.js").read_text(encoding="utf-8")
    css = (server.STATIC / "css/studio.css").read_text(encoding="utf-8")
    for needle in ("toDataURL", "toBlob", "download", "createObjectURL", "showSaveFilePicker", "type=\"file\"", "clipboard"):
        assert needle not in js and needle not in page, needle
    assert page.count(WATERMARK) >= 2 and f"var WATERMARK = '{WATERMARK}'" in js
    assert "watermark(ctx, CW, Hc)" in js and "classList.toggle('real', V.real)" in js
    assert "body.real .wm, body.real .wmbar { display: block; }" in css
    printed = css[css.index("@media print"):]
    assert "body.real .stage" in printed and "display: none !important" in printed
    routes = {(m, r.path) for r in server.create_app(Settings(Path("x"), Path("y"), Path("z"))).routes for m in getattr(r, "methods", [])}
    assert {m for m, _ in routes} <= {"GET", "HEAD"}
    assert {p for _, p in routes} == {"/", "/static/{rel:path}", "/api/index", "/api/cases/{pid}", "/api/cases/{pid}/volume"}


def test_demo_rule_never_runs_on_a_real_case():
    js = (server.STATIC / "js/studio.js").read_text(encoding="utf-8")
    assert "if (V.real || !V.hasPet) return { ok: false };" in js           # no demo metrics
    assert "o.unc && demo" in js and "o.pred && demo" in js                 # no demo overlay
    assert "REAL_NAMES = { 0: 'background (0)', 1: 'label 1', 2: 'label 2' }" in js
    assert "units not stated" in js


# ---------------------------------------------------------------- endpoints on the synthetic cache

def test_flips_follow_the_affine_and_rotated_affines_are_refused():
    assert cases.flips_from_affine(np.diag([1.0, 2.0, 3.0, 1.0])) == [True, True, True]
    assert cases.flips_from_affine(np.diag([-1.0, 2.0, -3.0, 1.0])) == [False, True, False]
    rot = np.eye(4)
    rot[0, 1] = 0.2
    with pytest.raises(ValueError):
        cases.flips_from_affine(rot)


def test_index_lists_every_case_with_split_scale_and_outlier_flag(toy):
    s, _ = toy
    ix = _client(s).get("/api/index").json()
    assert ix["n_cases"] == 2 and ix["centres"] == ["AAA", "BBB"]
    assert ix["split_counts"] == {"train": 1, "validation": 0, "test": 1} and ix["n_pet_outliers"] == 1
    a, b = ix["cases"]
    assert (a["patient_id"], a["split"], a["is_test"], a["pet_outlier"]) == ("AAA-001", "train", False, False)
    assert (b["patient_id"], b["split"], b["is_test"], b["pet_outlier"]) == ("BBB-002", "test", True, True)
    assert (a["sx"], a["sz"]) == (0.5, 3.0) and a["pet_scale"] is None
    assert [m["file"] for m in ix["missing"]] == [str(s.cache_dir / "pet_scale.json")]     # named, not guessed
    assert ix["test_note"] == "test set: view only, not for tuning"


def test_missing_sources_are_named_and_their_fields_left_empty(tmp_path, toy):
    s, _ = toy
    gone = Settings(cache_dir=s.cache_dir, split_path=tmp_path / "no_split.json", meta_csv=tmp_path / "no_scan.csv")
    ix = cases.build_index(gone)
    files = [m["file"] for m in ix["missing"]]
    assert str(gone.split_path) in files and str(gone.meta_csv) in files
    assert all(c["split"] is None and c["pet_outlier"] is None for c in ix["cases"]) and ix["n_pet_outliers"] is None
    empty = cases.build_index(Settings(cache_dir=tmp_path / "nowhere", split_path=s.split_path, meta_csv=s.meta_csv))
    assert empty["n_cases"] == 0 and "<case>.acf" in empty["missing"][0]["file"]


def test_display_volume_shapes_spacing_flips_and_values(toy):
    s, arrs = toy
    c = _client(s)
    for pid, (ct, pet, lab) in arrs.items():
        r = c.get(f"/api/cases/{pid}/volume")
        assert r.status_code == 200 and r.headers["content-type"] == "application/octet-stream"
        assert "content-disposition" not in r.headers                    # nothing invites a download
        h, a = cases.unpack_volume(r.content)
        nx, ny, nz = ct.shape
        f = cases.reduction_factor(nx, ny)
        assert f == (2 if pid == "AAA-001" else 1) and h["reduction"] == f
        assert (h["nx"], h["ny"], h["nz"]) == (math.ceil(nx / f), math.ceil(ny / f), nz)
        assert max(h["nx"], h["ny"]) <= cases.MAX_INPLANE
        meta = c.get(f"/api/cases/{pid}").json()
        sp = meta["spacing"]
        assert h["sx"] == sp[0] * f and h["sy"] == sp[1] * f and h["sz"] == sp[2]     # true spacing after the stride
        assert h["flip"] == meta["flip"] == ([True, True, True] if pid == "AAA-001" else [False, True, False])
        # physical extent, hence the drawn aspect ratio, is preserved to within one reduced voxel per axis
        assert 0 <= h["nx"] * h["sx"] - nx * sp[0] < h["sx"] and 0 <= h["ny"] * h["sy"] - ny * sp[1] < h["sy"]
        assert h["nz"] * h["sz"] == nz * sp[2]
        scale = pet_scale(ct, pet)
        assert h["pet"]["scale"] == scale and h["pet"]["clip_max"] == PET_CLIP_MAX and "units not stated" in h["pet"]["note"]
        assert "computed" in h["pet"]["scale_source"]                    # the toy cache has no sidecar
        assert a["ct"].dtype == np.int16 and a["pet"].dtype == np.float16 and a["label"].dtype == np.uint8
        assert np.array_equal(a["ct"], ct[::f, ::f])
        assert np.array_equal(a["pet"], _normalise(ct[::f, ::f], pet[::f, ::f], scale)[1].astype(np.float16))
        assert set(np.unique(a["label"])) == {0, 1, 2}                   # both single-voxel labels survive
        assert a["label"][3 // f, 5 // f, 2] == 1 and a["label"][(nx - 1) // f, (ny - 1) // f, 1] == 2
        assert h["split"] == meta["split"] and h["is_test"] == (pid == "BBB-002")
        assert (h["test_note"] == "test set: view only, not for tuning") == (pid == "BBB-002")


def test_binary_layout_is_x_fastest_and_aligned(toy):
    s, arrs = toy
    body = _client(s).get("/api/cases/BBB-002/volume").content
    h, _ = cases.unpack_volume(body)
    ct = arrs["BBB-002"][0]
    assert body[:8] == cases.VOLUME_MAGIC and all(d["offset"] % 8 == 0 for d in h["arrays"].values())
    flat = np.frombuffer(body, dtype="<i2", count=ct.size, offset=h["arrays"]["ct"]["offset"])
    i, j, k = 7, 11, 3
    assert flat[i + h["nx"] * (j + h["ny"] * k)] == ct[i, j, k]           # the index the browser code uses


def test_cold_start_serves_the_page_on_loopback_only(toy, tmp_path):
    s, _ = toy
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "ACF_STUDIO_CACHE_DIR": str(s.cache_dir), "ACF_STUDIO_SPLIT": str(s.split_path),
           "ACF_STUDIO_META_CSV": str(s.meta_csv), "ACF_STUDIO_PORT": str(port)}
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-m", "app.studio"], cwd=ROOT, env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == f"http://127.0.0.1:{port}"
        page = None
        while page is None and time.perf_counter() - t0 < 120:
            try:
                page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read().decode("utf-8")
            except OSError:
                time.sleep(0.3)
        cold = time.perf_counter() - t0
        assert page is not None and "ACF" in page and WATERMARK in page
        ix = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/index", timeout=30).read())
        assert ix["n_cases"] == 2
        print(f"cold start to first page: {cold:.1f} s")
        psutil = pytest.importorskip("psutil")
        listening = [c.laddr for c in psutil.Process(proc.pid).net_connections(kind="inet") if c.status == psutil.CONN_LISTEN]
        assert listening and all(a.ip == "127.0.0.1" for a in listening), listening
    finally:
        proc.kill()
        proc.wait(timeout=30)


# ---------------------------------------------------------------- the real cache (training cases only)

def _real_settings() -> Settings:
    s = load_settings({})
    for cand in (s.meta_csv, Path("D:/misc/Hecktor/results/crop/crop_scan.csv")):   # results/ is not in a fresh worktree
        if Path(cand).exists():
            return Settings(cache_dir=REAL, split_path=s.split_path, meta_csv=Path(cand))
    return Settings(cache_dir=REAL, split_path=s.split_path, meta_csv=s.meta_csv)


def _split():
    return load_split(ROOT / "configs/splits/hecktor2026_seed42.json")


def _n_diff(a, b) -> int:
    """Number of differing voxels. Real arrays are compared through this count so that a failing assertion prints
    an integer, never voxel values."""
    a, b = np.asarray(a), np.asarray(b)
    return -1 if a.shape != b.shape else int(np.count_nonzero(a != b))


@real
def test_real_index_matches_the_split_and_the_data_report():
    s = _real_settings()
    ix = _client(s).get("/api/index").json()
    assert ix["n_cases"] == 782 and ix["split_counts"] == {"train": 592, "validation": 67, "test": 123}
    assert all(c["pet_scale"] is not None and c["pet_scale"] > 0 for c in ix["cases"])
    assert {c["centre"] for c in ix["cases"] if c["is_test"]} == {"CHUS", "CHUV"}
    if not Path(s.meta_csv).exists():
        pytest.skip(f"{s.meta_csv} absent: outlier count not checked")
    assert ix["n_pet_outliers"] == 52 and not ix["missing"]                 # docs/DATA_REPORT.md section 4
    block = {c["patient_id"] for c in ix["cases"] if c["pet_outlier"] and c["patient_id"].startswith("MDA-4")}
    assert block == {f"MDA-{n}" for n in range(402, 448)}


@real
def test_real_endpoints_on_three_training_cases():
    assert not set(THREE) & set(_split()["test"])
    c = _client(_real_settings())
    for pid in THREE:
        nat = crop_cache.read_header(REAL / f"{pid}.acf")
        _, arr = crop_cache.open_case(REAL / f"{pid}.acf")
        h, a = cases.unpack_volume(c.get(f"/api/cases/{pid}/volume").content)
        nx, ny, nz = nat["shape"]
        f = h["reduction"]
        assert f == math.ceil(max(nx, ny) / 256) and (h["nx"], h["ny"], h["nz"]) == (math.ceil(nx / f), math.ceil(ny / f), nz)
        assert np.allclose([h["sx"], h["sy"], h["sz"]], [nat["spacing"][0] * f, nat["spacing"][1] * f, nat["spacing"][2]])
        assert h["flip"] == [bool(nat["affine"][i][i] > 0) for i in range(3)] == [True, True, True]   # cache is RAS
        ext, nat_ext = [h["nx"] * h["sx"], h["ny"] * h["sy"], h["nz"] * h["sz"]], [n * sp for n, sp in zip(nat["shape"], nat["spacing"])]
        assert abs(ext[0] - nat_ext[0]) < h["sx"] and abs(ext[1] - nat_ext[1]) < h["sy"] and np.isclose(ext[2], nat_ext[2])
        assert abs(ext[2] / ext[0] - nat_ext[2] / nat_ext[0]) < 0.02 * nat_ext[2] / nat_ext[0]      # aspect ratio
        assert _n_diff(a["ct"], arr["ct"][::f, ::f]) == 0
        assert set(np.unique(a["label"]).tolist()) == set(np.unique(np.asarray(arr["label"])).tolist()) <= {0, 1, 2}
        assert float(a["pet"].min()) >= 0 and float(a["pet"].max()) <= PET_CLIP_MAX
        assert bool(np.isfinite(a["pet"].astype(np.float32)).all())
        assert h["split"] == "train" and h["is_test"] is False and h["pet"]["scale_source"] == "pet_scale.json"


def _largest_component(mask2d):
    from scipy import ndimage

    lab, n = ndimage.label(mask2d)
    if n == 0:
        return mask2d
    return lab == 1 + int(np.argmax(ndimage.sum(mask2d, lab, range(1, n + 1))))


@real
def test_real_orientation_on_three_training_cases():
    """Anatomy agrees with the flips on three cases. These are heuristics on the arrays (no image is rendered):
      head up     the PET-hot region (brain) lies above the body's centre along the axis drawn upward;
      nose front  in the band 20 to 90 mm below the brain's centre, the body is narrower 10 mm in from its
                  anterior end (nose, face) than 10 mm in from its posterior end (occiput);
      left/right  CT, PET and label come from the same voxel indices, and PET is higher inside the label than in
                  the body, which a mirrored or shifted label would not give.
    The nose heuristic is weak on some other cases (see the stage 1 report); the affine is the authority."""
    c = _client(_real_settings())
    for pid in THREE:
        h, a = cases.unpack_volume(c.get(f"/api/cases/{pid}/volume").content)
        ct, pet, lab = a["ct"], a["pet"].astype(np.float32), a["label"]
        _, arr = crop_cache.open_case(REAL / f"{pid}.acf")
        f = h["reduction"]
        up = (lambda v: v) if h["flip"][2] else (lambda v: v[:, :, ::-1])       # index grows toward what is drawn on top
        front = (lambda v: v) if h["flip"][1] else (lambda v: v[:, ::-1])       # index grows toward what is drawn in front
        ct_o, pet_o = front(up(ct)), front(up(pet))
        body = ct_o > -500
        hot = (pet_o > 0.5) & body
        z = np.arange(ct.shape[2])
        z_body = float((body.sum((0, 1)) * z).sum() / body.sum())
        z_hot = float((hot.sum((0, 1)) * z).sum() / hot.sum())
        assert z_hot > z_body + 0.2 * ct.shape[2], (pid, z_hot, z_body)
        d = round(10.0 / h["sy"])
        narrower = []
        for k in range(max(0, int(z_hot - 90 / h["sz"])), max(0, int(z_hot - 20 / h["sz"]))):
            big = _largest_component(body[:, :, k])
            ys = np.where(big.any(0))[0]
            if len(ys) >= 4 * d:
                narrower.append(big[:, ys[-1] - d].sum() < big[:, ys[0] + d].sum())
        assert len(narrower) >= 5 and np.mean(narrower) > 0.75, (pid, len(narrower), float(np.mean(narrower)))
        scale = h["pet"]["scale"]                                                # PET sampled at the CT's own indices
        same_grid = _normalise(ct, np.asarray(arr["pet"][::f, ::f]), scale)[1].astype(np.float16)
        assert _n_diff(a["pet"], same_grid) == 0, pid
        assert bool(lab.any()) and float(pet[lab > 0].mean()) > 1.5 * float(pet[ct > -500].mean()), pid


@real
def test_real_pet_scale_matches_training_normalisation():
    """One MDA-402..447 case (PET hundreds of times larger than the rest) and one ordinary case: the Studio's PET
    is the array the training loader gives the network, sampled at the display stride."""
    import yaml

    pids = ["MDA-410", "MDA-071"]
    assert not set(pids) & set(_split()["test"])
    cfg = yaml.safe_load((ROOT / "configs/shared_b4_acf_cond.yaml").read_text(encoding="utf-8"))
    assert cfg.get("data", {}).get("mode", "native") == "native"
    ds = AcfVolumeDataset([{"patient_id": p, "centre": "MDA"} for p in pids], REAL, cfg, load_stats(ROOT / cfg["conditioning"]["stats_path"]))
    sidecar = load_sidecar(REAL)
    c = _client(_real_settings())
    scales = {}
    for i, pid in enumerate(pids):
        h, a = cases.unpack_volume(c.get(f"/api/cases/{pid}/volume").content)
        f = h["reduction"]
        train_pet = ds[i]["image"][1].numpy()[::f, ::f]
        assert _n_diff(a["pet"], train_pet.astype(np.float16)) == 0, pid
        assert float(np.abs(a["pet"].astype(np.float32) - train_pet).max()) <= 20.0 * 2.0 ** -10, pid   # float16 rounding only
        assert h["pet"]["scale"] == sidecar[pid]["scale"] and h["pet"]["scale_source"] == "pet_scale.json"
        scales[pid] = h["pet"]["scale"]
    assert scales["MDA-410"] > 1000 and scales["MDA-071"] < 100            # the block case really is on another scale


@real
def test_real_largest_training_case_stays_far_below_two_gigabytes():
    s = _real_settings()
    train = set(_split()["train"])
    big = max((c for c in cases.build_index(s)["cases"] if c["patient_id"] in train), key=lambda c: int(np.prod(c["shape"])))
    tracemalloc.start()
    try:
        h, a = cases.display_volume(s, big["patient_id"])
        body = cases.pack_volume(h, a)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    print(f"largest training case {big['shape']}: body {len(body) / 2**20:.0f} MiB, peak traced {peak / 2**20:.0f} MiB")
    assert peak < 2 * 2**30
