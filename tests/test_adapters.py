"""Unit tests for src/data/adapters.py.

DecathlonAdapter is tested against the real Task06_Lung data when it's
present on disk (skipped otherwise, e.g. on a fresh checkout/Colab without
the dataset downloaded yet) -- there is no reason to fake Decathlon's layout
when the real thing is sitting right there.

HecktorAdapter has to be tested against a synthetic directory tree: HECKTOR
access has not been granted (see CLAUDE.md), so there is no real data to
point it at. These tests check the adapter's own logic against the
documented naming convention -- they cannot and do not confirm that
convention matches the real dataset.
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.data.adapters import DecathlonAdapter, HecktorAdapter

REAL_DECATHLON_DIR = Path("data/Task06_Lung")


def _touch_nifti(path: Path, shape=(4, 4, 4)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))
    nib.save(img, str(path))


@pytest.mark.skipif(not REAL_DECATHLON_DIR.exists(), reason="Task06_Lung not downloaded")
class TestDecathlonAdapterOnRealData:
    def test_sample_count_matches_dataset_json(self):
        adapter = DecathlonAdapter(REAL_DECATHLON_DIR)
        with open(REAL_DECATHLON_DIR / "dataset.json") as f:
            meta = json.load(f)
        samples = adapter.training_samples()
        assert len(samples) == meta["numTraining"]

    def test_modality_is_ct_from_dataset_json_not_guessed(self):
        adapter = DecathlonAdapter(REAL_DECATHLON_DIR)
        samples = adapter.training_samples()
        assert all(s["modalities"] == ["CT"] for s in samples)

    def test_paths_exist_and_patient_id_matches_filename(self):
        adapter = DecathlonAdapter(REAL_DECATHLON_DIR)
        for s in adapter.training_samples()[:5]:
            assert Path(s["image"][0]).exists()
            assert Path(s["label"]).exists()
            assert s["patient_id"] == Path(s["image"][0]).name.replace(".nii.gz", "")

    def test_decathlon_has_no_centre_metadata(self):
        adapter = DecathlonAdapter(REAL_DECATHLON_DIR)
        samples = adapter.training_samples()
        assert all(s["centre"] == "UNKNOWN" for s in samples)


def test_decathlon_adapter_rejects_multi_channel_modality(tmp_path):
    (tmp_path / "imagesTr").mkdir()
    (tmp_path / "labelsTr").mkdir()
    dataset_json = {
        "modality": {"0": "CT", "1": "PT"},
        "training": [],
    }
    with open(tmp_path / "dataset.json", "w") as f:
        json.dump(dataset_json, f)

    with pytest.raises(NotImplementedError):
        DecathlonAdapter(tmp_path)


class TestHecktorAdapterSynthetic:
    def _make_patient(self, tmp_path, patient_id, modalities=("CT", "PT"), with_label=True):
        for m in modalities:
            _touch_nifti(tmp_path / "imagesTr" / f"{patient_id}__{m}.nii.gz")
        if with_label:
            _touch_nifti(tmp_path / "labelsTr" / f"{patient_id}.nii.gz")

    def test_pairs_ct_and_pt_per_patient(self, tmp_path):
        self._make_patient(tmp_path, "CHUM-001")
        self._make_patient(tmp_path, "CHUS-002")

        adapter = HecktorAdapter(tmp_path)
        samples = {s["patient_id"]: s for s in adapter.training_samples()}

        assert set(samples) == {"CHUM-001", "CHUS-002"}
        assert samples["CHUM-001"]["modalities"] == ["CT", "PT"]
        assert len(samples["CHUM-001"]["image"]) == 2

    def test_infers_centre_from_patient_id_prefix(self, tmp_path):
        self._make_patient(tmp_path, "CHUM-001")
        adapter = HecktorAdapter(tmp_path)
        samples = adapter.training_samples()
        assert samples[0]["centre"] == "CHUM"

    def test_raises_on_missing_label(self, tmp_path):
        self._make_patient(tmp_path, "CHUM-001", with_label=False)
        adapter = HecktorAdapter(tmp_path)
        with pytest.raises(FileNotFoundError):
            adapter.training_samples()

    def test_raises_on_untagged_filename(self, tmp_path):
        _touch_nifti(tmp_path / "imagesTr" / "CHUM-001.nii.gz")  # no __CT/__PT tag
        adapter = HecktorAdapter(tmp_path)
        with pytest.raises(ValueError):
            adapter.training_samples()


# ---------------------------------------------------------------------------
# HECKTOR 2026 straight from the distribution zip (synthetic zip, same layout and junk entries)
# ---------------------------------------------------------------------------
import zipfile  # noqa: E402

from src.data.hecktor2026 import Hecktor2026Adapter  # noqa: E402
from src.data.synthetic import SynthConfig, write_synthetic_hecktor2026  # noqa: E402
from src.data.zip_reader import (  # noqa: E402
    ZipCohort, is_junk_member, load_header, load_nifti, make_zip_path, split_zip_path,
)

TOP = "HECKTOR 2026 Training Data"


@pytest.fixture(scope="module")
def zipped(tmp_path_factory):
    """Synthetic cohort zipped like the real archive: top folder, macOS junk, CSV at the top."""
    base = tmp_path_factory.mktemp("zsrc")
    ids = write_synthetic_hecktor2026(base / "d", SynthConfig(fov_mm=(24, 24, 24), n_per_centre=1, seed=2))
    zpath = tmp_path_factory.mktemp("zout") / "HECKTOR 2026 Training Data.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{TOP}/", "")
        for f in sorted((base / "d").rglob("*")):
            if f.is_file():
                z.write(f, f"{TOP}/{f.relative_to(base / 'd').as_posix()}")
        z.writestr("__MACOSX/._" + TOP, b"junk")
        z.writestr(f"__MACOSX/{TOP}/CHUM-001/._CHUM-001__CT.nii.gz", b"junk")
        z.writestr(f"{TOP}/.DS_Store", b"junk")
        z.writestr(f"{TOP}/CHUM-001/._CHUM-001__PT.nii.gz", b"junk")
    return zpath, base / "d", ids


def test_junk_members_are_recognised():
    assert is_junk_member("__MACOSX/x") and is_junk_member(f"{TOP}/.DS_Store")
    assert is_junk_member(f"{TOP}/CHUM-001/._CHUM-001__CT.nii.gz")
    assert not is_junk_member(f"{TOP}/CHUM-001/CHUM-001__CT.nii.gz")


def test_zip_adapter_matches_directory_adapter_and_skips_junk(zipped):
    zpath, d, ids = zipped
    zs = Hecktor2026Adapter(zpath).training_samples()
    ds = Hecktor2026Adapter(d).training_samples()
    assert [s["patient_id"] for s in zs] == [s["patient_id"] for s in ds] == sorted(ids)
    assert all(s["centre"] == t["centre"] and s["modalities"] == ["CT", "PT"] for s, t in zip(zs, ds))
    assert all(split_zip_path(p) for s in zs for p in s["image"] + [s["label"]])  # every path is a zip path
    assert zs[0]["meta"]["Centre"] == zs[0]["centre"]                            # CSV read from inside the zip


def test_zip_read_equals_disk_read_for_header_and_voxels(zipped):
    zpath, d, _ = zipped
    for sample in Hecktor2026Adapter(zpath).training_samples()[:3]:
        pid = sample["patient_id"]
        for z_path, name in zip(sample["image"] + [sample["label"]],
                                (f"{pid}__CT.nii.gz", f"{pid}__PT.nii.gz", f"{pid}.nii.gz")):
            disk = nib.load(d / pid / name)
            zimg, zhdr = load_nifti(z_path), load_header(z_path)
            assert zimg.shape == disk.shape and np.allclose(zimg.affine, disk.affine)
            assert np.array_equal(np.asarray(zimg.dataobj), np.asarray(disk.dataobj))
            assert zhdr.get_data_shape() == disk.shape and zhdr.get_zooms()[:3] == disk.header.get_zooms()[:3]


def test_reader_never_writes_next_to_the_zip(zipped):
    zpath, _, _ = zipped
    before = sorted(p.name for p in zpath.parent.iterdir())
    for s in Hecktor2026Adapter(zpath).training_samples():
        load_nifti(s["image"][0])
    assert sorted(p.name for p in zpath.parent.iterdir()) == before


def test_zip_missing_modality_is_an_error(zipped, tmp_path):
    zpath, _, _ = zipped
    broken = tmp_path / "b.zip"
    with zipfile.ZipFile(zpath) as src, zipfile.ZipFile(broken, "w") as dst:
        for i in src.infolist():
            if not i.filename.endswith("CHUM-001__PT.nii.gz"):
                dst.writestr(i, src.read(i))
    with pytest.raises(FileNotFoundError, match="CHUM-001"):
        Hecktor2026Adapter(broken).training_samples()


def test_zip_cohort_index_lists_no_junk(zipped):
    zpath, _, _ = zipped
    members = ZipCohort(zpath).members
    assert members and not any(is_junk_member(m) for m in members)
    assert make_zip_path(zpath, members[0]).endswith(members[0])
