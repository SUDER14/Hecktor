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
