"""Dataset-specific adapters.

Each adapter turns a dataset's own on-disk layout into a common sample-dict
schema so the rest of the pipeline (transforms, splits, conditioning) never
has to know which dataset it is looking at:

    {
        "patient_id": str,
        "centre": str,            # "UNKNOWN" where the dataset does not encode one
        "modalities": [str, ...], # e.g. ["CT"] or ["CT", "PT"], parallel to "image"
        "image": [str, ...],      # one path per modality, same order as "modalities"
        "label": str,
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

CENTRE_RE = re.compile(r"^([A-Za-z]{2,6})[-_]\d+")


@runtime_checkable
class SegmentationDataset(Protocol):
    """Common interface every dataset adapter satisfies.

    Structural (a Protocol), not a base class to inherit from: DecathlonAdapter
    and HecktorAdapter already match this shape, so nothing downstream
    (splits.py, conditioning.py, dataset.py) needs to know which adapter it
    was handed -- it only relies on training_samples() returning the sample
    schema documented at the top of this module.
    """

    def training_samples(self) -> list[dict]: ...


def _strip_nii_suffix(name: str) -> str:
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


class DecathlonAdapter:
    """Medical Segmentation Decathlon layout: imagesTr/, labelsTr/, dataset.json.

    Every MSD task (including Task06_Lung) is single-modality -- there is no
    PET task in the Decathlon at all -- so this always yields exactly one
    entry in "modalities"/"image" per sample. Modality name is read from
    dataset.json rather than guessed from the filename, since Decathlon
    filenames carry no modality tag.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        with open(self.data_dir / "dataset.json") as f:
            self.meta = json.load(f)

        modality_map = self.meta["modality"]
        if len(modality_map) != 1:
            raise NotImplementedError(
                f"{self.data_dir}: multi-channel Decathlon modality map "
                f"{modality_map} is not handled -- every current MSD task is "
                "single-modality, so this adapter has never had a reason to "
                "support more than one."
            )
        self.modality = next(iter(modality_map.values())).upper()

    def _resolve(self, rel_path: str) -> Path:
        return self.data_dir / rel_path.lstrip("./")

    def training_samples(self) -> list[dict]:
        samples = []
        for entry in self.meta["training"]:
            image_path = self._resolve(entry["image"])
            label_path = self._resolve(entry["label"])
            samples.append(
                {
                    "patient_id": _strip_nii_suffix(image_path.name),
                    "centre": "UNKNOWN",  # Decathlon filenames carry no centre code
                    "modalities": [self.modality],
                    "image": [str(image_path)],
                    "label": str(label_path),
                }
            )
        return samples


class HecktorAdapter:
    """Expected HECKTOR layout, per the challenge's documented convention and
    the file-naming scheme HyperSpace (arXiv:2407.03681) reports:

        imagesTr/<patient_id>__CT.nii.gz
        imagesTr/<patient_id>__PT.nii.gz
        labelsTr/<patient_id>.nii.gz

    patient_id is "<CENTRE>-<number>", e.g. "CHUM-001"; centre is read off
    that prefix.

    UNVERIFIED. HECKTOR access has not been granted yet (see CLAUDE.md), so
    this class has never been run against a real HECKTOR file. It is written
    against the documented naming convention only -- re-validate the moment
    access is granted, the same way spacing_analysis.py's header assumptions
    had to be re-validated against real Decathlon NIfTI files before trusting
    them.
    """

    MODALITY_RE = re.compile(r"__(CT|PT|PET)\b", re.IGNORECASE)

    def __init__(self, data_dir: str | Path, images_subdir: str = "imagesTr",
                 labels_subdir: str = "labelsTr"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / images_subdir
        self.labels_dir = self.data_dir / labels_subdir

    def _infer_centre(self, patient_id: str) -> str:
        m = CENTRE_RE.match(patient_id)
        return m.group(1).upper() if m else "UNKNOWN"

    def training_samples(self) -> list[dict]:
        by_patient: dict[str, dict[str, Path]] = {}
        for path in sorted(self.images_dir.glob("*.nii.gz")):
            if path.name.startswith("."):
                continue
            m = self.MODALITY_RE.search(path.name)
            if not m:
                raise ValueError(
                    f"{path.name}: expected a __CT or __PT modality tag in the "
                    "filename per the documented HECKTOR convention; got none. "
                    "The real file-naming scheme has not been verified yet."
                )
            tag = m.group(1).upper()
            modality = "PT" if tag in ("PT", "PET") else tag
            patient_id = path.name.split("__")[0]
            by_patient.setdefault(patient_id, {})[modality] = path

        samples = []
        for patient_id, modality_paths in sorted(by_patient.items()):
            label_path = self.labels_dir / f"{patient_id}.nii.gz"
            if not label_path.exists():
                raise FileNotFoundError(
                    f"No label found for {patient_id} at {label_path}"
                )
            modalities = sorted(modality_paths)  # deterministic CT-before-PT order
            samples.append(
                {
                    "patient_id": patient_id,
                    "centre": self._infer_centre(patient_id),
                    "modalities": modalities,
                    "image": [str(modality_paths[m]) for m in modalities],
                    "label": str(label_path),
                }
            )
        return samples
