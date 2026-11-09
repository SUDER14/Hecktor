"""HECKTOR 2026 adapter (directory tree or the distribution zip itself).

Layout, VERIFIED against the real archive (782 cases):

    HECKTOR 2026 Training Data/<CENTRE-NNN>/<CENTRE-NNN>__CT.nii.gz
                                            <CENTRE-NNN>__PT.nii.gz
                                            <CENTRE-NNN>.nii.gz          (label)
    HECKTOR 2026 Training Data/HECKTOR_2026_training_data.csv

`data_dir` may be an extracted directory or the .zip; for a zip the sample paths
are ``<zip>::<member>`` strings that `src.data.zip_reader.load_nifti` reads in
memory (nothing is extracted). Junk entries (__MACOSX, ._*, .DS_Store) are skipped.

The centre is the prefix before the first dash. A patient missing CT, PT or
label is an error (silently dropping cases would change the cohort unnoticed).
The CSV is read only to attach its columns as `meta`.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from src.data.adapters import is_junk_file, resolve_nifti
from src.data.zip_reader import ZipCohort, make_zip_path


def centre_of(patient_id: str) -> str:
    return patient_id.split("-")[0].upper() if "-" in patient_id else "UNKNOWN"


CSV_NAMES = ("HECKTOR_2026_training_data.csv", "HECKTOR_2026_Training_Task.csv")


def _parse_csv(text: str) -> dict[str, dict]:
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return {}
    id_col = next((c for c in rows[0] if c.lower().replace("_", "") in ("patientid", "id", "subject")),
                  next(iter(rows[0])))
    return {r[id_col]: r for r in rows}


class Hecktor2026Adapter:
    def __init__(self, data_dir: str | Path, csv_name: str | None = None):
        self.root = Path(data_dir)
        self.is_zip = self.root.suffix.lower() == ".zip"
        self.csv_names = (csv_name,) if csv_name else CSV_NAMES
        self.csv_path = self.root / self.csv_names[0]  # kept for callers that print it

    def _read_csv(self) -> dict[str, dict]:
        if self.is_zip:
            return self._read_csv_zip(ZipCohort(self.root))
        for name in self.csv_names:
            p = self.root / name
            if p.exists():
                self.csv_path = p
                return _parse_csv(p.read_text(encoding="utf-8-sig"))
        return {}

    def _read_csv_zip(self, zc: ZipCohort) -> dict[str, dict]:
        for m in zc.members:
            parts = m.split("/")
            if len(parts) <= 2 and parts[-1] in self.csv_names:
                return _parse_csv(zc.read_text(m))
        return {}

    def training_samples(self) -> list[dict]:
        return self._zip_samples() if self.is_zip else self._dir_samples()

    def _zip_samples(self) -> list[dict]:
        zc = ZipCohort(self.root)
        meta = self._read_csv_zip(zc)
        by_pid: dict[str, dict[str, str]] = {}
        for m in zc.members:
            parts = m.split("/")
            if len(parts) < 2 or not parts[-1].endswith(".nii.gz"):
                continue
            pid, fname = parts[-2], parts[-1]
            kind = "CT" if fname.endswith("__CT.nii.gz") else "PT" if fname.endswith("__PT.nii.gz")                 else "label" if fname == f"{pid}.nii.gz" else None
            if kind:
                by_pid.setdefault(pid, {})[kind] = m
        samples = []
        for pid in sorted(by_pid):
            d = by_pid[pid]
            missing = [k for k in ("CT", "PT", "label") if k not in d]
            if missing:
                raise FileNotFoundError(f"{pid}: missing {missing} in {self.root}")
            samples.append({"patient_id": pid, "centre": centre_of(pid), "modalities": ["CT", "PT"],
                            "image": [make_zip_path(self.root, d["CT"]), make_zip_path(self.root, d["PT"])],
                            "label": make_zip_path(self.root, d["label"]), "meta": meta.get(pid, {})})
        if not samples:
            raise FileNotFoundError(f"no <CENTRE-NNN>/ patient folders in zip {self.root}")
        return samples

    def _dir_samples(self) -> list[dict]:
        meta = self._read_csv()
        samples = []
        for pdir in sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")
                           and p.name != "__MACOSX"):
            pid = pdir.name
            ct = resolve_nifti(pdir / f"{pid}__CT.nii.gz")
            pt = resolve_nifti(pdir / f"{pid}__PT.nii.gz")
            lab = resolve_nifti(pdir / f"{pid}.nii.gz")
            missing = [p.name for p in (ct, pt, lab) if not p.exists() or is_junk_file(p)]
            if missing:
                raise FileNotFoundError(f"{pid}: missing {missing} in {pdir}")
            samples.append({"patient_id": pid, "centre": centre_of(pid), "modalities": ["CT", "PT"],
                            "image": [str(ct), str(pt)], "label": str(lab), "meta": meta.get(pid, {})})
        if not samples:
            raise FileNotFoundError(f"no <CENTRE-NNN>/ patient folders under {self.root}")
        return samples
