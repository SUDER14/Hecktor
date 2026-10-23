"""HECKTOR 2026 adapter.

Layout (documented by the user; the real files have not been seen yet, so this
stays UNVERIFIED against real data -- tests use src/data/synthetic.py):

    hecktor2026_training/<CENTRE-NNN>/<CENTRE-NNN>__CT.nii.gz
                                      <CENTRE-NNN>__PT.nii.gz
                                      <CENTRE-NNN>.nii.gz          (GTV label)
    hecktor2026_training/HECKTOR_2026_Training_Task.csv

The centre is the prefix before the first dash. A patient missing CT, PT or
label is an error (silently dropping cases would change the cohort unnoticed).
The Task CSV is read only to attach its columns as `meta`; its schema is not
assumed beyond having a header, and a missing CSV is allowed.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.data.adapters import is_junk_file, resolve_nifti


def centre_of(patient_id: str) -> str:
    return patient_id.split("-")[0].upper() if "-" in patient_id else "UNKNOWN"


class Hecktor2026Adapter:
    def __init__(self, data_dir: str | Path, csv_name: str = "HECKTOR_2026_Training_Task.csv"):
        self.root = Path(data_dir)
        self.csv_path = self.root / csv_name

    def _read_csv(self) -> dict[str, dict]:
        if not self.csv_path.exists():
            return {}
        with open(self.csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        id_col = next((c for c in rows[0] if c.lower().replace("_", "") in ("patientid", "id", "subject")),
                      next(iter(rows[0])))
        return {r[id_col]: r for r in rows}

    def training_samples(self) -> list[dict]:
        meta = self._read_csv()
        samples = []
        for pdir in sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")):
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
