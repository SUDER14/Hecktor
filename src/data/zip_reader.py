"""Read HECKTOR NIfTI cases straight out of the distribution zip.

Why: the 2026 training archive is ~144 GB and the disk it sits on has no room to
extract it. Members are read into memory one at a time (a case is tens of MB) and
parsed with nibabel's in-memory loaders, so nothing is ever written next to the
zip and no temp file is needed. The zip is only opened read-only.

A "zip path" is the string  ``<zip file>::<member name>``; `load_nifti` accepts
either that or an ordinary filesystem path, so callers need not care which.

macOS junk (``__MACOSX/``, ``.DS_Store``, ``._*`` resource forks) is skipped.
"""

from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import nibabel as nib

SEP = "::"


def is_junk_member(name: str) -> bool:
    parts = [p for p in name.split("/") if p]
    return (not parts) or parts[0] == "__MACOSX" or any(p.startswith(".") for p in parts)


def make_zip_path(zip_file: str | Path, member: str) -> str:
    return f"{zip_file}{SEP}{member}"


def split_zip_path(path: str) -> tuple[str, str] | None:
    """(zip file, member) for a zip path, else None for a plain filesystem path."""
    if SEP in str(path):
        z, m = str(path).split(SEP, 1)
        if z.lower().endswith(".zip"):
            return z, m.replace("\\", "/")  # zip members always use "/"; Path() on Windows may have flipped them
    return None


class ZipCohort:
    """Index of the non-junk members of one HECKTOR zip.

    A ZipFile handle is opened lazily per instance and re-opened in each process
    (handles are not picklable, so DataLoader workers must build their own).
    """

    def __init__(self, zip_file: str | Path):
        self.zip_file = str(zip_file)
        self._zf: zipfile.ZipFile | None = None
        self.members = [i.filename for i in self.zf.infolist()
                        if not i.is_dir() and not is_junk_member(i.filename)]

    @property
    def zf(self) -> zipfile.ZipFile:
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_file, "r")
        return self._zf

    def __getstate__(self):
        return {"zip_file": self.zip_file, "members": self.members, "_zf": None}

    def read_bytes(self, member: str) -> bytes:
        return self.zf.read(member)

    def read_text(self, member: str) -> str:
        return self.read_bytes(member).decode("utf-8-sig")

    def load(self, member: str) -> nib.Nifti1Image:
        """Full image (data decoded lazily by nibabel from the in-memory bytes)."""
        return _parse(self.read_bytes(member), member)

    def header(self, member: str) -> nib.Nifti1Header:
        """Header only: reads the first ~1 kB of the decompressed stream.

        Header scans of all 782 cases would otherwise decompress ~140 GB.
        """
        with self.zf.open(member) as raw:
            head = _read_head(raw, member)
        return nib.Nifti1Header.from_fileobj(_Bytes(head), check=False)


class _Bytes:
    """Minimal file object over bytes (nibabel only needs read())."""

    def __init__(self, b: bytes):
        self._b, self._p = b, 0

    def read(self, n: int = -1) -> bytes:
        out = self._b[self._p:] if n < 0 else self._b[self._p:self._p + n]
        self._p += len(out)
        return out

    def seek(self, p: int, whence: int = 0) -> int:
        self._p = p if whence == 0 else self._p + p if whence == 1 else len(self._b) + p
        return self._p

    def tell(self) -> int:
        return self._p


def _read_head(raw, member: str, n: int = 1024) -> bytes:
    if member.endswith(".gz"):
        with gzip.GzipFile(fileobj=raw) as g:
            return g.read(n)
    return raw.read(n)


def _parse(data: bytes, member: str) -> nib.Nifti1Image:
    if member.endswith(".gz"):
        data = gzip.decompress(data)
    return nib.Nifti1Image.from_bytes(data)


_COHORTS: dict[str, ZipCohort] = {}


def _cohort(zip_file: str) -> ZipCohort:
    if zip_file not in _COHORTS:
        _COHORTS[zip_file] = ZipCohort(zip_file)
    return _COHORTS[zip_file]


def load_nifti(path: str | Path) -> nib.Nifti1Image:
    """nib.load for a filesystem path, or a ``zip::member`` path from the zip."""
    zp = split_zip_path(str(path))
    if zp is None:
        return nib.load(str(path))
    return _cohort(zp[0]).load(zp[1])


def load_header(path: str | Path) -> nib.Nifti1Header:
    zp = split_zip_path(str(path))
    if zp is None:
        return nib.load(str(path)).header
    return _cohort(zp[0]).header(zp[1])
