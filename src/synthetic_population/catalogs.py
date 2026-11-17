from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .utils import read_json


REPOSITORY_ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_catalog(relative_path: str) -> dict[str, Any]:
    return read_json(REPOSITORY_ROOT / relative_path)


def default_catalog(section: str, field: str = "catalog") -> str:
    """Return a catalog selected by the shipped current native specification."""
    return load_catalog("data/specs/native.json")[section][field]


def default_detector_catalog() -> str:
    return load_catalog(default_catalog("clinicalRealism"))["detectorSemanticsCatalog"]
