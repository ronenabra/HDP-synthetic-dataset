from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .utils import read_json


REPOSITORY_ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_catalog(relative_path: str) -> dict[str, Any]:
    return read_json(input_path(relative_path))


def input_path(relative_path: str) -> Path:
    """Preserve model path identifiers while resolving packaged construction inputs."""
    prefix = "tools/synthetic_population/data/"
    if relative_path.startswith(prefix):
        relative = Path(relative_path.removeprefix(prefix))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe construction catalog path")
        return REPOSITORY_ROOT / "data" / relative
    if relative_path == "bundle/certificator-bundle.v1.6.1.transaction.json":
        return REPOSITORY_ROOT / "data/coverage-seed.original.json"
    if relative_path == "FHIR-Profiles/ilcore/CodeSystem-city-symbol.json":
        return REPOSITORY_ROOT.parent / "reference/city-symbol-0.21.0.json"
    raise ValueError("Unsupported construction input: " + relative_path)
