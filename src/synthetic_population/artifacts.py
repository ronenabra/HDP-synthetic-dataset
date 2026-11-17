from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_json


SOURCES = ("hmo", "hospital-a", "hospital-b")


def is_v4_spec(spec: dict[str, Any]) -> bool:
    return bool(spec.get("remediation")) and spec.get("datasetVersion", "").startswith("4.")


def bundle_relative_path(spec: dict[str, Any], source: str) -> Path:
    if source not in SOURCES:
        raise ValueError(f"unknown source: {source}")
    return Path("fhir-server-load") / f"{source}.source.transaction.json" if is_v4_spec(spec) else Path(f"{source}.source.transaction.json")


def source_bundle_path(out: Path, source: str) -> Path:
    manifest_path = out / "dataset-manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        rows = manifest.get("loadableSourceBundles", [])
        row = next((item for item in rows if item.get("source") == source), None)
        if row:
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe source Bundle path in manifest: {relative}")
            resolved = (out / relative).resolve()
            if not resolved.is_relative_to(out.resolve()):
                raise ValueError(f"source Bundle path escapes output: {relative}")
            return resolved
    modern = out / "fhir-server-load" / f"{source}.source.transaction.json"
    legacy = out / f"{source}.source.transaction.json"
    return modern if modern.is_file() else legacy


def load_source_bundles(out: Path) -> dict[str, dict[str, Any]]:
    return {source: read_json(source_bundle_path(out, source)) for source in SOURCES}
