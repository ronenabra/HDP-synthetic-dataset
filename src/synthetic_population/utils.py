from __future__ import annotations

import hashlib
import json
import os
import shutil
import re
from pathlib import Path
from typing import Any, Iterable


GIB = 1024 ** 3


def canonical_bytes(value: Any, *, pretty: bool = True) -> bytes:
    if pretty:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return text.encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value, pretty=False))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def available_bytes(path: Path) -> int:
    existing = path
    while not existing.exists():
        existing = existing.parent
    return shutil.disk_usage(existing).free


def projected_storage(spec: dict[str, Any], bytes_per_patient: int | None = None) -> dict[str, int | bool]:
    storage = spec["storage"]
    bpp = bytes_per_patient or storage["fallbackBytesPerPatient"]
    final = int(spec["populationSize"] * bpp)
    peak = int(final * storage["projectedFinalMultiplier"] + storage["fixedValidatorOverheadGiB"] * GIB)
    reserve = int(storage["minimumFreeReserveGiB"] * GIB)
    return {"bytesPerPatient": bpp, "projectedFinalBytes": final, "projectedPeakBytes": peak, "reserveBytes": reserve}


def assert_storage_safe(spec: dict[str, Any], out: Path, bytes_per_patient: int | None = None) -> dict[str, int | bool]:
    report = projected_storage(spec, bytes_per_patient)
    free = available_bytes(out)
    required = int(report["projectedPeakBytes"]) + int(report["reserveBytes"])
    report.update({"availableBytes": free, "requiredBytes": required, "safe": free >= required})
    if free < required:
        raise StorageError(report)
    return report


class StorageError(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        super().__init__("insufficient free space for configured peak and reserve")
        self.report = report


def validate_output_target(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.name.startswith("synthetic-population"):
        raise ValueError("output directory name must start with synthetic-population")
    if resolved.parent.name != "generated":
        raise ValueError("output directory must be an immediate child of generated/")
    if resolved.name.startswith("synthetic-population.v4"):
        match = re.fullmatch(r"synthetic-population\.v?4(?:\.[0-9A-Za-z-]+)+-([a-z0-9][a-z0-9-]*)", resolved.name)
        if not match:
            raise ValueError("v4 output requires a safe non-empty run classifier")
    return resolved


def atomic_target(path: Path, force: bool = False) -> tuple[Path, Path]:
    target = validate_output_target(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.partial-{os.getpid()}")
    if temp.exists():
        shutil.rmtree(temp)
    if target.exists() and any(target.iterdir()) and target.name.startswith("synthetic-population.v4"):
        raise FileExistsError(f"v4 classified run history is immutable: {target}")
    if target.exists() and any(target.iterdir()) and not force:
        raise FileExistsError(f"refusing to overwrite non-empty target: {target}")
    return target, temp


def publish_atomic(target: Path, temp: Path, force: bool) -> None:
    if target.exists():
        if target.name.startswith("synthetic-population.v4"):
            raise FileExistsError(f"v4 classified run history is immutable: {target}")
        if not force:
            raise FileExistsError(target)
        shutil.rmtree(target)
    os.replace(temp, target)


def checksum_lines(root: Path, excluded: Iterable[str] = ("checksums.sha256",)) -> list[str]:
    skipped = set(excluded)
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in skipped or rel.endswith("/storage-pause.json"):
            continue
        lines.append(f"{sha256_file(path)}  {rel}")
    return lines
