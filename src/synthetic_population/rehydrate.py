"""Restore pinned public FHIR inputs and the derived synthetic coverage seed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

RESOURCE_TYPES = ("CodeSystem", "StructureDefinition")


def _resources(directory: Path) -> dict[str, bytes]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Reference directory is missing: {directory}")
    resources = {}
    for path in sorted(directory.iterdir()):
        if path.suffix != ".json" or not path.name.startswith(tuple(kind + "-" for kind in RESOURCE_TYPES)):
            continue
        raw = path.read_bytes()
        if json.loads(raw).get("resourceType") not in RESOURCE_TYPES:
            raise ValueError(f"Unexpected resource type in {path}")
        resources[path.name] = raw
    return resources


def content_digest(resources: dict[str, bytes]) -> str:
    index = "".join(f"{name}\0{hashlib.sha256(raw).hexdigest()}\n" for name, raw in sorted(resources.items()))
    return hashlib.sha256(index.encode("utf-8")).hexdigest()


def _safe_reference_path(package_root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != "reference" or ".." in path.parts:
        raise ValueError(f"Invalid derived reference path: {relative}")
    destination = package_root / path
    if destination.is_symlink() or (package_root / "reference").is_symlink():
        raise ValueError("Derived reference paths must not be symlinks")
    return destination


def _context(package_root: Path) -> dict:
    context = json.loads((package_root / "data/reference-context.json").read_text(encoding="utf-8"))
    paths = set()
    for row in context["profiles"]:
        _safe_reference_path(package_root, row["path"])
        if row["path"] in paths:
            raise ValueError("Duplicate reference destination")
        paths.add(row["path"])
        if context["packages"].get(row["package"]) != row["version"]:
            raise ValueError("Inconsistent package version pin")
    _safe_reference_path(package_root, context["coverageSeed"]["path"])
    source = context.get("nativeGenerator")
    if source is not None:
        _safe_reference_path(package_root, source["citySource"]["path"])
    return context


def _check_resources(row: dict, resources: dict[str, bytes]) -> None:
    if len(resources) != row["fileCount"] or content_digest(resources) != row["contentSha256"]:
        raise ValueError(f"Reference content does not match the pin for {row['package']}#{row['version']}")


def _check_bytes(pin: dict, raw: bytes, label: str) -> None:
    if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError(label + " does not match its content pin")


def _check_seed(pin: dict, raw: bytes) -> None:
    _check_bytes(pin, raw, "Coverage seed")
    resource = json.loads(raw)
    if resource.get("resourceType") != "Bundle" or resource.get("type") != "transaction":
        raise ValueError("Coverage seed is not a transaction Bundle")


def _bundled_native_generator(package_root: Path, source: dict) -> None:
    if source["root"] != "native_generator" or source["modulePath"] != "":
        raise ValueError("Invalid native native generator package location")
    base = package_root / "native_generator"
    if base.is_symlink():
        raise ValueError("Native generator package must not be a symlink")
    seen = set()
    for row in source["files"]:
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts or row["path"] in seen:
            raise ValueError("Invalid or duplicate native generator input path")
        seen.add(row["path"])
        destination = base / path
        if not destination.resolve().is_relative_to(base.resolve()) or not destination.is_file():
            raise ValueError("Missing bundled native generator input: " + row["path"])
        _check_bytes(row, destination.read_bytes(), "Bundled native generator input " + row["path"])


def require_native_generator(package_root: Path) -> dict:
    source = _context(package_root).get("nativeGenerator")
    if source is None:
        raise ValueError("Pinned native native generator inputs are not configured")
    _bundled_native_generator(package_root, source)
    city = source["citySource"]
    try:
        _check_bytes(city, _safe_reference_path(package_root, city["path"]).read_bytes(), "Native generator city CodeSystem")
    except OSError as exc:
        raise ValueError("Missing construction city CodeSystem; run hdp-synthetic-rehydrate --cache PATH") from exc
    return source


def require_references(package_root: Path) -> dict:
    context = _context(package_root)
    try:
        for row in context["profiles"]:
            _check_resources(row, _resources(package_root / row["path"]))
        seed = context["coverageSeed"]
        _check_seed(seed, (package_root / seed["path"]).read_bytes())
        if context.get("nativeGenerator"):
            require_native_generator(package_root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{exc}. Restore inputs with hdp-synthetic-rehydrate --cache PATH; use --replace for stale derived inputs.") from exc
    return {"status": "ready", "packages": context["packages"], "profileFileCount": sum(row["fileCount"] for row in context["profiles"]), "coverageSeedSha256": seed["sha256"]}


def _cache_package(cache: Path, pin: dict, fhir_version: str) -> Path:
    identity = f"{pin['package']}#{pin['version']}"
    directory = cache / identity / "package"
    if not directory.is_dir():
        raise FileNotFoundError(f"Pinned package {identity} is missing from {cache}; populate the public FHIR package cache first")
    meta = json.loads((directory / "package.json").read_text(encoding="utf-8"))
    if meta.get("name") != pin["package"] or meta.get("version") != pin["version"] or fhir_version not in meta.get("fhirVersions", []):
        raise ValueError("Package metadata does not match " + identity)
    return directory


def _coverage_seed(package_root: Path, pin: dict) -> bytes:
    """Migrate the bundled seed's actor identifiers; never construct a population."""
    from .business_identifiers import set_business_identifier
    if pin.get("migration") != "synthetic-business-identifiers-v1":
        raise ValueError("Unsupported synthetic seed migration")
    source = pin["source"]
    path = package_root / source["path"]
    if not path.resolve().is_relative_to((package_root / "native_generator").resolve()):
        raise ValueError("Coverage seed source escapes the native generator package")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("Bundled coverage seed source differs from its pin")
    value = json.loads(raw)
    identity_spec = {"seed": "s24-certificator", "datasetVersion": "1.6.1"}
    system = "http://fhir.health.gov.il/identifier/il-hdp-test-id"
    for entry in value["entry"]:
        resource = entry["resource"]
        if resource["resourceType"] in {"Patient", "Practitioner"}:
            test = next((i.get("value") for i in resource.get("identifier", []) if i.get("system") == system), None)
            anchor = test.removeprefix("certificator-") if test else None
            set_business_identifier(resource, identity_spec, "certificator", anchor)

    def references(item):
        if isinstance(item, dict):
            if item.get("system") == system and item.get("value") in {"certificator-1", "certificator-2"}:
                item["value"] = item["value"].removeprefix("certificator-")
            for child in item.values():
                references(child)
        elif isinstance(item, list):
            for child in item:
                references(child)

    references(value)
    # The pinned seed preserves the source fixture key order.
    result = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _check_seed(pin, result)
    return result


def rehydrate(package_root: Path, cache: Path, *, replace: bool = False) -> dict:
    context = _context(package_root)
    source = context.get("nativeGenerator")
    if source is not None:
        _bundled_native_generator(package_root, source)
    planned, reused = [], []
    for row in context["profiles"]:
        destination = _safe_reference_path(package_root, row["path"])
        if destination.exists():
            try:
                _check_resources(row, _resources(destination))
            except (OSError, ValueError):
                if not replace:
                    raise ValueError(f"Stale reference folder: {destination}; use --replace")
            else:
                reused.append(row["package"])
                continue
        directory = _cache_package(cache, row, context["fhirVersion"])
        meta = json.loads((directory / "package.json").read_text(encoding="utf-8"))
        for dependency, version in meta.get("dependencies", {}).items():
            if dependency in context["packages"] and version != context["packages"][dependency]:
                raise ValueError("Package dependency differs from the pin")
        resources = _resources(directory)
        _check_resources(row, resources)
        planned.append((row["package"], destination, resources))
    if source is not None:
        city = source["citySource"]
        destination = _safe_reference_path(package_root, city["path"])
        ready = destination.is_file() and destination.stat().st_size == city["bytes"] and hashlib.sha256(destination.read_bytes()).hexdigest() == city["sha256"]
        if ready:
            reused.append("construction-city-codes")
        else:
            if destination.exists() and not replace:
                raise ValueError("Stale construction city CodeSystem; use --replace")
            directory = _cache_package(cache, city, context["fhirVersion"])
            raw = (directory / city["filename"]).read_bytes()
            _check_bytes(city, raw, "Native generator city CodeSystem")
            planned.append(("construction-city-codes", destination, raw))
    seed = context["coverageSeed"]
    destination = _safe_reference_path(package_root, seed["path"])
    if destination.exists():
        try:
            _check_seed(seed, destination.read_bytes())
        except (OSError, ValueError):
            if not replace:
                raise ValueError("Stale coverage seed; use --replace")
        else:
            reused.append("coverage-seed")
    if "coverage-seed" not in reused:
        planned.append(("coverage-seed", destination, _coverage_seed(package_root, seed)))
    # All sources and exact contents are checked before any destination changes.
    if planned:
        parent = package_root / "reference"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".rehydrate-", dir=parent) as temporary:
            work = Path(temporary)
            for number, (_, _, content) in enumerate(planned):
                staged = work / f"new-{number}"
                if isinstance(content, dict):
                    staged.mkdir()
                    for name, raw in content.items():
                        (staged / name).write_bytes(raw)
                else:
                    staged.write_bytes(content)
            applied = []
            try:
                for number, (_, destination, _) in enumerate(planned):
                    backup = work / f"old-{number}"
                    if destination.exists():
                        os.replace(destination, backup)
                    applied.append((destination, backup))
                    os.replace(work / f"new-{number}", destination)
                ready = require_references(package_root)
            except BaseException:
                for destination, backup in reversed(applied):
                    if destination.is_dir():
                        shutil.rmtree(destination)
                    elif destination.exists():
                        destination.unlink()
                    if backup.exists():
                        os.replace(backup, destination)
                raise
    else:
        ready = require_references(package_root)
    return {**ready, "restoredInputs": [identity for identity, _, _ in planned], "reusedInputs": reused}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".fhir/packages", help="Cache containing <package>#<version>/package directories")
    parser.add_argument("--replace", action="store_true", help="Replace stale derived inputs after checking all required source contents")
    parser.add_argument("--check", action="store_true", help="Only check materialized inputs; do not access the cache")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    try:
        result = require_references(root) if args.check else rehydrate(root, args.cache.expanduser().resolve(), replace=args.replace)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
