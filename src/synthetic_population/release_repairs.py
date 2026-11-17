"""Reconstruct the accepted six source bundles from pinned native code and repairs.

Acceptance is checked by exact published bundle hashes. Audit receipts and
machine-specific reports are not part of the reproduction output.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from .business_identifiers import set_business_identifier
from .laboratory import finalize_laboratory, is_laboratory
from .rehydrate import require_native_generator
from .utils import (GIB, StorageError, available_bytes, checksum_lines, read_json,
                    sha256_file, validate_output_target, write_json)

PLAN_SHA256 = "755a9e73326fc4aace6985b3e76e34157c0368710f7779dbdd4eb8d2571938ce"
RELEASE_ID = "4.2.2-2026-10-10"
SOURCES = ("hmo", "hospital-a", "hospital-b")
LABORATORY_FIELDS = {"text", "interpretation", "referenceRange", "method", "bodySite", "valueQuantity"}


def release_plan(package_root: Path) -> dict:
    path = package_root / "data/accepted-release.json"
    if sha256_file(path) != PLAN_SHA256:
        raise ValueError("Accepted-release reproduction plan has changed")
    plan = read_json(path)
    if plan["releaseId"] != RELEASE_ID or set(plan["populations"]) != {"1000", "10000"}:
        raise ValueError("Unsupported accepted-release plan")
    return plan


def _case(population_size: int, package_root: Path) -> tuple[dict, dict]:
    plan = release_plan(package_root)
    case = plan["populations"].get(str(population_size))
    if case is None:
        raise ValueError("Accepted-release reproduction supports exactly 1,000 or 10,000 people")
    return plan, case


def _encoded_chunks(value):
    """The release's exact JSON formatting, with bounded serialization buffers."""
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, indent=2)
    pending, size = [], 0
    for text in encoder.iterencode(value):
        pending.append(text)
        size += len(text)
        if size >= 256 * 1024:
            yield "".join(pending).encode("utf-8")
            pending, size = [], 0
    if pending:
        yield "".join(pending).encode("utf-8")
    yield b"\n"


def _bundle_identity(bundle: dict, path: Path | None = None) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    stream = path.open("xb") if path is not None else None
    try:
        for raw in _encoded_chunks(bundle):
            digest.update(raw)
            size += len(raw)
            if stream is not None:
                stream.write(raw)
    finally:
        if stream is not None:
            stream.close()
    return digest.hexdigest(), size


def _project_resource(resource: dict, spec: dict, source: str) -> None:
    actor = resource.get("resourceType") in {"Patient", "Practitioner"}
    lab = is_laboratory(resource)
    if not actor and not lab:
        return
    before = deepcopy(resource)
    if actor:
        cohort = next((row.get("value") for row in resource.get("identifier", [])
                       if row.get("system") == spec["identity"]["cohortIdentifierSystem"]), None)
        anchors = {row["cohortId"]: row["testId"] for row in spec["identity"]["anchorPatients"]}
        anchor = anchors.get(cohort) if resource["resourceType"] == "Patient" else None
        set_business_identifier(resource, spec, source, anchor)
    if lab:
        finalize_laboratory(resource)
    changed = {key for key in before.keys() | resource.keys() if before.get(key) != resource.get(key)}
    allowed = ({"identifier"} if actor else set()) | (LABORATORY_FIELDS if lab else set())
    if changed - allowed:
        raise ValueError("Laboratory/identifier projection exceeded its field boundary")


def _project_bundle(bundle: dict, spec: dict, source: str) -> None:
    # Retained originals were read from JSON before these repairs. Break any
    # in-memory sharing introduced by construction so repairs have the same scope.
    for entry in bundle["entry"]:
        entry["resource"] = json.loads(json.dumps(entry["resource"], ensure_ascii=False, sort_keys=True))
        _project_resource(entry["resource"], spec, source)


def apply_field_edits(resource: dict, operations: list[dict]) -> None:
    """Apply exact approved dictionary-field edits; arrays are replaced as units."""
    for operation in operations:
        path = operation["path"]
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError("Repair paths must select a resource field")
        parts = [part.replace("~1", "/").replace("~0", "~") for part in path[1:].split("/")]
        parent = resource
        for part in parts[:-1]:
            if not isinstance(parent, dict) or part not in parent:
                raise ValueError(f"Repair parent is missing: {path}")
            parent = parent[part]
        if not isinstance(parent, dict):
            raise ValueError(f"Repair parent must be an object: {path}")
        key, kind = parts[-1], operation["op"]
        if kind == "add":
            if key in parent:
                raise ValueError(f"Repair addition would overwrite a field: {path}")
            parent[key] = deepcopy(operation["value"])
        elif kind == "replace":
            if key not in parent:
                raise ValueError(f"Repair replacement field is missing: {path}")
            parent[key] = deepcopy(operation["value"])
        elif kind == "remove":
            if key not in parent:
                raise ValueError(f"Repair removal field is missing: {path}")
            del parent[key]
        else:
            raise ValueError(f"Unsupported field repair operation: {kind}")


def _native_generator_builder(package_root: Path, revision: str, source_digest: str):
    source = require_native_generator(package_root)
    actual = hashlib.sha256(json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if source["revision"] != revision or actual != source_digest:
        raise ValueError("Native generator code revision differs from the release plan")
    from .native_generator.bundle import build_population
    return build_population, package_root / source["root"]


def reproduce_release(population_size: int, out: Path, package_root: Path) -> Path:
    plan, case = _case(population_size, package_root)
    repair_path = package_root / case["repairs"]["path"]
    if sha256_file(repair_path) != case["repairs"]["sha256"]:
        raise ValueError("Accepted-release field repair definitions have changed")
    repairs = read_json(repair_path)
    if set(repairs) != set(SOURCES) or sum(len(rows) for rows in repairs.values()) != case["repairs"]["targets"]:
        raise ValueError("Accepted-release repair target inventory differs")
    builder, native_root = _native_generator_builder(package_root, plan["nativeGeneratorRevision"], plan["nativeGeneratorSha256"])
    target = validate_output_target(out)
    if target.exists():
        raise FileExistsError(f"Preserve existing release output: {target}")
    partial = target.with_name(f".{target.name}.partial-{os.getpid()}")
    if partial.exists():
        raise FileExistsError(f"Preserve partial release output: {partial}")
    remaining = sum(row["bytes"] for row in case["bundles"])
    reserve = case["nativeSpec"]["storage"]["minimumFreeReserveGiB"] * GIB

    def check_storage():
        free = available_bytes(target)
        required = remaining + reserve + 1024 * 1024
        if free < required:
            raise StorageError({"availableBytes": free, "requiredBytes": required, "reserveBytes": reserve})

    check_storage()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    started = time.monotonic()
    state = {"status": "running", "release": RELEASE_ID, "populationSize": population_size, "phase": "native-generation"}
    write_json(partial / "run-state.json", state)
    try:
        # This constructs resources from the seed and rules; no population export is read.
        built = builder(deepcopy(case["nativeSpec"]), native_root, monitor=check_storage)
        bundles = built["bundles"]
        del built
        for row in case["bundles"]:
            source = row["source"]
            state["phase"] = source
            write_json(partial / "run-state.json", state)
            bundle = bundles.pop(source)
            if len(bundle["entry"]) != row["resourceCount"]:
                raise ValueError(f"Constructed resource count differs: {source}")
            if _bundle_identity(bundle)[0] != row["nativeGeneratorSha256"]:
                raise ValueError(f"Native construction differs from the release baseline: {source}")
            _project_bundle(bundle, case["nativeSpec"], source)
            if _bundle_identity(bundle)[0] != row["projectedSha256"]:
                raise ValueError(f"Laboratory/identifier projection differs: {source}")
            pending = dict(repairs[source])
            for entry in bundle["entry"]:
                operations = pending.pop(entry["fullUrl"], None)
                if operations is not None:
                    apply_field_edits(entry["resource"], operations)
            if pending:
                raise ValueError(f"Missing release repair targets: {source}")
            check_storage()
            destination = partial / row["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest, size = _bundle_identity(bundle, destination)
            if digest != row["sha256"] or size != row["bytes"]:
                raise ValueError(f"Final bundle does not match the accepted release: {source}")
            remaining -= size
            del bundle
        manifest = {
            "acceptedRelease": RELEASE_ID, "datasetFamily": "synthetic-population",
            "datasetVersion": case["datasetVersion"], "populationSize": population_size,
            "resourceNamespace": case["nativeSpec"]["datasetVersion"],
            "resourceCount": sum(row["resourceCount"] for row in case["bundles"]),
            "integrity": "exact-published-source-bundle-sha256", "officialValidation": "not-run",
            "loadableSourceBundles": [{key: row[key] for key in ("source", "path", "bytes", "sha256", "resourceCount")} for row in case["bundles"]],
        }
        write_json(partial / "dataset-manifest.json", manifest)
        write_json(partial / "dataset-spec.json", {"acceptedRelease": RELEASE_ID, "populationSize": population_size})
        state.update(status="complete", phase="complete", elapsedSeconds=round(time.monotonic() - started, 3))
        write_json(partial / "run-state.json", state)
        (partial / "checksums.sha256").write_text("\n".join(checksum_lines(partial)) + "\n", encoding="utf-8")
        if target.exists():
            raise FileExistsError(f"Output appeared during reproduction: {target}")
        os.rename(partial, target)
        return target
    except BaseException as exc:
        state.update(status="failed", error=str(exc), elapsedSeconds=round(time.monotonic() - started, 3))
        if partial.is_dir():
            write_json(partial / "run-state.json", state)
        raise


def verify_release(out: Path, package_root: Path) -> dict:
    manifest = read_json(out / "dataset-manifest.json")
    _, case = _case(manifest["populationSize"], package_root)
    errors = []
    if manifest.get("acceptedRelease") != RELEASE_ID or manifest.get("datasetVersion") != case["datasetVersion"]:
        errors.append("Release identity differs")
    expected_count = sum(row["resourceCount"] for row in case["bundles"])
    if (manifest.get("datasetFamily") != "synthetic-population"
            or manifest.get("populationSize") != case["nativeSpec"]["populationSize"]
            or manifest.get("resourceCount") != expected_count
            or manifest.get("resourceNamespace") != case["nativeSpec"]["datasetVersion"]):
        errors.append("Release metadata differs from the reproduction plan")
    expected_spec = {"acceptedRelease": RELEASE_ID, "populationSize": manifest["populationSize"]}
    spec_path = out / "dataset-spec.json"
    if not spec_path.is_file() or read_json(spec_path) != expected_spec:
        errors.append("Release selection differs from the reproduction plan")
    state_path = out / "run-state.json"
    if not state_path.is_file() or read_json(state_path).get("status") != "complete":
        errors.append("Release reproduction is incomplete")
    checksum_path = out / "checksums.sha256"
    inventory = checksum_lines(out)
    current_hashes = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in inventory}
    expected_checksums = "\n".join(inventory) + "\n"
    if not checksum_path.is_file() or checksum_path.read_text(encoding="utf-8") != expected_checksums:
        errors.append("Output checksums differ from the current files")
    expected_rows = [{key: row[key] for key in ("source", "path", "bytes", "sha256", "resourceCount")} for row in case["bundles"]]
    if manifest.get("loadableSourceBundles") != expected_rows:
        errors.append("Manifest bundle inventory differs from the published release")
    for row in case["bundles"]:
        path = out / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or current_hashes.get(row["path"]) != row["sha256"]:
            errors.append("Accepted bundle checksum mismatch: " + row["source"])
    if manifest.get("officialValidation") != "not-run":
        errors.append("Reproduction must not claim a fresh official validation")
    return {"ok": not errors, "errors": errors, "acceptedRelease": RELEASE_ID,
            "resourceCount": expected_count if not errors else None, "expectedResourceCount": expected_count,
            "verification": "exact-published-source-bundle-sha256", "officialValidation": "not-run"}
