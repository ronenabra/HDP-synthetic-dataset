from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

from .blind import blind_resource
from .utils import read_json, write_json
from .artifacts import source_bundle_path


def aggregate(out: Path, destination: Path) -> Path:
    entries = []
    for org_id in ("hmo", "hospital-a", "hospital-b"):
        entries.extend(read_json(source_bundle_path(out, org_id))["entry"])
    write_json(destination, {"resourceType": "Bundle", "type": "collection", "entry": entries})
    return destination


def _references(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "reference" and isinstance(item, str):
                yield item
            else:
                yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


def mobility(out: Path, destination: Path, allow_restricted: bool = False,
             authorized_sensitivity: tuple[str, ...] = ()) -> Path:
    spec = read_json(out / "dataset-spec.json")
    conf_system = spec["metadata"]["confidentialitySystem"]
    sensitivity_system = spec["metadata"]["clinicalSensitivitySystem"]
    if allow_restricted and not authorized_sensitivity:
        raise ValueError("restricted mobility requires at least one explicit authorized sensitivity code")
    entries = []
    excluded = {"V": 0, "R": 0, "referenceClosure": 0}
    for org_id in ("hmo", "hospital-a", "hospital-b"):
        for entry in read_json(source_bundle_path(out, org_id))["entry"]:
            codes = {item.get("code") for item in entry["resource"].get("meta", {}).get("security", []) if item.get("system") == conf_system}
            if "V" in codes:
                excluded["V"] += 1
                continue
            if "R" in codes:
                clinical_codes = {item.get("code") for item in entry["resource"].get("meta", {}).get("security", []) if item.get("system") == sensitivity_system}
                if not allow_restricted or not clinical_codes or not clinical_codes.issubset(set(authorized_sensitivity)):
                    excluded["R"] += 1
                    continue
            entries.append(entry)
    while True:
        urls = {entry["fullUrl"] for entry in entries}
        kept = [entry for entry in entries if not any(ref.startswith("urn:uuid:") and ref not in urls for ref in _references(entry["resource"]))]
        removed = len(entries) - len(kept)
        excluded["referenceClosure"] += removed
        entries = kept
        if not removed:
            break
    write_json(destination, {"resourceType": "Bundle", "type": "collection", "meta": {"tag": [{"system": "urn:hdp-test-bundle:projection", "code": "mobility"}, {"system": "urn:hdp-test-bundle:restricted-authorization", "code": "allowed" if allow_restricted else "denied"}]}, "entry": entries, "_projectionEvidence": {"excluded": excluded, "authorizedSensitivityCodes": sorted(authorized_sensitivity), "danglingReferences": 0}})
    return destination


def challenge(out: Path, destination: Path) -> Path:
    spec = read_json(out / "dataset-spec.json")
    entries = []
    for org_id in ("hmo", "hospital-a", "hospital-b"):
        for entry in read_json(source_bundle_path(out, org_id))["entry"]:
            copied = copy.deepcopy(entry)
            copied["resource"] = blind_resource(copied["resource"], spec)
            entries.append(copied)
    write_json(destination, {"resourceType": "Bundle", "type": "collection", "meta": {"tag": [{"system": "urn:hdp-test-bundle:projection", "code": "blind-challenge"}]}, "entry": entries})
    return destination
