"""S24 business identifiers, separate from source-local strong identifiers."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from uuid import UUID

TEST_SYSTEM = "http://fhir.health.gov.il/identifier/il-hdp-test-id"
SANDBOX_SYSTEM = "http://fhir.health.gov.il/identifier/il-hdp-sandbox-synthetic-id"
PRACTITIONER_SYSTEM = "http://fhir.health.gov.il/identifier/encrypted-id-primary-moh"
BUSINESS_SYSTEMS = {TEST_SYSTEM, SANDBOX_SYSTEM, PRACTITIONER_SYSTEM}


def business_guid(spec: dict, source: str, resource_type: str, strong_value: str) -> str:
    """Seed-controlled pseudorandom UUIDv4; replay never changes identifiers."""
    key = json.dumps(["hdp-business-id-v1", spec.get("seed"), spec["datasetVersion"],
                      source, resource_type, strong_value], separators=(",", ":"))
    return str(UUID(bytes=hashlib.sha256(key.encode()).digest()[:16], version=4))


def identifier_rows(resource: dict) -> list[dict]:
    rows = resource.get("identifier", [])
    return [rows] if isinstance(rows, dict) else rows


def is_guid(value: object) -> bool:
    try:
        return isinstance(value, str) and str(UUID(value)) == value.lower()
    except (ValueError, AttributeError):
        return False


def set_business_identifier(resource: dict, spec: dict, source: str,
                            anchor: str | None = None) -> None:
    """Change business identifiers only; strong/cohort identifiers stay intact."""
    rtype = resource["resourceType"]
    if rtype not in {"Patient", "Practitioner"}:
        return
    rows = identifier_rows(resource)
    strong = next(row for row in rows if any(
        code.get("system") == "http://fhir.health.gov.il/cs/il-core-identifier-type"
        and code.get("code") == "strong-id" for code in row.get("type", {}).get("coding", [])))
    business = ({"system": TEST_SYSTEM, "value": anchor} if anchor else {
        "system": SANDBOX_SYSTEM if rtype == "Patient" else PRACTITIONER_SYSTEM,
        "value": business_guid(spec, source, rtype, strong["value"]),
    })
    resource["identifier"] = [row for row in rows if row.get("system") not in BUSINESS_SYSTEMS] + [business]


def resource_errors(resource: dict, anchor: str | None = None) -> list[str]:
    rtype = resource.get("resourceType")
    if rtype not in {"Patient", "Practitioner"}:
        return []
    rows = [row for row in identifier_rows(resource) if row.get("system") in BUSINESS_SYSTEMS]
    system = TEST_SYSTEM if anchor else SANDBOX_SYSTEM if rtype == "Patient" else PRACTITIONER_SYSTEM
    if len(rows) != 1 or rows[0].get("system") != system:
        return ["business identifier cardinality/namespace mismatch"]
    value = rows[0].get("value")
    if anchor:
        return [] if anchor in {"1", "2"} and value == anchor else ["anchor test identifier mismatch"]
    return [] if is_guid(value) else ["business identifier must be a GUID"]


def bundle_errors(bundles: dict, spec: dict) -> list[str]:
    """Check identities across sources, including overlap with existing GUIDs."""
    errors, guids, existing = [], {}, {}
    anchors = {row["cohortId"]: row["testId"] for row in spec["identity"]["anchorPatients"]}
    for source, bundle in bundles.items():
        counts = Counter()
        for entry in bundle.get("entry", []):
            resource = entry["resource"]
            owner = (source, entry["fullUrl"])
            rows = identifier_rows(resource)
            cohort = next((row.get("value") for row in rows
                           if row.get("system") == spec["identity"]["cohortIdentifierSystem"]), None)
            anchor = anchors.get(cohort) if resource.get("resourceType") == "Patient" else None
            errors.extend(f"{source}:{entry['fullUrl']}: {error}" for error in resource_errors(resource, anchor))
            for row in rows:
                value = row.get("value")
                if row.get("system") == TEST_SYSTEM:
                    counts[value] += 1
                if is_guid(value):
                    key = value.lower()
                    if row.get("system") in {SANDBOX_SYSTEM, PRACTITIONER_SYSTEM}:
                        if key in guids:
                            errors.append(f"{source}:{entry['fullUrl']}: business GUID collision")
                        guids[key] = owner
                    else:
                        existing[key] = owner
        if counts != Counter({"1": 1, "2": 1}):
            errors.append(f"{source}: expected exactly two test anchors with values 1 and 2")
    for value in guids.keys() & existing.keys():
        errors.append(f"{guids[value]}: business GUID overlaps an existing identifier")
    return errors
