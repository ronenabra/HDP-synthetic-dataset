from __future__ import annotations

import re
from typing import Any, Iterable

from .catalogs import load_catalog
from .business_identifiers import TEST_SYSTEM, SANDBOX_SYSTEM, PRACTITIONER_SYSTEM, resource_errors, identifier_rows


FORBIDDEN_PRODUCTION_IDENTIFIER_SYSTEMS = {
    "http://fhir.health.gov.il/identifier/il-national-id",
    "http://fhir.health.gov.il/identifier/pna-national-id",
}
NINE_DIGIT_ID = re.compile(r"^\d{9}$")


def _objects(value: Any) -> Iterable[dict]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _objects(item)


def scan(resource: dict, spec: dict) -> list[str]:
    errors = []
    if spec.get("schemaVersion") == "2.0.0":
        identity_policy = load_catalog(spec["identityPolicy"]["catalog"])
        safe_system = identity_policy["safeRepresentation"]["system"]
        if spec["identity"]["testIdentifierSystem"] != safe_system:
            errors.append("test identifier namespace differs from the reviewed identity policy")
    archetypes = {
        tag.get("code") for tag in resource.get("meta", {}).get("tag", [])
        if tag.get("system") == "urn:hdp-test-bundle:synthetic-archetype"
    }
    if "compact-coverage" in archetypes:
        return errors
    for item in _objects(resource):
        system = item.get("system")
        value = item.get("value")
        if system in FORBIDDEN_PRODUCTION_IDENTIFIER_SYSTEMS:
            errors.append(f"forbidden production identifier system: {system}")
        if isinstance(system, str) and "passport" in system and "encrypted-passport" not in system:
            errors.append(f"forbidden plaintext passport system: {system}")
        if isinstance(value, str) and NINE_DIGIT_ID.fullmatch(value) and system not in {
            spec["identity"]["testIdentifierSystem"],
        }:
            errors.append("nine-digit production-like identifier value")
        if system in {TEST_SYSTEM, SANDBOX_SYSTEM, PRACTITIONER_SYSTEM} and isinstance(value, str) and NINE_DIGIT_ID.fullmatch(value):
            errors.append("production-like value in synthetic business namespace")
        if system == "email":
            errors.append("email contacts are forbidden in canonical synthetic truth")
        if system == "phone" and isinstance(value, str) and not value.startswith(spec["demographics"]["telephonePrefix"]):
            errors.append("phone does not use the reserved synthetic prefix")
    if spec.get("schemaVersion") == "2.0.0" and identity_policy.get("businessIdentifiers"):
        cohort = next((row.get("value") for row in identifier_rows(resource)
                       if row.get("system") == spec["identity"]["cohortIdentifierSystem"]), None)
        anchor = next((row["testId"] for row in spec["identity"]["anchorPatients"]
                       if row["cohortId"] == cohort), None) if resource.get("resourceType") == "Patient" else None
        errors.extend(resource_errors(resource, anchor))
    if resource.get("resourceType") in {"Patient", "RelatedPerson"}:
        for address in resource.get("address", []):
            if not any("Synthetic" in line for line in address.get("line", [])):
                errors.append("address line is not visibly synthetic")
    return errors
