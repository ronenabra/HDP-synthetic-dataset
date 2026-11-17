from __future__ import annotations

import uuid


ROOT_NAMESPACE = uuid.UUID("310cae4d-c9e5-58c7-a4fb-c4ee84a54a21")


def deterministic_uuid(dataset_version: str, *parts: object) -> str:
    name = "|".join([dataset_version, *(str(part) for part in parts)])
    return str(uuid.uuid5(ROOT_NAMESPACE, name))


def full_url(dataset_version: str, *parts: object) -> str:
    return f"urn:uuid:{deterministic_uuid(dataset_version, *parts)}"


def cohort_id(index: int) -> str:
    return f"cohort-{index:06d}"


def strong_identifier(spec: dict, organization: str, resource_type: str, value: str) -> dict:
    return {
        "system": f"{spec['identity']['strongIdentifierPrefix']}:{organization}:{resource_type.lower()}",
        "value": value,
        "type": {"coding": [{"system": "http://fhir.health.gov.il/cs/il-core-identifier-type", "code": "strong-id"}]},
    }
