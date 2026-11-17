from __future__ import annotations

from typing import Iterable

from .buckets import bucket_for_role


def meta(spec: dict, organization: str, profile: str, semantic_role: str, *, confidentiality: str | None = None,
         sensitivity: Iterable[str] = (), nopat: bool = False, problem_ids: Iterable[str] = (),
         problem_kinds: Iterable[str] = (), archetype: str = "strict-patient-clinical") -> dict:
    md = spec["metadata"]
    bucket = bucket_for_role(semantic_role)
    security = [
        {"system": md["bucketSystem"], "code": bucket},
        {"system": md["actReasonSystem"], "code": md["testReasonCode"]},
    ]
    if confidentiality:
        if confidentiality not in md["allowedConfidentialityCodes"]:
            raise ValueError(f"invalid confidentiality code: {confidentiality}")
        security.append({"system": md["confidentialitySystem"], "code": confidentiality})
    security.extend({"system": md["clinicalSensitivitySystem"], "code": code} for code in sensitivity)
    if nopat:
        security.append({"system": md["clinicalSensitivitySystem"], "code": "NOPAT"})
    tags = [
        {"system": md["datasetTagSystem"], "code": spec["datasetVersion"]},
        {"system": md["sourceTagSystem"], "code": organization},
        {"system": "urn:hdp-test-bundle:synthetic-archetype", "code": archetype},
    ]
    problem_ids = list(problem_ids)
    if problem_ids:
        tags.append({"system": md["problemMarkerSystem"], "code": "intentional-problem"})
        tags.extend({"system": md["problemKindSystem"], "code": kind} for kind in sorted(set(problem_kinds)))
        tags.extend({"system": md["problemIdSystem"], "code": problem_id} for problem_id in sorted(problem_ids))
    return {"profile": [profile], "tag": tags, "security": security}


def strip_problem_tags(resource: dict, spec: dict) -> dict:
    systems = {
        spec["metadata"]["problemMarkerSystem"],
        spec["metadata"]["problemKindSystem"],
        spec["metadata"]["problemIdSystem"],
    }
    resource.get("meta", {})["tag"] = [tag for tag in resource.get("meta", {}).get("tag", []) if tag.get("system") not in systems]
    return resource
