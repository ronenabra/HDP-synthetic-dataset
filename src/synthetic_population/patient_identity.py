"""Patient lookup and attributable intentional duplicate identities."""
from __future__ import annotations

from typing import Any
from pathlib import Path

from .utils import sha256_json, read_json


def identifiers(resource: dict) -> list[dict]:
    value = resource.get("identifier", [])
    return [value] if isinstance(value, dict) else value


def patient_index(bundle: dict, spec: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Index all real cohort Patients before looking at any clinical facts."""
    cohorts, births = {}, {}
    for entry in bundle.get("entry", []):
        resource = entry["resource"]
        if resource.get("resourceType") != "Patient":
            continue
        cohort = next((row.get("value") for row in identifiers(resource)
                       if row.get("system") == spec["identity"]["cohortIdentifierSystem"]), None)
        if cohort:
            cohorts[entry["fullUrl"]] = cohort
            births[entry["fullUrl"]] = resource.get("birthDate")
    return cohorts, births


def duplicate_root_evidence(bundles: dict, spec: dict, answer: dict, crosswalk: dict) -> dict:
    """Bind every declared overlay to its exact counterpart and problem record.

    This is eligibility evidence only: overlays remain in the original source
    Bundles and in forward reference closure. Tags alone never authorize omission.
    """
    result: dict[str, dict[str, Any]] = {source: {} for source in bundles}
    problems = {row["problemId"]: row for row in answer.get("problems", [])}
    if len(problems) != len(answer.get("problems", [])):
        raise ValueError("duplicate problem identities in root provenance")
    cohort_rows = crosswalk.get("patients", [])
    if len({row["cohortId"] for row in cohort_rows}) != len(cohort_rows):
        raise ValueError("duplicate cohort identities in root provenance")
    by_source = {}
    for source, bundle in bundles.items():
        entries = {row["fullUrl"]: row["resource"] for row in bundle["entry"]}
        if len(entries) != len(bundle["entry"]):
            raise ValueError("duplicate fullUrl in root provenance")
        by_source[source] = entries
    for cohort in cohort_rows:
        for source, view in cohort.get("sources", {}).items():
            if source not in bundles:
                continue
            entries = by_source[source]
            for duplicate in view.get("duplicateCandidates", []):
                url = duplicate["patientFullUrl"]
                resource = entries.get(url, {})
                canonical_url = view.get("patientFullUrl")
                canonical = entries.get(canonical_url, {})
                tags: dict[str, list[str]] = {}
                for row in resource.get("meta", {}).get("tag", []):
                    tags.setdefault(row.get("system"), []).append(row.get("code"))
                problem_ids = tags.get(spec["metadata"]["problemIdSystem"], [])
                problem = problems.get(problem_ids[0], {}) if len(problem_ids) == 1 else {}
                participants = {(row.get("organization"), row.get("fullUrl"))
                                for row in problem.get("participants", [])}
                if (url in result[source] or url == canonical_url
                        or resource.get("resourceType") != "Patient"
                        or canonical.get("resourceType") != "Patient"
                        or duplicate.get("canonicalCohortId") != cohort["cohortId"]
                        or problem.get("cohortId") != cohort["cohortId"]
                        or problem.get("kind") != "probable-duplicate"
                        or not {(source, url), (source, canonical_url)} <= participants
                        or tags.get(spec["metadata"]["problemMarkerSystem"]) != ["intentional-problem"]
                        or tags.get(spec["metadata"]["problemKindSystem"]) != ["probable-duplicate"]
                        or any(row.get("system") == spec["identity"]["cohortIdentifierSystem"]
                               for row in identifiers(resource))
                        or [row.get("value") for row in identifiers(canonical)
                            if row.get("system") == spec["identity"]["cohortIdentifierSystem"]] != [cohort["cohortId"]]):
                    raise ValueError(f"unattributable intentional duplicate Patient: {source}/{url}")
                result[source][url] = {
                    "resourceSha256": sha256_json(resource), "canonicalPatientFullUrl": canonical_url,
                    "canonicalCohortId": cohort["cohortId"], "problemId": problem["problemId"],
                    "answerProblemSha256": sha256_json(problem), "crosswalkRowSha256": sha256_json(cohort),
                }
    return result


def load_duplicate_evidence(out: Path, bundles: dict, spec: dict) -> dict:
    # Provenance files are unnecessary for an input with no overlay Patients.
    cohort_system = spec["identity"]["cohortIdentifierSystem"]
    overlays = any(entry["resource"].get("resourceType") == "Patient" and not any(
        row.get("system") == cohort_system for row in identifiers(entry["resource"]))
        for bundle in bundles.values() for entry in bundle["entry"])
    if not overlays:
        return {}
    return duplicate_root_evidence(bundles, spec, read_json(out / "problem-answer-key.json"),
                                   read_json(out / "cohort-crosswalk.json"))
