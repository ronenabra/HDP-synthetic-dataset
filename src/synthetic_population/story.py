from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from typing import Any, Iterable

from .catalogs import load_catalog


PLACEHOLDER_CODES = {
    ("urn:hdp-test-bundle:synthetic:communication-type", "SolicitedAttachmentRequest"),
    ("urn:hdp-test-bundle:synthetic:flag-code", "bigdog"),
    ("urn:hdp-test-bundle:synthetic:diet-type", "1040"),
    ("http://snomed.info/sct", "346313005"),
}


def _codings(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("system"), str) and isinstance(value.get("code"), str):
            yield value
        for child in value.values():
            yield from _codings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _codings(child)


def build_story_coverage_audit(
    bundles: dict[str, dict[str, Any]], crosswalk: dict[str, Any], spec: dict[str, Any],
    coverage_bundles: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    realism_catalog = load_catalog(spec["clinicalRealism"]["catalog"])
    immunization_decision = realism_catalog.get("pathwayReleaseDecisions", {}).get(
        "pediatric-immunization", {}
    )
    cohort_system = spec["identity"]["cohortIdentifierSystem"]
    patient_by_url: dict[tuple[str, str], dict[str, Any]] = {}
    placeholder_occurrences = []
    integrated_profiles: Counter[str] = Counter()
    isolated_profiles: Counter[str] = Counter()
    immunizations: list[dict[str, Any]] = []
    canonical_coverage_resources = 0
    for source, bundle in bundles.items():
        for entry in bundle["entry"]:
            resource = entry["resource"]
            compact = any(
                tag.get("system") == "urn:hdp-test-bundle:synthetic-archetype" and tag.get("code") == "compact-coverage"
                for tag in resource.get("meta", {}).get("tag", [])
            )
            canonical_coverage_resources += int(compact)
            target = isolated_profiles if compact else integrated_profiles
            target.update(resource.get("meta", {}).get("profile", []))
            for coding in _codings(resource):
                if (coding["system"], coding["code"]) in PLACEHOLDER_CODES or coding.get("display") in {"Big dog", "Allergen extract vaccines"}:
                    placeholder_occurrences.append({
                        "organization": source, "fullUrl": entry["fullUrl"], "resourceType": resource["resourceType"],
                        "system": coding["system"], "code": coding["code"], "display": coding.get("display"),
                    })
            if resource["resourceType"] == "Patient":
                cohort = next((row.get("value") for row in resource.get("identifier", []) if row.get("system") == cohort_system), None)
                if cohort:
                    patient_by_url[(source, entry["fullUrl"])] = resource
            elif resource["resourceType"] == "Immunization":
                reference = resource.get("patient", {}).get("reference")
                patient = patient_by_url.get((source, reference))
                if patient:
                    coding = next(_codings(resource.get("vaccineCode")), {})
                    immunizations.append({
                        "organization": source, "cohortBirthDate": patient["birthDate"],
                        "occurrenceDate": resource.get("occurrenceDateTime", "")[:10],
                        "code": coding.get("code"), "display": coding.get("display"),
                    })
    for bundle in (coverage_bundles or {}).values():
        for entry in bundle.get("entry", []):
            isolated_profiles.update(entry["resource"].get("meta", {}).get("profile", []))
    by_age_band: dict[str, Counter[str]] = defaultdict(Counter)
    invalid_dates = []
    for row in immunizations:
        born = date.fromisoformat(row["cohortBirthDate"])
        occurred = date.fromisoformat(row["occurrenceDate"])
        age_days = (occurred - born).days
        if age_days < 0:
            invalid_dates.append(row)
        band = "infant" if age_days < 365 else "child" if age_days < 13 * 365 else "adolescent"
        by_age_band[band][row["code"]] += 1
    result = {
        "schemaVersion": "1.0.0",
        "patientStoryPlaceholderOccurrenceCount": len(placeholder_occurrences),
        "patientStoryPlaceholderOccurrences": placeholder_occurrences,
        "integratedProfileCoverage": dict(sorted(integrated_profiles.items())),
        "isolatedCoverageFixtureProfiles": dict(sorted(isolated_profiles.items())),
        "isolatedCoverageResourceCount": sum(isolated_profiles.values()),
        "coverageOnlyResourcesInPatientStoryBundles": canonical_coverage_resources,
        "pediatricImmunization": {
            "count": len(immunizations),
            "distinctConceptCount": len({row["code"] for row in immunizations}),
            "conceptCountsByAgeAtAdministration": {band: dict(sorted(counts.items())) for band, counts in sorted(by_age_band.items())},
            "preBirthCount": len(invalid_dates),
            "releaseDecision": immunization_decision,
            "generationMode": immunization_decision.get("generationDefault", "enabled"),
        },
        "releaseEligible": bool(immunization_decision.get("defaultCandidatePermitted", True)),
        "releaseBlockers": ([immunization_decision] if not immunization_decision.get("defaultCandidatePermitted", True) else []),
    }
    errors = []
    if placeholder_occurrences:
        errors.append("canonical patient-story bundles contain pedagogical placeholder concepts")
    if canonical_coverage_resources:
        errors.append("coverage-only resources remain inside canonical patient-story bundles")
    if immunizations and result["pediatricImmunization"]["distinctConceptCount"] < 3:
        errors.append("routine pediatric histories use fewer than three vaccine concepts")
    if invalid_dates:
        errors.append("pediatric immunization precedes birth")
    if immunization_decision.get("generationDefault") == "excluded" and immunizations:
        errors.append("release-blocked pediatric immunizations were emitted in the default candidate")
    result["diagnostics"] = errors
    result["status"] = "pass" if not errors else "fail"
    return result
