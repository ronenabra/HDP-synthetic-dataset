from __future__ import annotations

from datetime import date
from collections import defaultdict
from typing import Any

from .archetypes import distribution, primary_archetype, archetype_ordinal
from .nested_allocation import nested
from .catalogs import load_catalog


def age_at(birth_date: str, event_date: str) -> int:
    born = date.fromisoformat(birth_date[:10])
    event = date.fromisoformat(event_date[:10])
    return event.year - born.year - ((event.month, event.day) < (born.month, born.day))


def classify(spec: dict[str, Any], fact_code: str, role: str, birth_date: str, event_date: str) -> tuple[str | None, list[str], bool]:
    catalog = load_catalog(spec["sensitivity"]["catalog"])
    rule = next((item for item in catalog["rules"] if item["factCode"] == fact_code), None)
    if not rule:
        raise ValueError(f"unknown sensitive fact: {fact_code}")
    age_rule = rule.get("ageAtEvent")
    age = age_at(birth_date, event_date)
    if age_rule == "18+" and age < 18:
        raise ValueError(f"fact {fact_code} requires adult age")
    if age_rule == "under-18" and age >= 18:
        raise ValueError(f"fact {fact_code} requires minor age")
    if role in rule.get("stopAtRoles", []):
        return None, [], False
    if rule.get("propagateToRoles") and role not in rule["propagateToRoles"]:
        raise ValueError(f"role {role} is outside propagation policy for {fact_code}")
    return rule["confidentiality"], list(rule["clinicalSensitivity"]), bool(rule["nopat"])


def fact_for_patient(spec: dict[str, Any], index: int) -> str | None:
    if nested(spec):
        archetype = primary_archetype(spec, index)
        position = archetype_ordinal(spec, index, archetype)
        spec = {**spec, "populationSize": 1000,
                "sensitivity": {**spec["sensitivity"], "factCohorts": {
                    key: value // (spec["populationSize"] // 1000)
                    for key, value in spec["sensitivity"]["factCohorts"].items()}}}
    else:
        archetype = primary_archetype(spec, index)
        position = archetype_ordinal(spec, index, archetype)
    counts = spec["sensitivity"]["factCohorts"]
    placements = {
        "chronic-care": ("adult-hiv-positive", "adult-hiv-negative", "mental-health-diagnosis"),
        "pediatric": ("minor-hiv-positive",),
        "womens-health": ("protected-domestic-violence-indicator", "ordinary-injury"),
    }
    for row in distribution(spec):
        offset = 0
        for fact in placements.get(row["id"], ()):
            count = counts[fact]
            if row["id"] == archetype and offset <= position < offset + count:
                return fact
            offset += count
    return None


def fact_rule(spec: dict[str, Any], fact_code: str) -> dict[str, Any]:
    """Return the complete reviewed rule for a configured clinical fact."""
    catalog = load_catalog(spec["sensitivity"]["catalog"])
    rule = next((item for item in catalog["rules"] if item["factCode"] == fact_code), None)
    if rule is None:
        raise ValueError(f"unknown sensitive fact: {fact_code}")
    return rule


def owning_source(spec: dict[str, Any], fact_code: str) -> str:
    """Resolve fact ownership from the catalog, retaining the v2 compatibility rule."""
    rule = fact_rule(spec, fact_code)
    return rule.get("owningSource", {
        "protected-domestic-violence-indicator": "hospital-a",
        "ordinary-injury": "hospital-b",
    }.get(fact_code, "hmo"))


def requires_source(spec: dict[str, Any], index: int, organization: str) -> bool:
    """A configured fact is a mandatory capture event, never an optional utilization side effect."""
    fact = fact_for_patient(spec, index)
    return bool(fact and owning_source(spec, fact) == organization)


def realization_audit(bundles: dict[str, dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    """Reconcile requested fact subjects, graph multiplicity, and mobility behavior."""
    cohort_system = spec["identity"]["cohortIdentifierSystem"]
    patient_by_url: dict[tuple[str, str], str] = {}
    for organization, bundle in bundles.items():
        for entry in bundle["entry"]:
            resource = entry["resource"]
            if resource.get("resourceType") == "Patient":
                cohort = next((row.get("value") for row in resource.get("identifier", []) if row.get("system") == cohort_system), None)
                if cohort:
                    patient_by_url[(organization, entry["fullUrl"])] = cohort

    observed: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "subjects": set(), "resources": 0, "sources": set(), "roles": set(),
        "ordinaryMobilitySubjects": set(), "ordinaryMobilityResources": 0,
        "authorizedMobilitySubjects": set(), "authorizedMobilityResources": 0,
    })
    conf_system = spec["metadata"]["confidentialitySystem"]
    for organization, bundle in bundles.items():
        for entry in bundle["entry"]:
            resource = entry["resource"]
            tags = resource.get("meta", {}).get("tag", [])
            fact = next((row.get("code") for row in tags if row.get("system") == "urn:hdp-test-bundle:synthetic-sensitivity-fact"), None)
            if not fact:
                continue
            role = next((row.get("code") for row in tags if row.get("system") == "urn:hdp-test-bundle:synthetic-fact-role"), None)
            cohort = resource.get("subject", {}).get("identifier", {}).get("value")
            if not cohort:
                cohort = next((patient_by_url.get((organization, ref.get("reference"))) for ref in (resource.get("subject", {}), resource.get("patient", {})) if ref.get("reference")), None)
            if not cohort:
                raise ValueError(f"fact resource has no cohort identity: {organization}:{entry['fullUrl']}")
            confidentiality = {
                row.get("code") for row in resource.get("meta", {}).get("security", [])
                if row.get("system") == conf_system
            }
            row = observed[fact]
            row["subjects"].add(cohort)
            row["resources"] += 1
            row["sources"].add(organization)
            row["roles"].add(role)
            if not confidentiality:
                row["ordinaryMobilitySubjects"].add(cohort)
                row["ordinaryMobilityResources"] += 1
            if "V" not in confidentiality:
                row["authorizedMobilitySubjects"].add(cohort)
                row["authorizedMobilityResources"] += 1

    rows = []
    errors = []
    # Artifact JSON is written with sorted keys, so audit row order must not
    # depend on the in-memory order of the input spec.
    for fact, requested in sorted(spec["sensitivity"]["factCohorts"].items()):
        rule = fact_rule(spec, fact)
        actual = observed[fact]
        confidentiality = rule["confidentiality"]
        expected_multiplicity = int(rule.get("expectedResourceMultiplicity", 2 if confidentiality else 1))
        expected_resources = requested * expected_multiplicity
        expected_ordinary = requested if confidentiality is None else 0
        expected_authorized = 0 if confidentiality == "V" else requested
        expected_source = owning_source(spec, fact)
        result = {
            "factCode": fact,
            "requestedUniqueSubjects": requested,
            "emittedUniqueSubjects": len(actual["subjects"]),
            "expectedResourceMultiplicity": expected_multiplicity,
            "emittedResources": actual["resources"],
            "owningSource": expected_source,
            "observedSources": sorted(actual["sources"]),
            "factRoles": sorted(role for role in actual["roles"] if role),
            "confidentiality": confidentiality or "ordinary-unlabeled",
            "clinicalSensitivity": rule["clinicalSensitivity"],
            "nopat": bool(rule["nopat"]),
            "projections": {
                "internal": {"uniqueSubjects": len(actual["subjects"]), "resources": actual["resources"]},
                "ordinaryMobility": {"uniqueSubjects": len(actual["ordinaryMobilitySubjects"]), "resources": actual["ordinaryMobilityResources"]},
                "sensitivityAuthorized": {"uniqueSubjects": len(actual["authorizedMobilitySubjects"]), "resources": actual["authorizedMobilityResources"]},
            },
        }
        if result["emittedUniqueSubjects"] != requested:
            errors.append(f"{fact}: requested {requested} subjects, emitted {result['emittedUniqueSubjects']}")
        if result["emittedResources"] != expected_resources:
            errors.append(f"{fact}: expected {expected_resources} graph resources, emitted {result['emittedResources']}")
        expected_sources = [expected_source] if requested else []
        if result["observedSources"] != expected_sources:
            errors.append(f"{fact}: expected sources {expected_sources}, observed {result['observedSources']}")
        if result["projections"]["ordinaryMobility"]["uniqueSubjects"] != expected_ordinary:
            errors.append(f"{fact}: ordinary projection subject mismatch")
        if result["projections"]["sensitivityAuthorized"]["uniqueSubjects"] != expected_authorized:
            errors.append(f"{fact}: authorized projection subject mismatch")
        rows.append(result)
    return {"schemaVersion": "1.0.0", "status": "pass" if not errors else "fail", "facts": rows, "errors": errors}
