"""Focused S18 checks at the source-record and reconciled-fact grains."""


import synthetic_population

import copy
from pathlib import Path

import pytest

from synthetic_population.bundle import build_population
from synthetic_population.condition_aggregate import build_condition_aggregate
from synthetic_population.condition_allocation import hamilton_counts
from synthetic_population.condition_coding import CLASS_SYSTEM, semantic_condition_codings
from synthetic_population.observable import codings


ROOT = Path(synthetic_population.__file__).resolve().parent


@pytest.fixture()
def built_conditions(condition_spec):
    spec = condition_spec
    return spec, build_population(spec, ROOT)


def _audit(spec, built, bundles=None):
    return build_condition_aggregate(
        bundles or built["bundles"], built["crosswalk"], built["latentTruthLedger"], spec, ROOT,
    )


def _first_condition(bundles, kind):
    for bundle in bundles.values():
        for entry in bundle["entry"]:
            resource = entry["resource"]
            if resource.get("resourceType") == "Condition" and any(
                row.get("system") == CLASS_SYSTEM and row.get("code") == kind
                for row in resource.get("meta", {}).get("tag", [])
            ):
                return resource
    raise AssertionError(kind)


def test_reconciles_classes_aliases_and_protected_support(built_conditions):
    spec, built = built_conditions
    audit = _audit(spec, built)
    assert audit["status"] == "pass", audit["errors"]
    n = audit["denominators"]["sourceConditionsN"]
    assert tuple(audit["actualClasses"][kind] for kind in (
        "local-only", "standard-plus-local", "standard-only",
    )) == hamilton_counts(n)
    assert audit["denominators"]["rawConditionCodingTokens"] == n + audit["actualClasses"]["standard-plus-local"]
    assert audit["denominators"]["clinicalFactsF"] < n  # linked protected support rows
    assert audit["distinctOrdinaryConcepts"] >= 12
    assert audit["unsupportedCodes"] == []


def test_rejects_mapping_version_and_ordinary_injury_drift(built_conditions):
    spec, built = built_conditions
    version_drift = copy.deepcopy(built["bundles"])
    local = _first_condition(version_drift, "standard-plus-local")
    next(row for row in local["code"]["coding"] if row["system"].startswith("urn:hdp-test-bundle:synthetic:condition"))["version"] = "9.9.9"
    assert any("drifted local" in error for error in _audit(spec, built, version_drift)["errors"])

    injury_drift = copy.deepcopy(built["bundles"])
    ordinary_people = {row["cohortId"] for row in built["crosswalk"]["patients"]
                       if row["evaluationClass"] == "ordinary"}
    wound = next(
        entry["resource"] for bundle in injury_drift.values() for entry in bundle["entry"]
        if entry["resource"].get("resourceType") == "Condition"
        and entry["resource"].get("code", {}).get("coding", [{}])[0].get("code") == "125643001"
        and entry["resource"].get("subject", {}).get("identifier", {}).get("value") in ordinary_people
        and not any(row.get("code") == "intentional-problem" for row in entry["resource"]["meta"]["tag"])
    )
    wound["code"]["coding"][0].update({"code": "44465007", "display": "Sprain of ankle"})
    errors = _audit(spec, built, injury_drift)["errors"]
    assert any("contradicts selected pathway branch" in error for error in errors)
    assert any("differ from frozen conditional expectation" in error for error in errors)


def test_semantic_comparison_keeps_local_only_warning_visible(built_conditions):
    _spec, built = built_conditions
    resource = _first_condition(built["bundles"], "local-only")
    assert not any(system == "http://snomed.info/sct" for system, _code in codings(resource["code"]))
    assert any(system == "http://snomed.info/sct" for system, _code in semantic_condition_codings(resource))
    drifted = copy.deepcopy(resource)
    drifted["code"]["coding"][0]["version"] = "9.9.9"
    assert not any(system == "http://snomed.info/sct" for system, _code in semantic_condition_codings(drifted))
