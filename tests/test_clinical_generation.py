import copy
import json

import pytest

from synthetic_population.blind import _identity_key
from synthetic_population.buckets import BUCKET_REQUIREMENT_IDS, COVERAGE_RESOURCE_ROLES, ROLE_TO_BUCKET, bucket_for_role, coverage_role
from synthetic_population.detectors import detects, detect_group, normalized_quantity, stale_snapshot_evidence
from synthetic_population.grammar import CLAUSE_CONTRACTS, GRAMMARS, PREDICATES
from synthetic_population.realism import _frequency_metrics, _resource_signature
from synthetic_population.utils import read_json


def test_blind_identity_uses_observable_normalized_phone():
    # Contact variation must not break independent name/date/sex linkage.
    base = {"name": [{"family": "Cohen", "given": ["Rina"]}], "birthDate": "1980-01-01", "gender": "female", "telecom": [{"system": "phone", "value": "+972-50-1234567"}]}
    source_variant = copy.deepcopy(base)
    source_variant["telecom"][0]["value"] = "+972 50 1234567x4"
    different = copy.deepcopy(base)
    different["telecom"][0]["value"] = "+972-50-7654321"
    spec = {"schemaVersion": "2.0.0"}
    assert _identity_key(base, spec) == _identity_key(source_variant, spec)
    assert _identity_key(base, spec) == _identity_key(different, spec)
    different["name"][0]["family"] = "Levi"
    assert _identity_key(base, spec) != _identity_key(different, spec)


def test_semantic_bucket_catalog_is_total_and_rejects_fallback(repository_root):
    catalog = read_json(repository_root / "data/bucket-semantics.v2.0.1.json")
    assert catalog["roles"] == ROLE_TO_BUCKET
    assert catalog["coverageResourceRoles"] == COVERAGE_RESOURCE_ROLES
    assert len(set(ROLE_TO_BUCKET.values())) == 19
    assert set(BUCKET_REQUIREMENT_IDS) == set(ROLE_TO_BUCKET.values())
    assert all(value.startswith("BUCKET-") for value in BUCKET_REQUIREMENT_IDS.values())
    assert coverage_role("CarePlan", "x") == "recommendation"
    assert coverage_role("ImagingStudy", "x") == "diagnostic-imaging"
    assert coverage_role("MolecularSequence", "x") == "genetic-result"
    assert coverage_role("ResearchStudy", "x") == "research-participation"
    assert coverage_role("QuestionnaireResponse", "x") == "assessment"
    assert coverage_role("Immunization", "x") == "immunization"
    assert coverage_role("ImmunizationRecommendation", "x") == "recommendation"
    assert coverage_role("Patient", "x") == "patient-demographics"
    assert coverage_role("Observation", "http://fhir.health.gov.il/StructureDefinition/il-core-immunization-reaction-observation") == "immunization"
    assert coverage_role("Specimen", "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-specimen-pathology") == "pathology"
    with pytest.raises(ValueError, match="unknown.*semantic bucket role"):
        bucket_for_role("copied-compact-default")
    with pytest.raises(ValueError, match="no reviewed semantic bucket role"):
        coverage_role("UnknownResource", "x")


def test_blind_evaluation_is_one_to_one_for_overlapping_truths():
    from synthetic_population.blind import evaluate_candidates

    candidate = {"kind": "timestamp-order", "resourceFullUrls": ["urn:a", "urn:b"]}
    answer = {"problems": [
        {"kind": "timestamp-order", "problemId": "p1", "participants": [{"fullUrl": "urn:a"}]},
        {"kind": "timestamp-order", "problemId": "p2", "participants": [{"fullUrl": "urn:b"}]},
    ]}
    result = evaluate_candidates([candidate], answer, {"scenarios": []})["byProblemKind"]["timestamp-order"]
    assert result["truePositiveCount"] == 1
    assert result["falseNegativeCount"] == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5


def test_blind_group_detectors_use_semantic_shape():
    conflict = [
        {"resourceType": "Condition", "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]}, "code": {"coding": [{"system": "http://snomed.info/sct", "code": "46635009"}]}},
        {"resourceType": "Condition", "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]}, "code": {"coding": [{"system": "http://snomed.info/sct", "code": "44054006"}]}},
    ]
    discordant = [
        {"resourceType": "Observation", "status": "final", "effectiveDateTime": "2025-01-01T10:00:00Z", "code": {"coding": [{"system": "http://loinc.org", "code": "6298-4"}]}, "valueQuantity": {"value": 4.1, "code": "mmol/L"}},
        {"resourceType": "Observation", "status": "final", "effectiveDateTime": "2025-01-01T10:10:00Z", "code": {"coding": [{"system": "http://loinc.org", "code": "6298-4"}]}, "valueQuantity": {"value": 5.1, "code": "mmol/L"}},
    ]
    duplicate = [
        {"resourceType": "Patient", "birthDate": "1980-01-01", "gender": "female", "name": [{"family": "Cohen", "given": ["Rina"]}], "telecom": [{"value": "+972-50-1234567"}], "address": [{"city": "Haifa"}]},
        {"resourceType": "Patient", "birthDate": "1980-01-01", "gender": "female", "name": [{"family": "Cohen", "given": ["Rinah"]}], "telecom": [{"value": "+972-50-1234567"}], "address": [{"city": "Haifa"}]},
    ]
    assert detect_group(conflict, "conflicting-diagnosis")
    assert not detect_group(discordant, "discordant-lab")
    for row in discordant:
        row["specimen"] = {"reference": "urn:uuid:shared-specimen"}
    assert detect_group(discordant, "discordant-lab")
    unrelated = copy.deepcopy(discordant)
    unrelated[1]["specimen"]["reference"] = "urn:uuid:other-specimen"
    assert not detect_group(unrelated, "discordant-lab")
    assert detect_group(duplicate, "probable-duplicate")
    assert not detect_group(conflict[:1], "conflicting-diagnosis")
    assert not detect_group(discordant[:1], "discordant-lab")
    assert not detect_group(duplicate[:1], "probable-duplicate")


def test_detector_semantics_are_versioned_and_truth_free(repository_root):
    catalog = read_json(repository_root / "data/detector-semantics.v1.2.0.json")
    assert catalog["status"] == "frozen-before-canonical-generation"
    serialized = json.dumps(catalog, sort_keys=True).lower()
    assert all(token not in serialized for token in ("problemid", "cohortid", "scenarioid", "generator tag"))
    wrong_codes = {"119297000", "368208006", "396550006"}
    for code in wrong_codes:
        resource = {
            "resourceType": "Condition",
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": code}]},
        }
        assert detects(resource, "semantically-wrong-code")
    ordinary = {
        "resourceType": "Condition",
        "code": {"coding": [{"system": "http://snomed.info/sct", "code": "38341003"}]},
    }
    assert not detects(ordinary, "semantically-wrong-code")


def test_structural_signature_rejects_superficial_uniqueness():
    base = {
        "resourceType": "Observation",
        "id": "one",
        "meta": {
            "profile": ["http://example.test/StructureDefinition/lab"],
            "tag": [{"system": "urn:hdp-test-bundle:synthetic-source", "code": "hmo"}],
        },
        "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": "718-7", "display": "Hemoglobin"}]},
        "subject": {"reference": "urn:uuid:one"},
        "effectiveDateTime": "2025-01-01T10:00:00Z",
        "valueQuantity": {"value": 13.1, "unit": "g/dL"},
        "note": [{"text": "one"}],
    }
    superficial = copy.deepcopy(base)
    superficial.update({"id": "two", "effectiveDateTime": "2026-07-09T12:34:00Z"})
    superficial["valueQuantity"]["value"] = 15.8
    superficial["note"][0]["text"] = "random uniqueness"
    superficial["meta"]["tag"][0]["code"] = "hospital-b"
    assert _resource_signature(base, ("Patient",)) == _resource_signature(superficial, ("Patient",))
    superficial["status"] = "cancelled"
    assert _resource_signature(base, ("Patient",)) != _resource_signature(superficial, ("Patient",))


def test_pairwise_near_duplicate_metric_uses_normalized_components():
    rows = [
        ("Observation", ("lab",), ("final",), (("http://loinc.org", "718-7"),), ("Patient",)),
        ("Observation", ("lab",), ("cancelled",), (("http://loinc.org", "718-7"),), ("Patient",)),
        ("Condition", ("condition",), ("active",), (("http://snomed.info/sct", "38341003"),), ("Patient",)),
    ]
    metrics = _frequency_metrics(rows)
    assert metrics["pairCount"] == 3
    assert metrics["nearDuplicatePairCount"] == 1
    assert metrics["pairwiseNearDuplicateRate"] == pytest.approx(1 / 3, abs=1e-6)
    sequence_metrics = _frequency_metrics([(rows[0],), (rows[0], rows[2])], sequence=True)
    assert sequence_metrics["nearDuplicatePairCount"] == 1


def test_every_grammar_clause_is_executable_and_tag_independent():
    registered = {name for grammar in GRAMMARS.values() for clauses in grammar.values() for name in clauses}
    assert registered == set(PREDICATES)
    assert set(CLAUSE_CONTRACTS) == registered
    assert all({
        "clauseTypes", "archetypes", "eventResourceTypes", "minimumCardinality",
        "maximumCardinality", "allowedProfiles", "allowedCodeFamilies",
        "sourceOwnership", "temporalRelation", "lifecycleTransition",
        "branchProbability", "forbiddenCombinations", "evaluationInput",
    } <= set(contract) for contract in CLAUSE_CONTRACTS.values())
    rows = [{
        "resourceType": "Condition",
        "clinicalStatus": {"coding": [{"system": "x", "code": "resolved"}]},
        "abatementDateTime": "2025-01-02",
        "meta": {"tag": [{"system": "urn:hdp-test-bundle:synthetic-archetype", "code": "generally-healthy"}]},
        "_source": "hmo",
    }]
    patient = {"birthDate": "1980-01-01", "age": 45, "gender": "female"}
    assert PREDICATES["resolved-minor-condition"](rows, patient)
    rows[0].pop("abatementDateTime")
    assert not PREDICATES["resolved-minor-condition"](rows, patient)


def test_stale_snapshot_detector_uses_provenance_periods_and_competing_facts():
    current = {
        "resourceType": "Patient", "meta": {"lastUpdated": "2025-01-01T00:00:00Z"},
        "name": [{"family": "Cohen", "given": ["Rina"], "use": "official", "period": {"start": "2024-01-01"}}],
        "address": [{"line": ["New 1"], "city": "Haifa", "period": {"start": "2024-01-01"}}],
        "telecom": [{"value": "+972501111111", "period": {"start": "2024-01-01"}}],
    }
    stale = copy.deepcopy(current)
    stale["meta"]["lastUpdated"] = "2020-01-01T00:00:00Z"
    stale["name"][0]["period"]["start"] = "2019-01-01"
    stale["address"][0]["period"]["start"] = "2019-01-01"
    stale["telecom"][0]["period"]["start"] = "2019-01-01"
    stale["telecom"][0]["value"] = "+972509999999"
    assert stale_snapshot_evidence(stale, [current])["stale"]
    historical = copy.deepcopy(stale)
    historical["telecom"][0]["period"]["end"] = "2020-01-01"
    assert not stale_snapshot_evidence(historical, [current])["stale"]


def test_ucum_convertible_units_are_normalized_not_mislabeled():
    hemoglobin = {
        "resourceType": "Observation", "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": "718-7"}]},
        "valueQuantity": {"value": 120, "unit": "g/L", "code": "g/L"},
    }
    assert normalized_quantity(hemoglobin) == ("718-7", 12.0)
    assert not detects(hemoglobin, "unit-mismatch")


def _references_for_test(value):
    if isinstance(value, dict):
        if isinstance(value.get("reference"), str):
            yield value["reference"]
        for child in value.values():
            yield from _references_for_test(child)
    elif isinstance(value, list):
        for child in value:
            yield from _references_for_test(child)


def _dicts_for_test(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, dict):
                yield key, child
            yield from _dicts_for_test(child)
    elif isinstance(value, list):
        for child in value:
            yield from _dicts_for_test(child)


def test_active_source_model_has_no_decorative_fields(repository_root):
    model = read_json(repository_root / "data/source-models.v2.0.0.json")
    assert set(model) == {"catalogVersion", "utilizationAllocation", "sources"}
    assert all(
        set(row) == {"eventOffsetDays", "documentationLagMinutes", "condition", "observation"}
        for row in model["sources"].values()
    )
