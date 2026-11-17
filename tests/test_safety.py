from synthetic_population.metadata import meta
from synthetic_population.safety import scan


def test_safety_scanner_rejects_production_identifiers_and_contacts(tiny_spec):
    patient = {
        "resourceType": "Patient",
        "meta": meta(tiny_spec, "hmo", "profile", "patient-demographics"),
        "identifier": [{"system": "http://fhir.health.gov.il/identifier/il-national-id", "value": "123456789"}],
        "telecom": [{"system": "email", "value": "person@example.com"}],
        "address": [{"line": ["Real Street"]}],
    }
    errors = scan(patient, tiny_spec)
    assert any("production identifier" in error for error in errors)
    assert any("email" in error for error in errors)
    assert any("visibly synthetic" in error for error in errors)


def test_safety_scanner_accepts_reserved_synthetic_truth(tiny_spec):
    patient = {
        "resourceType": "Patient",
        "meta": meta(tiny_spec, "hmo", "profile", "patient-demographics"),
        "identifier": [
            {"system": tiny_spec["identity"]["cohortIdentifierSystem"], "value": "cohort-000001"},
            {"system": tiny_spec["identity"]["testIdentifierSystem"], "value": "1"},
        ],
        "telecom": [{"system": "phone", "value": "+972-55-000001"}],
        "address": [{"line": ["Synthetic Street 1"]}],
    }
    assert scan(patient, tiny_spec) == []
