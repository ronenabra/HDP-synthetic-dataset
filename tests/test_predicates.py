import pytest

from synthetic_population.predicates import evaluate, evaluate_variant


@pytest.mark.parametrize(
    ("name", "facts"),
    [
        ("renal-medication-risk", [
            {"code": {"coding": [{"system": "http://snomed.info/sct", "code": "431857002"}]}},
            {"resourceType": "Observation", "code": {"coding": [{"system": "http://loinc.org", "code": "6298-4"}]}, "valueQuantity": {"value": 5.3}},
            {"medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "9997"}]}},
        ]),
        ("bleeding-risk", [
            {"medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "11289"}]}},
            {"resourceType": "Observation", "code": {"coding": [{"system": "http://loinc.org", "code": "718-7"}]}, "valueQuantity": {"value": 11.2}},
            {"code": {"coding": [{"system": "http://snomed.info/sct", "code": "13200003"}]}},
        ]),
        ("allergy-treatment-risk", [
            {"code": {"coding": [{"system": "http://snomed.info/sct", "code": "91936005"}]}},
            {"code": {"coding": [{"system": "http://loinc.org", "code": "630-4"}]}},
            {"medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "723"}]}},
        ]),
    ],
)
def test_cross_source_predicate_requires_union(name, facts):
    assert all(not evaluate(name, [fact]) for fact in facts)
    assert evaluate(name, facts)
    assert not evaluate(name, [])


def test_falling_hemoglobin_uses_a_coherent_pair_among_historical_results():
    def hemoglobin(day, value):
        return {
            "resourceType": "Observation", "status": "final",
            "code": {"coding": [{"system": "http://loinc.org", "code": "718-7"}]},
            "effectiveDateTime": day,
            "valueQuantity": {"value": value, "code": "g/dL"},
            "interpretation": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "L"}]}],
        }

    medication = {
        "resourceType": "MedicationRequest", "status": "active", "authoredOn": "2025-06-11",
        "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "11289"}]},
        "dosageInstruction": [{
            "timing": {"repeat": {"frequency": 1, "period": 1, "periodUnit": "d"}},
            "route": {"coding": [{"code": "26643006"}]},
            "doseAndRate": [{"doseQuantity": {"value": 5, "code": "mg"}}],
        }],
        "courseOfTherapyType": {"coding": [{"code": "acute"}]},
    }
    bleed = {
        "resourceType": "Condition", "recordedDate": "2025-06-16",
        "code": {"coding": [{"system": "http://snomed.info/sct", "code": "74474003"}]},
        "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
    }
    facts = [medication, bleed, hemoglobin("2025-01-02", 13.23),
             hemoglobin("2025-05-19", 13.1), hemoglobin("2025-06-18", 11.2)]
    assert evaluate_variant("bleeding-risk", "falling-hemoglobin", facts, require_three_source_context=True)
    facts[-1] = hemoglobin("2025-06-18", 13.2)
    assert not evaluate_variant("bleeding-risk", "falling-hemoglobin", facts, require_three_source_context=True)
