from copy import deepcopy
from pathlib import Path

from synthetic_population.bundle import build_population, _apply_matched_control, _control_medication
from synthetic_population.config import scale_spec
from synthetic_population.demographics import demographic_truth
from synthetic_population.lifecycle import build_lifecycle_audit
from synthetic_population.metrics import build_quality_metrics, metric_failures


def _fixture(repository_root: Path, lifecycle_spec):
    spec = scale_spec(lifecycle_spec,
                      30, "4.1.61-s17-focused30", zero_problems=True)
    built = build_population(spec, repository_root)
    return spec, built


def test_death_truth_and_native_target(repository_root, lifecycle_spec):
    spec, built = _fixture(repository_root, lifecycle_spec)
    audit = build_lifecycle_audit(built["bundles"], spec)
    assert audit["status"] == "pass"
    assert (audit["denominator"], audit["sourcePatientRecordCount"], audit["targetDeceased"], audit["observedDeceased"]) == (30, 90, 1, 1)
    assert audit["clinicalOccurrencesChecked"] > 0
    assert not metric_failures(build_quality_metrics(built["bundles"], built["crosswalk"], spec), spec)


def test_cross_source_death_drift_and_postdeath_event_are_detected(repository_root, lifecycle_spec):
    spec, built = _fixture(repository_root, lifecycle_spec)
    deceased = build_lifecycle_audit(built["bundles"], spec)["examples"]["deceased"][0]
    patient_url = next(row["sources"]["hospital-a"]["patientFullUrl"]
                       for row in built["crosswalk"]["patients"] if row["cohortId"] == deceased)
    drift = deepcopy(built["bundles"])
    patient = next(entry["resource"] for entry in drift["hospital-a"]["entry"] if entry["fullUrl"] == patient_url)
    del patient["deceasedDateTime"]
    assert any("source lifecycle truth disagrees" in error for error in build_lifecycle_audit(drift, spec)["failures"])

    late = deepcopy(built["bundles"])
    clinical = next(entry["resource"] for entry in late["hmo"]["entry"]
                    if entry["resource"].get("resourceType") == "Encounter"
                    and entry["resource"].get("subject", {}).get("reference") == next(
                        row["sources"]["hmo"]["patientFullUrl"] for row in built["crosswalk"]["patients"]
                        if row["cohortId"] == deceased))
    clinical["period"] = {"start": "2026-01-02T08:00:00Z", "end": "2026-01-02T09:00:00Z"}
    assert any("outside life interval" in error for error in build_lifecycle_audit(late, spec)["failures"])

    early = deepcopy(built["bundles"])
    clinical = next(entry["resource"] for entry in early["hmo"]["entry"]
                    if entry["resource"].get("resourceType") == "Encounter"
                    and entry["resource"].get("subject", {}).get("reference") == next(
                        row["sources"]["hmo"]["patientFullUrl"] for row in built["crosswalk"]["patients"]
                        if row["cohortId"] == deceased))
    clinical["period"] = {"start": "1900-01-01T08:00:00Z", "end": "1900-01-01T09:00:00Z"}
    assert any("outside life interval" in error for error in build_lifecycle_audit(early, spec)["failures"])


def test_administrative_recording_after_death_is_allowed(repository_root, lifecycle_spec):
    spec, built = _fixture(repository_root, lifecycle_spec)
    deceased = build_lifecycle_audit(built["bundles"], spec)["examples"]["deceased"][0]
    patient_url = next(row["sources"]["hmo"]["patientFullUrl"]
                       for row in built["crosswalk"]["patients"] if row["cohortId"] == deceased)
    changed = deepcopy(built["bundles"])
    changed["hmo"]["entry"].append({
        "fullUrl": "urn:uuid:administrative-condition-s17",
        "resource": {"resourceType": "Condition", "subject": {"reference": patient_url},
                     "recordedDate": "2026-01-03"},
        "request": {"method": "POST", "url": "Condition"},
    })
    assert build_lifecycle_audit(changed, spec)["status"] == "pass"


def test_allergy_recorded_after_death_without_onset_is_administrative(repository_root, lifecycle_spec):
    spec, built = _fixture(repository_root, lifecycle_spec)
    baseline = build_lifecycle_audit(built["bundles"], spec)
    assert baseline["status"] == "pass"
    deceased = baseline["examples"]["deceased"][0]
    patient_url = next(row["sources"]["hmo"]["patientFullUrl"]
                       for row in built["crosswalk"]["patients"] if row["cohortId"] == deceased)
    changed = deepcopy(built["bundles"])
    allergy = {"resourceType": "AllergyIntolerance", "patient": {"reference": patient_url},
               "recordedDate": "2026-01-03"}
    changed["hmo"]["entry"].append({
        "fullUrl": "urn:uuid:administrative-allergy-s17",
        "resource": allergy,
        "request": {"method": "POST", "url": "AllergyIntolerance"},
    })
    administrative = build_lifecycle_audit(changed, spec)
    assert administrative["status"] == "pass"
    assert administrative["clinicalOccurrencesChecked"] == baseline["clinicalOccurrencesChecked"]

    allergy["onsetDateTime"] = "2026-01-02T08:00:00Z"
    clinical = build_lifecycle_audit(changed, spec)
    assert clinical["status"] == "fail"
    assert f"{deceased}: AllergyIntolerance occurrence outside life interval" in clinical["failures"]


def test_second_block_infant_controls_start_after_birth(repository_root, lifecycle_spec):
    spec = scale_spec(lifecycle_spec,
                      2000, "4.1.61-s17-control-boundaries")
    locality = {"code": "3000", "nameHe": "ירושלים", "nameEn": "Jerusalem", "district": "Jerusalem"}
    allergy_birth = demographic_truth(spec, 1004, locality)["birthDate"]
    rows = []
    _apply_matched_control(spec, 1004, "allergy-treatment-risk", "hmo", "urn:uuid:patient", rows, 0, allergy_birth)
    allergy = next(resource for _, resource in rows if resource["resourceType"] == "AllergyIntolerance")
    assert allergy["onsetDateTime"] >= allergy_birth
    assert allergy["recordedDate"] >= allergy["onsetDateTime"]

    medication_birth = demographic_truth(spec, 1008, locality)["birthDate"]
    _, medication = _control_medication(spec, "hospital-b", 1008, "urn:uuid:patient",
                                        "allergy-treatment-risk", "active", medication_birth)
    assert medication["authoredOn"] >= medication_birth
