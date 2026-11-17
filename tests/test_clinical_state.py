import synthetic_population
"""Minimal synthetic regressions tracing F02/F03/F05/F06/F07 (4.1.63 review)."""
import copy
from pathlib import Path

import pytest

from synthetic_population.clinical_state import (
    acute_snapshot, body_measurements, egfr_2021, stage4_creatinine,
    state_findings, under_18_months, population_state_audit,
)
from synthetic_population.config import load_spec, scale_spec
from synthetic_population.bundle import build_population

ROOT = Path(synthetic_population.__file__).resolve().parent


def observation(code, value, when="2025-01-01", unit="mg/dL"):
    return {"resourceType": "Observation", "status": "final", "code": {"coding": [{"code": code}]},
            "effectiveDateTime": when, "specimen": {"identifier": {"value": "sample-" + when}},
            "valueQuantity": {"value": value, "code": unit}}


def condition(code, when="2025-01-01"):
    return {"resourceType": "Condition", "code": {"coding": [{"code": code}]},
            "clinicalStatus": {"coding": [{"code": "active"}]}, "onsetDateTime": when}


def findings(rows, birth="1966-10-11", sex="male"):
    return {f["finding"] for f in state_findings(rows, birth, sex, "2026-01-01")}


def test_f02_longitudinal_stage4_contract_and_controls():
    renal = condition("431857002")
    assert "F02" in findings([renal, observation("2160-0", .8)])
    rows = [renal, observation("2160-0", stage4_creatinine(58, "male", 23), "2025-01-01"),
            observation("2160-0", stage4_creatinine(58, "male", 24), "2025-05-01")]
    assert not findings(rows)
    for sex in ("male", "female"):
        for age in (18, 40, 80):
            value = stage4_creatinine(age, sex, 25)
            assert egfr_2021(value, age, sex) == pytest.approx(25, abs=.2)
    with pytest.raises(ValueError):
        stage4_creatinine(17, "male", 25)
    rows[-1]["effectiveDateTime"] = "2025-01-02"
    assert "F02" in findings(rows)


def hiv(code, specimen, when="2025-11-10"):
    r = observation(code, 1, when)
    r.pop("valueQuantity")
    r["valueCodeableConcept"] = {"coding": [{"code": "260373001"}]}
    r["specimen"] = {"identifier": {"value": specimen}}
    return r


def test_f03_independent_virology_and_calendar_boundary():
    assert under_18_months("2024-02-29", "2025-08-28")
    assert not under_18_months("2024-02-29", "2025-08-29")
    birth = "2025-10-29"
    rows = [hiv("68961-2", "a"), hiv("68961-2", "a")]
    assert "F03" in findings(rows, birth)
    rows = [hiv("5018-7", "a"), hiv("5018-7", "b", "2025-11-11")]
    assert not findings(rows, birth)
    rows[1]["derivedFrom"] = [{"reference": "first-sample"}]
    assert "F03" in findings(rows, birth)
    rows[1].pop("derivedFrom")
    rows[1]["specimen"] = rows[0]["specimen"]
    assert "F03" in findings(rows, birth)


def test_f05_f07_snapshot_and_life_intervals():
    past = condition("74400008", "2024-12-04")
    assert "F05" in findings([past])
    acute_snapshot(past, "2026-01-01")
    assert not findings([past])
    assert past["note"] and past["abatementDateTime"].startswith("2024-12-11")
    future = condition("195662009", "2025-12-30")
    future["clinicalStatus"]["coding"][0]["code"] = "resolved"
    future["abatementDateTime"] = "2026-01-06"
    assert "F07" in findings([future])
    acute_snapshot(future, "2026-01-01")
    assert future["clinicalStatus"]["coding"][0]["code"] == "active"
    assert "abatementDateTime" not in future
    assert not findings([future, {"resourceType": "ServiceRequest", "occurrenceDateTime": "2026-02-01"}])
    past["abatementDateTime"] = "2024-12-01"
    assert "F07" in findings([past])


@pytest.mark.parametrize("age", [0, .02, 1, 5, 17.99, 18, 60])
def test_f06_standalone_growth_ordinary_distribution(age):
    height, weight = body_measurements(age)
    assert height > 0 and weight > 0
    if age >= 18:
        assert 150 <= height <= 190 and 40 <= weight <= 110


def test_f06_exact_adult_weight_and_neonatal_fvc():
    assert "F06" in findings([observation("29463-7", 2.69, "2025-09-07", "kg")])
    assert "F06" in findings([observation("19868-9", 3.59, "2025-11-05", "L")], "2025-10-29")
    assert not findings([observation("29463-7", 70000, unit="g")])
    assert not findings([observation("8302-2", 1.7, unit="m")])


def test_bounded_generation_and_native_audit():
    spec = scale_spec(load_spec(ROOT / "data/specs/native.json"), 30, "4.2.0-s20-test")
    built = build_population(spec, ROOT)
    audit = population_state_audit(built["bundles"], spec)
    assert audit["status"] == "pass", audit["findings"]
    assert audit["eventAgeMeasurements"]


def test_f02_measured_gfr_exception_needs_linked_independent_longitudinal_samples():
    renal = condition("431857002")
    rows = [renal, observation("2160-0", .8)]
    renal["evidence"] = [{"detail": [{"reference": "gfr-a"}, {"reference": "gfr-b"}]}]
    for key, when in (("gfr-a", "2024-09-01"), ("gfr-b", "2025-01-01")):
        row = observation("measured-gfr", 24, when, "mL/min/{1.73_m2}")
        row.update(_fullUrl=key, method={"text": "measured GFR: iohexol clearance"},
                   specimen={"identifier": {"value": key}})
        rows.append(row)
    assert not findings(rows)
    rows[-1]["specimen"] = rows[-2]["specimen"]
    assert "F02" in findings(rows)


def test_f06_exception_requires_resolving_clinical_assessment():
    height = observation("8302-2", 55, unit="cm")
    height["derivedFrom"] = [{"reference": "assessment"}]
    assert "F06" in findings([height])
    assessment = {"resourceType": "Observation", "_fullUrl": "assessment", "status": "final",
                  "code": {"coding": [{"code": "anthropometry-assessment"}]},
                  "valueString": "Known severe skeletal dysplasia; length confirmed on repeat assessment."}
    assert "F06" in findings([height, assessment])
    height["subject"] = {"reference": "patient"}
    assessment.update(subject=height["subject"], effectiveDateTime="2025-01-01",
                      valueCodeableConcept={"coding": [{"code": "skeletal-dysplasia"}]},
                      derivedFrom=[{"reference": "diagnosis"}])
    diagnosis = condition("skeletal-dysplasia")
    diagnosis.update(_fullUrl="diagnosis", subject=height["subject"],
                     verificationStatus={"coding": [{"code": "confirmed"}]})
    assert not findings([height, assessment, diagnosis])


def test_reviewer_rejects_nonfinal_future_derived_and_copied_renal_evidence():
    renal = condition("431857002")
    positive = [renal, observation("2160-0", 2.8, "2025-01-01"), observation("2160-0", 2.8, "2025-05-01")]
    assert not findings(positive)
    for field, value in (("status", "preliminary"), ("effectiveDateTime", "2026-05-01"),
                         ("derivedFrom", [{"reference": "first"}]),
                         ("specimen", positive[1]["specimen"])):
        rows = copy.deepcopy(positive)
        rows[-1][field] = value
        assert "F02" in findings(rows), field


def test_reviewer_rejects_unconfirmed_infant_evidence_and_future_conditions():
    diagnosis = condition("86406008", "2025-11-12")
    diagnosis["verificationStatus"] = {"coding": [{"code": "confirmed"}]}
    assert "F03" in findings([diagnosis], "2025-10-29")
    rows = [hiv("5018-7", "a"), hiv("5018-7", "b", "2025-11-11")]
    rows[-1]["status"] = "preliminary"
    assert "F03" in findings(rows, "2025-10-29")
    future = condition("74400008", "2026-05-01")
    assert "F07" in findings([future])
    future["onsetDateTime"] = "2025-05-01"
    future["clinicalStatus"]["coding"][0]["code"] = "resolved"
    assert "F07" not in findings([future])


def test_reviewer_reference_metadata_period_cancelled_and_absent_hiv():
    weight = observation("29463-7", 70, unit="kg")
    weight["referenceRange"] = [{"low": {"value": 3}, "high": {"value": 120}}]
    assert "F06" in findings([weight])
    weight.pop("referenceRange")
    weight["status"] = "cancelled"
    assert "F06" in findings([weight])
    weight["status"] = "final"
    weight["valueQuantity"]["value"] = 2.69
    weight["effectivePeriod"] = {"start": weight.pop("effectiveDateTime")}
    assert "F06" in findings([weight])
    assert any(f["finding"] == "F03" for f in state_findings([], "2025-10-29", "male", "2026-01-01", required_infant_hiv=True))


@pytest.mark.parametrize("field,period", [
    ("abatementPeriod", {"start": "2026-02-01"}),
    ("abatementPeriod", {"start": "2026-02-01", "end": "2025-12-01"}),
    ("onsetPeriod", {"start": "2025-01-01", "end": "2026-02-01"}),
])
def test_reviewer_period_endpoints(field, period):
    row = condition("74400008")
    row["clinicalStatus"]["coding"][0]["code"] = "resolved"
    row.pop("onsetDateTime", None)
    row[field] = period
    assert "F07" in findings([row])


def test_reviewer_hiv_test_age_and_period_population_summary():
    spec = load_spec(ROOT / "data/specs/native.json")
    patient = {"resourceType": "Patient", "birthDate": "2024-06-15", "gender": "male",
               "identifier": [{"system": spec["identity"]["cohortIdentifierSystem"], "value": "cohort-000551"}]}
    bundle = {"hmo": {"entry": [{"fullUrl": "patient", "resource": patient}]}}
    assert any(f["finding"] == "F03" for f in population_state_audit(bundle, spec)["findings"])
    patient["birthDate"] = "1966-10-11"
    patient["identifier"][0]["value"] = "cohort-000001"
    weight = observation("29463-7", 70, unit="kg")
    weight.update(subject={"reference": "patient"}, effectivePeriod={"start": weight.pop("effectiveDateTime")})
    bundle["hmo"]["entry"].append({"fullUrl": "weight", "resource": weight})
    assert population_state_audit(bundle, spec)["eventAgeMeasurements"]["29463-7:adult"]["facts"] == 1
