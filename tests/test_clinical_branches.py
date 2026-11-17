import synthetic_population
"""S14A branch contracts are reconstructed from FHIR, including adversarial edits."""
import copy
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from synthetic_population.care_pathway import build_patient_pathway
from synthetic_population.capture import project_capture
from synthetic_population.clinical_branch_predicates import branch_proof
from synthetic_population.config import load_spec
from synthetic_population.fhir_projection import project_pathway


ROOT = Path(synthetic_population.__file__).resolve().parent
SPEC = load_spec(ROOT / "data/specs/native.json")
FAMILIES = {
    "womens-health": ("age-eligible-screening", ("mammography", "cervical-screen", "result-review")),
    "perioperative-surgical": ("emergency-appendectomy", ("laparoscopic", "open", "short-stay")),
    "medication-intensive": ("dispense-follow-up", ("on-time-refill", "late-refill", "pharmacist-review")),
    "chronic-care": ("diabetes-stable", ("hba1c-stable", "diet-review", "retinal-referral")),
}


def fixture(archetype, variant, branch, birth_date="1980-01-01", renewal=None):
    for index in range(1, 120):
        pathway = build_patient_pathway(
            SPEC, index, archetype, variant, birth_date,
            SPEC["remediation"]["pathwayTemplateCatalog"],
            SPEC["remediation"]["clinicalContentCatalog"],
        )
        if pathway.secondary_branch != branch:
            continue
        if renewal is not None and any(f.family == "metformin-renewal" for f in pathway.facts) != renewal:
            continue
        captures, _ = project_capture(pathway, SPEC["remediation"]["capturePolicyCatalog"], SPEC["seed"])
        resources = []
        for source, dispositions in captures.items():
            projected, _ = project_pathway(SPEC, {"id": source}, index, "urn:uuid:synthetic-patient",
                                           pathway, dispositions)
            resources.extend({**row, "_source": source, "_fullUrl": url} for url, row in projected)
        return pathway, resources
    raise AssertionError(f"no deterministic fixture for {archetype}/{variant}/{branch}")


@pytest.mark.parametrize("archetype,variant,branch", [
    (archetype, variant, branch)
    for archetype, (variant, branches) in FAMILIES.items() for branch in branches
])
def test_branch_positive_and_swapped_graphs_fail(archetype, variant, branch):
    birth = "1965-01-01" if branch == "mammography" else "1980-01-01"
    _, resources = fixture(archetype, variant, branch, birth)
    assert branch_proof(branch, resources, birth), branch
    for other in FAMILIES[archetype][1]:
        if other != branch:
            assert not branch_proof(other, resources, birth), (branch, other)


@pytest.mark.parametrize("branch,field", [
    ("mammography", "age"), ("cervical-screen", "specimen"),
    ("result-review", "review"), ("laparoscopic", "procedure"),
    ("open", "approach"), ("short-stay", "discharge"),
    ("on-time-refill", "daysSupply"), ("late-refill", "gap"),
    ("pharmacist-review", "reconciliation"), ("hba1c-stable", "serial"),
    ("diet-review", "counselling"), ("retinal-referral", "referral"),
])
def test_branch_adversarial_fixture_fails(branch, field):
    archetype, (variant, _) = next((a, data) for a, data in FAMILIES.items() if branch in data[1])
    birth = "1965-01-01" if branch == "mammography" else "1980-01-01"
    _, positive = fixture(archetype, variant, branch, birth)
    resources = copy.deepcopy(positive)
    if field == "age":
        birth = "2005-01-01"
    elif field == "specimen":
        resources = [row for row in resources if row["resourceType"] != "Specimen"]
    elif field == "review":
        resources = [row for row in resources if row["resourceType"] != "Encounter"]
    elif field in {"procedure", "approach"}:
        for row in resources:
            if row["resourceType"] == "Procedure" and row.get("code", {}).get("coding", [{}])[0].get("code") in {"80146002", "174041007"}:
                if field == "procedure":
                    row["code"]["coding"][0]["code"] = "80146002"
                else:
                    row["code"].pop("text", None)
    elif field == "discharge":
        resources = [row for row in resources if row["resourceType"] != "DocumentReference"
                     or row.get("type", {}).get("coding", [{}])[0].get("code") != "373942005"]
    elif field == "daysSupply":
        for row in resources:
            if row["resourceType"] == "MedicationDispense": row.pop("daysSupply", None)
    elif field == "gap":
        dispenses = sorted((row for row in resources if row["resourceType"] == "MedicationDispense"),
                          key=lambda row: row["whenHandedOver"])
        for row in dispenses[1:]: row["whenHandedOver"] = dispenses[0]["whenHandedOver"]
    elif field == "reconciliation":
        resources = [row for row in resources if row["resourceType"] != "MedicationStatement"]
    elif field == "serial":
        hba1c = [row for row in resources if row["resourceType"] == "Observation"
                 and row.get("code", {}).get("coding", [{}])[0].get("code") == "4548-4"]
        for row in hba1c: row["valueQuantity"]["value"] = 6.0
    elif field == "counselling":
        for row in resources:
            if row["resourceType"] == "Encounter":
                for reason in row.get("reasonCode", []): reason.pop("text", None)
    else:
        resources = [row for row in resources if row["resourceType"] != "ServiceRequest"]
    assert not branch_proof(branch, resources, birth), field


def test_appendectomy_capture_tag_and_owner_actor_are_required():
    _, positive = fixture("perioperative-surgical", "emergency-appendectomy", "open")
    assert branch_proof("open", positive, "1980-01-01")
    performed = next(row for row in positive if row["resourceType"] == "Procedure"
                     and row.get("_source", "").startswith("hospital")
                     and row.get("code", {}).get("text") == "Open appendectomy"
                     and any(tag.get("code") == f"{row['_source']}-performed"
                             for tag in row.get("meta", {}).get("tag", [])))
    assert "category" not in performed
    for field in ("tag", "actor"):
        altered = copy.deepcopy(positive)
        procedure = next(row for row in altered if row.get("_fullUrl") == performed["_fullUrl"])
        if field == "tag":
            procedure["meta"]["tag"] = [row for row in procedure["meta"]["tag"]
                                          if row.get("system") != "urn:hdp-test-bundle:synthetic:capture-class"]
        else:
            procedure["performer"][0]["actor"]["identifier"]["system"] = "urn:wrong:source:practitioner"
        assert not branch_proof("open", altered, "1980-01-01"), field


def test_nonpregnant_followup_and_lisinopril_exposure_are_structural():
    pathway, resources = fixture("womens-health", "reproductive-follow-up", "nonpregnant-follow-up")
    assert branch_proof("nonpregnant-follow-up", resources, "1980-01-01")
    broken = copy.deepcopy(resources)
    for row in broken:
        if row["resourceType"] == "Observation": row["component"] = []
    assert not branch_proof("nonpregnant-follow-up", broken, "1980-01-01")
    for index in range(1, 30):
        exposure = build_patient_pathway(SPEC, index, "chronic-care", "medication-adjustment",
                                         "1980-01-01", SPEC["remediation"]["pathwayTemplateCatalog"],
                                         SPEC["remediation"]["clinicalContentCatalog"], "medication-exposure")
        captures, _ = project_capture(exposure, SPEC["remediation"]["capturePolicyCatalog"], SPEC["seed"])
        emitted = []
        for source, dispositions in captures.items():
            rows, _ = project_pathway(SPEC, {"id": source}, index, "urn:uuid:synthetic-patient",
                                      exposure, dispositions)
            emitted.extend({**row, "_fullUrl": url} for url, row in rows)
        if branch_proof("lisinopril-potassium-exposure", emitted, "1980-01-01"):
            assert not any(row.get("code", {}).get("coding", [{}])[0].get("code") == "125666000"
                           for row in emitted if row["resourceType"] == "Condition")
            emitted = [row for row in emitted if row["resourceType"] != "MedicationRequest"]
            assert not branch_proof("lisinopril-potassium-exposure", emitted, "1980-01-01")
            break
    else:
        raise AssertionError("no eligible active exposure fixture")


def test_screening_age_window_is_measured_at_order_and_ineligible_routing():
    from synthetic_population.pathways import care_pathway
    for born in ("1995-01-01", "1994-01-01"):
        for index in range(1, 20):
            pathway = build_patient_pathway(SPEC, index, "womens-health", "age-eligible-screening",
                                            born, SPEC["remediation"]["pathwayTemplateCatalog"],
                                            SPEC["remediation"]["clinicalContentCatalog"])
            captures, _ = project_capture(pathway, SPEC["remediation"]["capturePolicyCatalog"], SPEC["seed"])
            resources = [dict(row, _fullUrl=url) for source, dispositions in captures.items()
                         for url, row in project_pathway(SPEC, {"id": source}, index,
                                                         "urn:uuid:synthetic-patient", pathway, dispositions)[0]]
            assert branch_proof(pathway.secondary_branch, resources, born)
    with pytest.raises(ValueError):
        build_patient_pathway(SPEC, 1, "womens-health", "age-eligible-screening",
                              "2000-01-01", SPEC["remediation"]["pathwayTemplateCatalog"],
                              SPEC["remediation"]["clinicalContentCatalog"])
    for index in range(1, 120):
        routed = care_pathway(SPEC, index, "womens-health", birth_date="2000-01-01")
        assert routed["pathwayVariant"] != "age-eligible-screening"


def test_reviewed_screening_upper_birthday_reproducers_and_inclusive_edges():
    cervical_turns_55 = build_patient_pathway(
        SPEC, 6, "womens-health", "age-eligible-screening", "1969-10-23",
        SPEC["remediation"]["pathwayTemplateCatalog"], SPEC["remediation"]["clinicalContentCatalog"],
    )
    order = next(row for row in cervical_turns_55.events if row.role == "order")
    assert order.occurrence.start.date().isoformat() == "2024-10-30"
    assert cervical_turns_55.secondary_branch == "mammography"
    assert next(f.code for f in cervical_turns_55.facts if f.id == order.fact_id) == "71651007"
    with pytest.raises(ValueError, match="ineligible age"):
        build_patient_pathway(
            SPEC, 1, "womens-health", "age-eligible-screening", "1949-09-27",
            SPEC["remediation"]["pathwayTemplateCatalog"], SPEC["remediation"]["clinicalContentCatalog"],
        )
    for index, birth, expected_date, code in (
        (6, "1970-10-30", "2024-10-30", "82675-0"),
        (1, "1950-10-04", "2024-10-04", "71651007"),
    ):
        pathway = build_patient_pathway(
            SPEC, index, "womens-health", "age-eligible-screening", birth,
            SPEC["remediation"]["pathwayTemplateCatalog"], SPEC["remediation"]["clinicalContentCatalog"],
        )
        order = next(row for row in pathway.events if row.role == "order")
        assert order.occurrence.start.date().isoformat() == expected_date
        assert next(f.code for f in pathway.facts if f.id == order.fact_id) == code
    from synthetic_population.pathways import care_pathway
    earlier_eligible = [index for index in range(1, 120)
                        if care_pathway(SPEC, index, "womens-health", birth_date="1950-06-01")
                        ["pathwayVariant"] == "age-eligible-screening"]
    assert earlier_eligible, "anchor age 75 can still have an eligible earlier order"
    assert all(care_pathway(SPEC, index, "womens-health", birth_date="1948-01-01")
               ["pathwayVariant"] != "age-eligible-screening" for index in range(1, 120))


def test_diet_followup_order_supports_latest_hba1c_and_links_review():
    _, positive = fixture("chronic-care", "diabetes-stable", "diet-review")
    requests = [row for row in positive if row["resourceType"] == "ServiceRequest"
                and row.get("code", {}).get("coding", [{}])[0].get("code") == "390906007"]
    observations = {row["_fullUrl"] for row in positive if row["resourceType"] == "Observation"
                    and row.get("code", {}).get("coding", [{}])[0].get("code") == "4548-4"}
    assert requests and all(row.get("supportingInfo", [{}])[0].get("reference") in observations
                            for row in requests)
    for field in ("supportingInfo", "basedOn"):
        broken = copy.deepcopy(positive)
        kind = "ServiceRequest" if field == "supportingInfo" else "Encounter"
        for row in broken:
            if row["resourceType"] == kind and field in row:
                row.pop(field)
        assert not branch_proof("diet-review", broken, "1980-01-01"), field


def test_short_stay_review_requires_received_surgery_reference():
    _, positive = fixture("perioperative-surgical", "emergency-appendectomy", "short-stay")
    assert branch_proof("short-stay", positive, "1980-01-01")
    procedures = {row["_fullUrl"] for row in positive if row["resourceType"] == "Procedure"}
    assert all(not any(item.get("condition", {}).get("reference") in procedures
                       for item in row.get("diagnosis", []))
               for row in positive if row["resourceType"] == "Encounter")
    for replacement in ([], [{"type": "Condition", "display": "Discharge summary"}]):
        broken = copy.deepcopy(positive)
        for row in broken:
            if (row["resourceType"] == "Encounter" and row.get("_source") == "hmo"
                    and row.get("reasonReference") and row.get("period", {}).get("start", "") > "2025-06-14"):
                row["reasonReference"] = replacement
        assert not branch_proof("short-stay", broken, "1980-01-01")


def test_retinal_referral_requires_diabetes_and_latest_hba1c_links():
    _, positive = fixture("chronic-care", "diabetes-stable", "retinal-referral")
    for field in ("reasonReference", "supportingInfo"):
        broken = copy.deepcopy(positive)
        for row in broken:
            if row["resourceType"] == "ServiceRequest" and row.get("code", {}).get("text") == "Diabetic eye assessment":
                row.pop(field, None)
        assert not branch_proof("retinal-referral", broken, "1980-01-01"), field


@pytest.mark.parametrize("field", ["frequency", "dose", "strength", "tablet_count", "quantity", "supply"])
def test_refill_arithmetic_rejects_adversarial_changes(field):
    _, positive = fixture("medication-intensive", "dispense-follow-up", "on-time-refill")
    broken = copy.deepcopy(positive)
    for row in broken:
        if row.get("_source") != "hmo" or row["resourceType"] not in {"MedicationRequest", "MedicationDispense"}:
            continue
        if row.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("code") != "6809":
            continue
        if field == "frequency":
            row["dosageInstruction"][0]["timing"]["repeat"]["frequency"] = 2
        elif field == "dose":
            row["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"]["value"] = 1000
        elif field == "strength":
            row["medicationCodeableConcept"]["text"] = "Metformin 1000 mg per tablet"
        elif field == "tablet_count":
            row["dosageInstruction"][0]["doseAndRate"][1]["doseQuantity"]["value"] = 2
        elif field == "quantity" and row["resourceType"] == "MedicationDispense":
            row["quantity"]["value"] = 60
        elif field == "supply" and row["resourceType"] == "MedicationDispense":
            row["daysSupply"]["value"] = 60
    assert not branch_proof("on-time-refill", broken, "1980-01-01"), field


def test_refill_renewal_and_repeat_authorizations_are_distinct():
    for renewed in (True, False):
        _, positive = fixture("medication-intensive", "dispense-follow-up", "on-time-refill", renewal=renewed)
        assert branch_proof("on-time-refill", positive, "1980-01-01")
        broken = copy.deepcopy(positive)
        requests = sorted((row for row in broken if row.get("_source") == "hmo"
                           and row["resourceType"] == "MedicationRequest"
                           and row.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("code") == "6809"),
                          key=lambda row: row["authoredOn"])
        if renewed:
            requests[1].pop("priorPrescription")
            assert not branch_proof("on-time-refill", broken, "1980-01-01")
            for field in ("statusReason", "reasonReference"):
                broken = copy.deepcopy(positive)
                target = next(row for row in broken if row.get("_fullUrl") == requests[1]["_fullUrl"])
                target.pop(field)
                assert not branch_proof("on-time-refill", broken, "1980-01-01"), field
            broken = copy.deepcopy(positive)
            target = next(row for row in broken if row.get("_fullUrl") == requests[0]["_fullUrl"])
            target["status"] = "active"
            assert not branch_proof("on-time-refill", broken, "1980-01-01")
        else:
            requests[0]["dispenseRequest"]["numberOfRepeatsAllowed"] = 0
            assert not branch_proof("on-time-refill", broken, "1980-01-01")


@pytest.mark.parametrize("branch,birth", [("mammography", "1965-01-01"), ("cervical-screen", "1980-01-01")])
def test_recent_prior_screen_and_repeat_after_review_are_rejected(branch, birth):
    _, resources = fixture("womens-health", "age-eligible-screening", branch, birth)
    prior = copy.deepcopy(next(row for row in resources if row["resourceType"] == "Observation"
                               and row.get("code", {}).get("coding", [{}])[0].get("code") in {"24606-6", "82675-0"}))
    order = next(row for row in resources if row["resourceType"] == "ServiceRequest"
                 and row.get("code", {}).get("coding", [{}])[0].get("code") in {"71651007", "82675-0"})
    when = datetime.fromisoformat(order["authoredOn"].replace("Z", "+00:00")) - timedelta(days=30)
    prior["effectiveDateTime"] = when.isoformat().replace("+00:00", "Z")
    prior["_fullUrl"] = "urn:uuid:prior-screen"
    assert not branch_proof(branch, resources + [prior], birth)
    _, review_resources = fixture("womens-health", "age-eligible-screening", "result-review")
    extra_order = copy.deepcopy(next(row for row in review_resources if row["resourceType"] == "ServiceRequest"
                                     and row.get("code", {}).get("coding", [{}])[0].get("code") in {"71651007", "82675-0"}))
    result = next(row for row in review_resources if row["resourceType"] == "Observation"
                  and row.get("code", {}).get("coding", [{}])[0].get("code") in {"24606-6", "82675-0"})
    after = datetime.fromisoformat(result["effectiveDateTime"].replace("Z", "+00:00")) + timedelta(days=1)
    extra_order["authoredOn"] = after.isoformat().replace("+00:00", "Z")
    assert not branch_proof("result-review", review_resources + [extra_order], "1980-01-01")


def test_hmo_summary_cannot_prove_hospital_surgery_or_replace_open_review():
    _, resources = fixture("perioperative-surgical", "emergency-appendectomy", "open")
    assert branch_proof("open", resources, "1980-01-01")
    assert not branch_proof("open", [row for row in resources if row.get("_source") == "hmo"], "1980-01-01")
    no_review = [row for row in resources if row.get("resourceType") != "Encounter"
                 or row.get("_source") != "hmo"
                 or row.get("class", {}).get("code") != "AMB"
                 or row.get("period", {}).get("start", "") <= next(
                     item["date"] for item in resources if item.get("resourceType") == "DocumentReference"
                     and item.get("_source") == "hmo")]
    assert not branch_proof("open", no_review, "1980-01-01")
