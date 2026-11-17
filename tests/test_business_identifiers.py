import copy
from uuid import UUID

from synthetic_population.business_identifiers import (
    TEST_SYSTEM, SANDBOX_SYSTEM, PRACTITIONER_SYSTEM, bundle_errors,
    resource_errors, set_business_identifier,
)
from synthetic_population.bundle import _master_resources
from synthetic_population.config import load_spec
from synthetic_population.demographics import demographic_truth, patient_resource
from synthetic_population.ids import strong_identifier
from synthetic_population.safety import scan


def small_bundle(spec, source):
    locality = {"code": "3000", "nameHe": "ירושלים", "nameEn": "Jerusalem", "district": "Jerusalem"}
    entries = []
    for index in (1, 2, 3):
        patient, _, _ = patient_resource(spec, source, demographic_truth(spec, index, locality), index)
        entries.append({"fullUrl": f"{source}/Patient/{index}", "resource": patient})
    duplicate = copy.deepcopy(entries[-1]["resource"])
    duplicate["identifier"] = [strong_identifier(spec, source, "Patient", "a37dca1f-3e67-4d00-a64d-f9cd6630cd0e")]
    set_business_identifier(duplicate, spec, source)
    entries.append({"fullUrl": f"{source}/Patient/duplicate", "resource": duplicate})
    practitioner = next(r for _, r in _master_resources(spec, next(o for o in spec["organizations"] if o["id"] == source))
                        if r["resourceType"] == "Practitioner")
    entries.append({"fullUrl": f"{source}/Practitioner/1", "resource": practitioner})
    return {"resourceType": "Bundle", "type": "transaction", "entry": entries}


def test_native_business_identifiers_are_additional_unique_and_replayable(tiny_spec):
    bundles = {source: small_bundle(tiny_spec, source) for source in ("hmo", "hospital-a", "hospital-b")}
    assert bundle_errors(bundles, tiny_spec) == []
    assert bundles["hmo"] == small_bundle(tiny_spec, "hmo")
    for bundle in bundles.values():
        actors = [e["resource"] for e in bundle["entry"]]
        assert [actors[i]["identifier"][-1] for i in (0, 1)] == [
            {"system": TEST_SYSTEM, "value": "1"}, {"system": TEST_SYSTEM, "value": "2"}]
        for actor in actors[2:]:
            assert UUID(actor["identifier"][-1]["value"]).version == 4
        assert actors[2]["identifier"][-1]["system"] == SANDBOX_SYSTEM
        assert actors[4]["identifier"][-1]["system"] == PRACTITIONER_SYSTEM
        assert len(actors[3]["identifier"]) == 2  # no cohort key on an intentional duplicate


def test_rejects_missing_wrong_anchor_and_cross_actor_guid_collision(tiny_spec):
    bundle = small_bundle(tiny_spec, "hmo")
    a, b, ordinary, duplicate, practitioner = [e["resource"] for e in bundle["entry"]]
    a["identifier"][-1]["value"] = "certificator-1"
    assert resource_errors(a, "1") == ["anchor test identifier mismatch"]
    b["identifier"][-1]["value"] = "1"
    ordinary["identifier"][-1]["value"] = practitioner["identifier"][-1]["value"].upper()
    duplicate["identifier"].pop()
    errors = bundle_errors({"hmo": bundle}, tiny_spec)
    assert any("collision" in error for error in errors)
    assert any("exactly two" in error for error in errors)
    assert any("namespace" in error for error in errors)


def test_rejects_business_guid_equal_to_its_strong_id(tiny_spec):
    bundle = small_bundle(tiny_spec, "hmo")
    r = bundle["entry"][2]["resource"]
    r["identifier"][-1]["value"] = r["identifier"][0]["value"]
    assert any("overlaps" in error for error in bundle_errors({"hmo": bundle}, tiny_spec))


def test_updated_safety_keeps_production_identifier_protection(repository_root):
    spec = load_spec(repository_root / "data/specs/native.json")
    r = {"resourceType": "Patient", "identifier": [{"system": SANDBOX_SYSTEM, "value": "123456789"}]}
    errors = scan(r, spec)
    assert any("production-like" in error for error in errors)
    assert any("GUID" in error for error in errors)
