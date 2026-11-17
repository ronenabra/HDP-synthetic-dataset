from copy import deepcopy
import json

import pytest

from synthetic_population.release_repairs import (
    _bundle_identity, _case, apply_field_edits, release_plan,
)
from synthetic_population.utils import canonical_bytes


def test_release_contract_has_exact_published_bundle_identities(repository_root):
    plan = release_plan(repository_root)
    expected = {
        "1000": ["1ecbbc2bb2892147e310795e2f987a846b5696eaacb4c2033a860c0e1a78fe18", "4cc751b576cd54a9ac13021053c611c792efdd7467e0106da43a9121d15f3ec4", "305a70b9b2279497c85ec7ea92143ab9837eb62c92e0a6904e52af4879561c5f"],
        "10000": ["b510540915bbc8b697ec4660eaa0e9d8915544091c082f098e6914e59c086ea0", "31ba6839a5e2c9db909d2bb2def078b12046b5c042e76644599425ba4baf7ea8", "c9a787d47bfad48ad57a7abe30b797cd9111636774394bc2b5eb618a701c0bbb"],
    }
    for population, case in plan["populations"].items():
        assert [row["sha256"] for row in case["bundles"]] == expected[population]
        assert case["nativeSpec"]["populationSize"] == int(population)
        assert all(key not in case["nativeSpec"] for key in ("derivation", "validationPrerequisites", "clinicalEvidence"))


def test_release_rejects_unplanned_population_size(repository_root):
    with pytest.raises(ValueError, match="exactly 1,000 or 10,000"):
        _case(5000, repository_root)


def test_bounded_serializer_matches_release_json_format(tmp_path):
    import hashlib
    bundle = {"resourceType": "Bundle", "type": "transaction", "entry": [{"fullUrl": "urn:uuid:example", "resource": {"resourceType": "Observation", "text": "Synthetic ירושלים", "valueQuantity": {"value": 46.4}}}]}
    expected = canonical_bytes(bundle)
    path = tmp_path / "bundle.json"
    assert _bundle_identity(bundle, path) == (hashlib.sha256(expected).hexdigest(), len(expected))
    assert path.read_bytes() == expected


def test_field_repairs_preserve_security_and_apply_exact_unknown_dose():
    resource = {"resourceType": "MedicationRequest", "dosageInstruction": [{"text": "old dose"}], "text": {"div": "old narrative"}, "meta": {"security": [{"system": "test", "code": "HTEST"}]}}
    security = deepcopy(resource["meta"])
    apply_field_edits(resource, [{"op": "remove", "path": "/dosageInstruction"}, {"op": "replace", "path": "/text/div", "value": "Pediatric dose not captured; weight required."}])
    assert "dosageInstruction" not in resource
    assert resource["meta"] == security
    assert resource["text"]["div"] == "Pediatric dose not captured; weight required."


def test_repairs_fail_on_missing_fields():
    with pytest.raises(ValueError, match="replacement field is missing"):
        apply_field_edits({"resourceType": "Observation"}, [{"op": "replace", "path": "/missing", "value": 1}])


def test_modified_plan_is_rejected_before_loading_native_generator(tmp_path, repository_root):
    path = tmp_path / "data/accepted-release.json"
    path.parent.mkdir()
    value = json.loads((repository_root / "data/accepted-release.json").read_text())
    value["populations"]["1000"]["nativeSpec"]["seed"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="plan has changed"):
        release_plan(tmp_path)


def test_native_generator_inputs_are_bundled_without_a_private_repository_dependency(repository_root):
    context = json.loads((repository_root / "data/reference-context.json").read_text())
    source = context["nativeGenerator"]
    assert source["root"] == "native_generator"
    assert "repository" not in source
    assert "url" not in context["coverageSeed"]
    assert context["coverageSeed"]["source"]["path"].startswith("native_generator/")
    assert all((repository_root / source["root"] / row["path"]).is_file() for row in source["files"])


def test_generic_seed_restores_exact_current_identifiers_from_bundled_input(repository_root):
    import hashlib
    from synthetic_population.rehydrate import _coverage_seed
    context = json.loads((repository_root / "data/reference-context.json").read_text())
    seed = _coverage_seed(repository_root, context["coverageSeed"])
    assert len(seed) == 209258
    assert hashlib.sha256(seed).hexdigest() == "625eead8e3dac1b5d29f29a501984e127c51bdaf980afc3e291e720ad5344a04"


def test_projection_isolates_resource_objects_before_field_repairs():
    from synthetic_population.release_repairs import _project_bundle
    shared = {"coding": [{"system": "http://snomed.info/sct", "code": "123"}]}
    bundle = {"entry": [{"resource": {"resourceType": "Condition", "code": shared}}, {"resource": {"resourceType": "Condition", "code": shared}}]}
    _project_bundle(bundle, {}, "hmo")
    apply_field_edits(bundle["entry"][0]["resource"], [{"op": "replace", "path": "/code/coding", "value": [{"system": "http://snomed.info/sct", "code": "456"}]}])
    assert bundle["entry"][1]["resource"]["code"]["coding"][0]["code"] == "123"
