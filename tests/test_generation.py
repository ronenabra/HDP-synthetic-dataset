import json

import pytest

from synthetic_population.artifacts import source_bundle_path
from synthetic_population.config import scale_spec
from synthetic_population.manifest import generate
from synthetic_population.manifest import refresh_checksums
from synthetic_population.projections import challenge, mobility
from synthetic_population.utils import read_json
from synthetic_population.verify import _empty_array_paths, verify


def _target(tmp_path, name):
    return tmp_path / "generated" / f"synthetic-population.{name}"


def test_tiny_generation_verifies(tmp_path, tiny_spec, repository_root):
    out = generate(tiny_spec, _target(tmp_path, "tiny"), repository_root)
    result = verify(out)
    assert result["ok"], result["errors"]
    assert result["resourceCount"] > 120
    assert result["taggedClinicalRate"] <= result["taggedClinicalRateLimit"]
    assert result["taggedClinicalRateLimit"] >= tiny_spec["problems"]["maxTaggedClinicalRate"]


def test_empty_fhir_arrays_are_rejected_by_native_shape_gate():
    assert list(_empty_array_paths({"resourceType": "Patient", "telecom": []})) == ["telecom"]
    assert list(_empty_array_paths({"resourceType": "Patient", "name": [{"given": []}]})) == ["name[0].given"]
    assert list(_empty_array_paths({"resourceType": "Patient", "active": True})) == []


def test_generation_is_byte_deterministic(tmp_path, tiny_spec, repository_root):
    one = generate(tiny_spec, _target(tmp_path, "one"), repository_root)
    two = generate(tiny_spec, _target(tmp_path, "two"), repository_root)
    assert (one / "checksums.sha256").read_bytes() == (two / "checksums.sha256").read_bytes()


def test_nonempty_target_requires_force(tmp_path, tiny_spec, repository_root):
    out = generate(tiny_spec, _target(tmp_path, "force"), repository_root)
    with pytest.raises(FileExistsError):
        generate(tiny_spec, out, repository_root)
    assert generate(tiny_spec, out, repository_root, force=True) == out


def test_checksum_tampering_is_detected(tmp_path, tiny_spec, repository_root):
    out = generate(tiny_spec, _target(tmp_path, "tamper"), repository_root)
    path = out / "dataset-manifest.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    result = verify(out)
    assert not result["ok"]
    assert any("checksum mismatch" in item for item in result["errors"])


def test_reference_tampering_is_detected_after_rechecksumming(tmp_path, tiny_spec, repository_root):
    out = generate(tiny_spec, _target(tmp_path, "reference-tamper"), repository_root)
    path = source_bundle_path(out, "hmo")
    bundle = read_json(path)
    condition = next(entry["resource"] for entry in bundle["entry"] if entry["resource"]["resourceType"] == "Condition")
    condition["subject"]["reference"] = "urn:uuid:00000000-0000-4000-8000-000000000000"
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    refresh_checksums(out)
    result = verify(out)
    assert not result["ok"]
    assert any("unresolved urn:uuid" in item for item in result["errors"])


def test_answer_key_tampering_is_detected_after_rechecksumming(tmp_path, tiny_spec, repository_root):
    out = generate(tiny_spec, _target(tmp_path, "answer-tamper"), repository_root)
    path = out / "problem-answer-key.json"
    answer = read_json(path)
    answer["problems"].pop()
    answer["count"] -= 1
    path.write_text(json.dumps(answer, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    refresh_checksums(out)
    result = verify(out)
    assert not result["ok"]
    assert any("problem tag/key mismatch" in item or "problem allocation mismatch" in item for item in result["errors"])


def test_mobility_always_excludes_v(tmp_path, native_spec, repository_root):
    tiny_spec = scale_spec(native_spec, 200, "4.0.0-mobility", zero_problems=True)
    out = generate(tiny_spec, _target(tmp_path, "mobility"), repository_root)
    destination = tmp_path / "mobility.json"
    mobility(out, destination, allow_restricted=True, authorized_sensitivity=("MH", "HIV", "SDV"))
    projection = read_json(destination)
    assert projection["_projectionEvidence"]["excluded"]["V"] > 0
    assert all(item.get("code") != "V" for entry in projection["entry"] for item in entry["resource"].get("meta", {}).get("security", []))
    urls = {entry["fullUrl"] for entry in projection["entry"]}
    assert all(
        not (value.startswith("urn:uuid:") and value not in urls)
        for entry in projection["entry"]
        for value in _references(entry["resource"])
    )


def test_restricted_projection_requires_explicit_domain_authority(tmp_path, tiny_spec, repository_root):
    out = generate(tiny_spec, _target(tmp_path, "restricted"), repository_root)
    with pytest.raises(ValueError, match="explicit authorized sensitivity"):
        mobility(out, tmp_path / "invalid-mobility.json", allow_restricted=True)


def _references(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "reference" and isinstance(item, str):
                yield item
            else:
                yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


def _identifiers(resource):
    value = resource.get("identifier", [])
    if isinstance(value, dict):
        return [value]
    return value if isinstance(value, list) else []


def test_challenge_removes_all_project_truth_markers(tmp_path, tiny_spec, repository_root):
    out = generate(tiny_spec, _target(tmp_path, "challenge"), repository_root)
    destination = tmp_path / "challenge.json"
    challenge(out, destination)
    data = read_json(destination)
    assert all(
        not tag.get("system", "").startswith("urn:hdp-test-bundle:synthetic")
        for entry in data["entry"]
        for tag in entry["resource"].get("meta", {}).get("tag", [])
    )
    assert all(
        identifier.get("system") not in {
            tiny_spec["identity"]["cohortIdentifierSystem"],
            "urn:hdp-test-bundle:synthetic:linkage-key",
        }
        for entry in data["entry"]
        for identifier in _identifiers(entry["resource"])
    )
    assert all(any(item.get("code") == "HTEST" for item in entry["resource"].get("meta", {}).get("security", [])) for entry in data["entry"])


def test_zero_problem_mode_is_clean_and_explicit(tmp_path, tiny_spec, repository_root):
    tiny_spec["problems"].update({"patientsWithOne": 0, "patientsWithSecond": 0, "crossSourceClinicalPatients": 0})
    out = generate(tiny_spec, _target(tmp_path, "clean"), repository_root)
    assert read_json(out / "problem-answer-key.json")["count"] == 0
    marker = tiny_spec["metadata"]["problemMarkerSystem"]
    assert all(
        tag.get("system") != marker
        for org_id in ("hmo", "hospital-a", "hospital-b")
        for entry in read_json(source_bundle_path(out, org_id))["entry"]
        for tag in entry["resource"].get("meta", {}).get("tag", [])
    )
    from synthetic_population.detectors import detect_group
    quality_kinds = {
        "conflicting-diagnosis", "discordant-lab", "timestamp-order",
        "semantically-wrong-code", "unit-mismatch", "stale-demographics",
        "probable-duplicate",
    }
    for org_id in ("hmo", "hospital-a", "hospital-b"):
        entries = read_json(source_bundle_path(out, org_id))["entry"]
        for patient in (entry for entry in entries if entry["resource"]["resourceType"] == "Patient"):
            patient_url = patient["fullUrl"]
            graph = [patient["resource"], *[
                entry["resource"] for entry in entries
                if entry is not patient and patient_url in set(_references(entry["resource"]))
            ]]
            assert all(not detect_group(graph, kind) for kind in quality_kinds)
