"""Cache restoration checks; independent of real caches, repositories, or network."""
import hashlib
import json
import shutil

import pytest

from synthetic_population.rehydrate import content_digest, rehydrate, require_references


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def reference_inputs(tmp_path):
    root, cache = tmp_path / "package", tmp_path / "cache"
    context = {"fhirVersion": "4.0.1", "packages": {"example.core": "1.0.0", "example.hdp": "2.0.0"}, "profiles": []}
    for package, version in context["packages"].items():
        directory = cache / f"{package}#{version}" / "package"
        _write_json(directory / "package.json", {"name": package, "version": version, "fhirVersions": ["4.0.1"]})
        _write_json(directory / "StructureDefinition-example.json", {"resourceType": "StructureDefinition", "id": package})
        _write_json(directory / "examples/Patient-example.json", {"resourceType": "Patient"})
        resources = {"StructureDefinition-example.json": (directory / "StructureDefinition-example.json").read_bytes()}
        context["profiles"].append({"path": f"reference/{package}", "package": package, "version": version, "fileCount": 1, "contentSha256": content_digest(resources)})
    seed = root / "reference/coverage-seed.json"
    _write_json(seed, {"resourceType": "Bundle", "type": "transaction", "entry": []})
    context["coverageSeed"] = {"path": "reference/coverage-seed.json", "bytes": seed.stat().st_size, "sha256": hashlib.sha256(seed.read_bytes()).hexdigest()}
    _write_json(root / "data/reference-context.json", context)
    return root, cache


def test_restore_selected_resources_then_reuse_without_cache(reference_inputs, tmp_path):
    root, cache = reference_inputs
    result = rehydrate(root, cache)
    assert result["restoredInputs"] == ["example.core", "example.hdp"]
    assert not list((root / "reference").rglob("Patient-*.json"))
    assert require_references(root)["profileFileCount"] == 2
    reused = rehydrate(root, tmp_path / "missing-cache")
    assert reused["restoredInputs"] == []
    assert set(reused["reusedInputs"]) == {"example.core", "example.hdp", "coverage-seed"}


def test_missing_second_package_does_not_partially_restore(reference_inputs):
    root, cache = reference_inputs
    shutil.rmtree(cache / "example.hdp#2.0.0")
    with pytest.raises(FileNotFoundError, match="example.hdp#2.0.0"):
        rehydrate(root, cache)
    assert not (root / "reference/example.core").exists()
    assert (root / "reference/coverage-seed.json").is_file()


def test_same_version_with_changed_bytes_is_rejected(reference_inputs):
    root, cache = reference_inputs
    _write_json(cache / "example.hdp#2.0.0/package/StructureDefinition-example.json", {"resourceType": "StructureDefinition", "id": "tampered"})
    with pytest.raises(ValueError, match="does not match the pin"):
        rehydrate(root, cache)
    assert not (root / "reference/example.core").exists()


def test_stale_outputs_require_explicit_replacement(reference_inputs):
    root, cache = reference_inputs
    rehydrate(root, cache)
    destination = root / "reference/example.core/StructureDefinition-example.json"
    expected = destination.read_bytes()
    destination.write_bytes(b'{}')
    with pytest.raises(ValueError, match="Stale reference folder"):
        rehydrate(root, cache)
    assert destination.read_bytes() == b'{}'
    assert "example.core" in rehydrate(root, cache, replace=True)["restoredInputs"]
    assert destination.read_bytes() == expected
