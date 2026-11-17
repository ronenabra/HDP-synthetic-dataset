import pytest

from synthetic_population.ids import deterministic_uuid
from synthetic_population.metadata import meta, strip_problem_tags


def test_uuid_is_deterministic_and_contextual():
    assert deterministic_uuid("1", "hmo", "Patient", 1) == deterministic_uuid("1", "hmo", "Patient", 1)
    assert deterministic_uuid("1", "hmo", "Patient", 1) != deterministic_uuid("1", "hospital-a", "Patient", 1)


def test_ordinary_metadata_has_no_confidentiality(tiny_spec):
    value = meta(tiny_spec, "hmo", "http://example/Profile", "diagnosis")
    systems = {item["system"] for item in value["security"]}
    assert tiny_spec["metadata"]["confidentialitySystem"] not in systems
    assert [item["code"] for item in value["security"] if item["system"] == tiny_spec["metadata"]["bucketSystem"]] == ["diagnoses"]


def test_n_confidentiality_is_rejected(tiny_spec):
    with pytest.raises(ValueError):
        meta(tiny_spec, "hmo", "http://example/Profile", "diagnosis", confidentiality="N")


def test_problem_tags_are_exactly_strippable(tiny_spec):
    resource = {"meta": meta(tiny_spec, "hmo", "p", "diagnosis", problem_ids=["problem-000001"], problem_kinds=["discordant-lab"])}
    original_security = list(resource["meta"]["security"])
    strip_problem_tags(resource, tiny_spec)
    assert resource["meta"]["security"] == original_security
    assert all("synthetic-problem" not in item["system"] for item in resource["meta"]["tag"])
