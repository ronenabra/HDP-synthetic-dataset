import json

import pytest

from synthetic_population.catalogs import default_catalog
from synthetic_population.config import load_spec
from synthetic_population.variability_v4 import normalized_signature


def test_role_topology_uses_selected_catalog_and_isolates_catalogs(tmp_path):
    resource = {
        "resourceType": "Condition", "clinicalStatus": {"coding": [{"code": "active"}]},
        "code": {"coding": [{"system": "http://snomed.info/sct", "code": "123"}]},
    }
    paths = []
    for family in ("alpha", "beta"):
        path = tmp_path / f"{family}.json"
        path.write_text(json.dumps({"archetypes": {"example": [{"events": [{
            "system": "http://snomed.info/sct", "code": "123", "family": family,
        }]}]}}))
        paths.append(str(path))
    first = normalized_signature([resource], "role-topology", pathway_catalog_path=paths[0])
    second = normalized_signature([resource], "role-topology", pathway_catalog_path=paths[1])
    assert first != second
    assert normalized_signature([resource], "role-topology", pathway_catalog_path=paths[0]) == first
    assert normalized_signature([resource], "semantic", pathway_catalog_path=paths[0]) == normalized_signature([resource], "semantic", pathway_catalog_path=paths[1])


def test_default_signature_uses_current_native_catalog():
    resource = {"resourceType": "Condition", "code": {"coding": [{"system": "http://snomed.info/sct", "code": "38341003"}]}}
    assert normalized_signature([resource], "role-topology") == normalized_signature(
        [resource], "role-topology",
        pathway_catalog_path=default_catalog("remediation", "pathwayTemplateCatalog"),
    )


@pytest.mark.parametrize("legacy_schema", [True, False])
def test_retired_specifications_fail_explicitly(tmp_path, native_spec, legacy_schema):
    if legacy_schema:
        native_spec["schemaVersion"] = "1.0.0"
    else:
        native_spec.pop("remediation")
    path = tmp_path / "retired-spec.json"
    path.write_text(json.dumps(native_spec))
    with pytest.raises(ValueError, match="only current native specifications"):
        load_spec(path)
