from copy import deepcopy
from pathlib import Path

import pytest
import synthetic_population

from synthetic_population.config import load_spec, scale_spec


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return Path(synthetic_population.__file__).resolve().parent


@pytest.fixture()
def native_spec(repository_root):
    return load_spec(repository_root / "data/specs/native.json")


@pytest.fixture()
def tiny_spec(native_spec):
    return scale_spec(native_spec, 30, "4.0.0-test30")


@pytest.fixture()
def lifecycle_spec(native_spec):
    spec = deepcopy(native_spec)
    spec["datasetVersion"] = "4.0.0-lifecycle"
    spec["demographics"]["lifecyclePolicyCatalog"] = "data/demographic-lifecycle.v1.0.0.json"
    return spec


@pytest.fixture()
def condition_spec(lifecycle_spec):
    spec = deepcopy(lifecycle_spec)
    spec["datasetVersion"] = "4.0.0-condition-capture"
    spec["terminology"]["catalog"] = "data/terminology.v2.4.0.json"
    return spec
