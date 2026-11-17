import copy

import pytest

from synthetic_population.config import load_spec, scale_spec
from synthetic_population.utils import write_json


def test_canonical_spec_loads(repository_root):
    spec = load_spec(repository_root / "data/specs/native.json")
    assert spec["populationSize"] == 1000
    assert spec["problems"]["patientsWithOne"] == 100


def test_unknown_field_is_rejected(tmp_path, tiny_spec):
    tiny_spec["surprise"] = True
    path = tmp_path / "spec.json"
    write_json(path, tiny_spec)
    with pytest.raises(ValueError, match="unknown"):
        load_spec(path)


def test_missing_field_is_rejected(tmp_path, tiny_spec):
    tiny_spec.pop("seed")
    path = tmp_path / "spec.json"
    write_json(path, tiny_spec)
    with pytest.raises(ValueError, match="missing"):
        load_spec(path)


def test_nested_unknown_field_is_rejected(tmp_path, tiny_spec):
    tiny_spec["storage"]["surprise"] = 1
    path = tmp_path / "spec.json"
    write_json(path, tiny_spec)
    with pytest.raises(ValueError, match="storage fields mismatch"):
        load_spec(path)


def test_invalid_organization_order_is_rejected(tmp_path, tiny_spec):
    tiny_spec["organizations"].reverse()
    path = tmp_path / "spec.json"
    write_json(path, tiny_spec)
    with pytest.raises(ValueError, match="organizations"):
        load_spec(path)


def test_scale_spec_has_explicit_counts(repository_root):
    source = load_spec(repository_root / "data/specs/native.json")
    scaled = scale_spec(source, 100, "4.0.0-test100")
    assert scaled["problems"]["patientsWithOne"] == 10
    assert scaled["problems"]["patientsWithSecond"] == 2
    assert scaled["problems"]["crossSourceClinicalPatients"] == 3
    assert source["populationSize"] == 1000


@pytest.mark.parametrize('size,primary,secondary,safety', [
    (1, 0, 0, 0), (4, 0, 0, 0), (5, 1, 0, 0), (16, 2, 0, 0),
    (17, 2, 0, 1), (30, 3, 0, 1), (49, 5, 0, 1), (50, 5, 1, 2),
    (100, 10, 2, 3), (1000, 100, 20, 30),
])
def test_v4_natural_rounding_boundaries(repository_root, tmp_path, size, primary, secondary, safety):
    from synthetic_population.config import NATURAL_ALLOCATION_POLICY, NESTED_ALLOCATION_POLICY
    source = load_spec(repository_root / 'data/specs/native.json')
    scaled = scale_spec(source, size, f'4.1.0-test{size}')
    assert scaled == scale_spec(source, size, f'4.1.0-test{size}')
    cfg = scaled['problems']
    assert (cfg['patientsWithOne'], cfg['patientsWithSecond'], cfg['crossSourceClinicalPatients']) == (primary, secondary, safety)
    assert cfg['allocationPolicyVersion'] == (NESTED_ALLOCATION_POLICY if size >= 1000 else NATURAL_ALLOCATION_POLICY)
    assert cfg['maxTaggedClinicalRate'] == .025
    write_json(tmp_path / 'spec.json', scaled)
    assert load_spec(tmp_path / 'spec.json') == scaled
    assert source['problems']['allocationPolicyVersion'] == NESTED_ALLOCATION_POLICY


@pytest.mark.parametrize('size', [1, 30, 100, 1000])
def test_explicit_zero_problem_scaling(repository_root, size):
    from synthetic_population.problems import allocation_table
    source = load_spec(repository_root / 'data/specs/native.json')
    spec = scale_spec(source, size, f'4.1.0-zero{size}', zero_problems=True)
    assert all(spec['problems'][key] == 0 for key in ('patientsWithOne', 'patientsWithSecond', 'crossSourceClinicalPatients'))
    assert not any(allocation_table(spec).values())
    assert spec['sensitivity'] == scale_spec(source, size, f'4.1.0-normal{size}')['sensitivity']
    assert not any(allocation_table(scale_spec(spec, 1000, '4.1.0-zero-derived')).values())


@pytest.mark.parametrize('size,primary,secondary,safety', [(30, 3, 0, 1), (100, 10, 2, 3)])
def test_natural_population_assignments_are_exact_and_compatible(repository_root, size, primary, secondary, safety):
    from synthetic_population.archetypes import primary_archetype
    from synthetic_population.problems import allocation_table, _ALLOCATION_CACHE
    source = load_spec(repository_root / 'data/specs/native.json')
    spec = scale_spec(source, size, f'4.1.0-assignment{size}')
    first = copy.deepcopy(allocation_table(spec))
    _ALLOCATION_CACHE.clear()
    assert allocation_table(spec) == first
    assert sum(bool(rows) for rows in first.values()) == primary
    assert sum(len(rows) == 2 for rows in first.values()) == secondary
    assert sum(row['class'] == 'cross-source-clinical' for rows in first.values() for row in rows) == safety
    for index, rows in first.items():
        for row in rows:
            assert primary_archetype(spec, index) in row['prerequisites']['compatibleArchetypes']
