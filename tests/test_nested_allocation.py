from collections import Counter
from datetime import date

import pytest

from synthetic_population.archetypes import primary_archetype
from synthetic_population.bundle import matched_controls
from synthetic_population.care_pathway import build_patient_pathway
from synthetic_population.catalogs import load_catalog
from synthetic_population.config import load_spec, scale_spec
from synthetic_population.demographics import demographic_truth, gender_for_patient
from synthetic_population.geography import choose_locality, load_catalog as load_geography
from synthetic_population.nested_allocation import POLICY_VERSION
from synthetic_population.pathways import care_pathway
from synthetic_population.problems import allocation_table
from synthetic_population.sensitivity import fact_for_patient
from synthetic_population.utils import write_json




@pytest.mark.parametrize('size,first,second,cross,controls', [
    (30, 3, 0, 1, 1), (100, 10, 2, 3, 3),
    (1000, 100, 20, 30, 30), (10000, 1000, 200, 300, 300),
])
@pytest.mark.parametrize('zero', [False, True])
def test_allocation_only_exact_and_eligible(repository_root, size, first, second, cross, controls, zero):
    source = load_spec(repository_root / 'data/specs/native.json')
    spec = scale_spec(source, size, f'4.2.0-check-{size}', zero_problems=zero)
    table = allocation_table(spec)
    assert len(table) == size
    assert len({f'cohort-{index:06d}' for index in table}) == size
    assert sum(bool(rows) for rows in table.values()) == (0 if zero else first)
    assert sum(len(rows) == 2 for rows in table.values()) == (0 if zero else second)
    assert sum(row['class'] == 'cross-source-clinical' for rows in table.values() for row in rows) == (0 if zero else cross)
    selected_controls = matched_controls(spec)
    assert len(selected_controls) == (0 if zero else controls)
    assert not set(selected_controls) & {index for index, rows in table.items() if rows}
    assert len({row['problemId'] for rows in table.values() for row in rows}) == sum(map(len, table.values()))
    if size in (1000, 10000):
        from synthetic_population.problems import _ALLOCATION_CACHE
        from synthetic_population.demographics import _GENDER_CACHE
        sample_gender = [gender_for_patient(spec, index) for index in (1, 301, size)]
        _ALLOCATION_CACHE.clear()
        _GENDER_CACHE.clear()
        assert allocation_table(spec) == table
        assert [gender_for_patient(spec, index) for index in (1, 301, size)] == sample_gender
    archetypes = Counter(primary_archetype(spec, index) for index in table)
    sexes = Counter(gender_for_patient(spec, index) for index in table)
    facts = Counter(fact_for_patient(spec, index) for index in table)
    assert sum(archetypes.values()) == size
    assert sexes['female'] == round(size * spec['demographics']['genderDistribution']['female'])
    assert all(gender_for_patient(spec, index) == 'female' for index in table if primary_archetype(spec, index) == 'womens-health')
    assert all(facts[key] == value for key, value in spec['sensitivity']['factCohorts'].items())
    if size >= 1000:
        assert spec['problems']['allocationPolicyVersion'] == POLICY_VERSION
        assert archetypes == {row['id']: row['count'] * size // 1000 for row in load_catalog(spec['archetypes']['catalog'])['primaryDistribution']}
        assert len({primary_archetype(spec, index) for index in range(1001, min(size, 2000) + 1)}) == (8 if size > 1000 else 0)
        for start in range(1, size + 1, 1000):
            block = range(start, start + 1000)
            assert sum(bool(table[index]) for index in block) == (0 if zero else 100)
            assert sum(len(table[index]) == 2 for index in block) == (0 if zero else 20)
            assert sum(index in selected_controls for index in block) == (0 if zero else 30)
    for index, rows in table.items():
        for row in rows:
            assert primary_archetype(spec, index) in row['prerequisites']['compatibleArchetypes']
            assert row['class'] != 'cross-source-clinical' or row['scenario'] in {
                'renal-medication-risk', 'bleeding-risk', 'allergy-treatment-risk'}
    if size >= 1000:
        geography = load_geography(repository_root / spec['geography']['catalog'], repository_root)
        districts = Counter(choose_locality(geography, (index - 1) % 1000 + 1, 1000)['district'] for index in table)
        base_districts = Counter(choose_locality(geography, index, 1000)['district'] for index in range(1, 1001))
        assert districts == {key: count * size // 1000 for key, count in base_districts.items()}
        birth_bands = Counter()
        for index in table:
            if primary_archetype(spec, index) == 'pediatric':
                birth = date.fromisoformat(demographic_truth(spec, index, choose_locality(geography, (index - 1) % 1000 + 1, 1000))['birthDate'])
                anchor = date.fromisoformat(spec['timelineAnchor'][:10])
                age = anchor.year - birth.year - ((anchor.month, anchor.day) < (birth.month, birth.day))
                birth_bands['infant' if age < 1 else 'child' if age < 13 else 'adolescent'] += 1
        assert birth_bands == {key: count * size // 1000 for key, count in spec['demographics']['pediatricAgeBands'].items()}


@pytest.mark.parametrize('mutation,reason', [
    (lambda spec: spec['sensitivity']['factCohorts'].update({'minor-hiv-positive': 151}), 'sensitivity eligibility'),
    (lambda spec: spec['demographics']['genderDistribution'].update({'female': .05, 'male': .95}), 'sex and anchor'),
    (lambda spec: spec['problems'].update({'patientsWithSecond': 80}), 'secondary problems'),
    (lambda spec: spec['demographics']['pediatricAgeBands'].update({'infant': 16}), 'pediatric age'),
])
def test_infeasible_nested_spec_rejected(repository_root, tmp_path, mutation, reason):
    source = load_spec(repository_root / 'data/specs/native.json')
    mutation(source)
    write_json(tmp_path / 'spec.json', source)
    with pytest.raises(ValueError, match=reason):
        load_spec(tmp_path / 'spec.json')


@pytest.mark.parametrize('size', [30, 100, 999, 1001])
def test_nested_policy_rejects_incomplete_blocks(repository_root, tmp_path, size):
    source = load_spec(repository_root / 'data/specs/native.json')
    spec = scale_spec(source, size, f'4.2.0-incomplete-{size}')
    if size == 100:
        assert len(allocation_table(spec)) == 100
        assert sum(bool(rows) for rows in allocation_table(spec).values()) == 10
    spec['problems']['allocationPolicyVersion'] = POLICY_VERSION
    path = tmp_path / 'spec.json'
    write_json(path, spec)
    with pytest.raises(ValueError, match='nested allocation requires complete 1,000-person blocks'):
        load_spec(path)


@pytest.mark.parametrize('zero', [False, True])
def test_all_later_block_clinical_paths_are_age_eligible(repository_root, zero):
    source = load_spec(repository_root / 'data/specs/native.json')
    spec = scale_spec(source, 10000, f'4.2.0-clinical-check-{zero}', zero_problems=zero)
    geography = load_geography(repository_root / spec['geography']['catalog'], repository_root)
    problems = allocation_table(spec)
    controls = set(matched_controls(spec))
    branches = Counter()
    for index in range(1, 10001):
        locality = choose_locality(geography, (index - 1) % 1000 + 1, 1000)
        truth = demographic_truth(spec, index, locality)
        force = index in controls or any(row['class'] == 'cross-source-clinical' for row in problems[index])
        path = care_pathway(spec, index, truth['primaryArchetype'], force_all_sources=force,
                            birth_date=truth['birthDate'])
        typed = build_patient_pathway(spec, index, truth['primaryArchetype'], path['pathwayVariant'],
                                      truth['birthDate'], spec['remediation']['pathwayTemplateCatalog'],
                                      spec['remediation']['clinicalContentCatalog'], path['secondaryTrait'])
        assert typed.secondary_branch
        branches[(truth['primaryArchetype'], path['pathwayVariant'])] += 1
    assert sum(branches.values()) == 10000
    assert len(branches) >= 40
