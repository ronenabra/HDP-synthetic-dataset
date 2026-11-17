def test_v4_canonical_targets_and_compatible_allocations_unchanged(repository_root):
    from synthetic_population.config import load_spec, scale_spec
    from synthetic_population.problems import allocation_table
    spec = load_spec(repository_root / 'data/specs/native.json')
    before = allocation_table(spec)
    after = allocation_table(scale_spec(spec, 1000, '4.1.0-test-canonical'))
    assert before == after
    assert sum(bool(rows) for rows in after.values()) == 100
    assert sum(len(rows) == 2 for rows in after.values()) == 20
    assert sum(row['class'] == 'cross-source-clinical' for rows in after.values() for row in rows) == 30
    from synthetic_population.archetypes import primary_archetype
    for index, rows in after.items():
        for row in rows:
            assert primary_archetype(spec, index) in row['prerequisites']['compatibleArchetypes']
