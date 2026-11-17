def test_v4_natural_pilot_and_zero_problem_realization(repository_root):
    from synthetic_population.config import load_spec, scale_spec
    from synthetic_population.bundle import build_population
    from synthetic_population.metrics import build_quality_metrics, metric_failures, realized_problem_allocation
    from synthetic_population.protected import build_protected_invariants_audit
    from synthetic_population.sensitivity import realization_audit
    from synthetic_population.scenario_audit import build_scenario_causality_audit
    source = load_spec(repository_root / 'data/specs/native.json')
    for zero in (False, True):
        spec = scale_spec(source, 30, f'4.1.0-test30-{zero}', zero_problems=zero)
        built = build_population(spec, repository_root)
        metrics = build_quality_metrics(built['bundles'], built['crosswalk'], spec)
        assert metric_failures(metrics, spec) == []
        counts = metrics['realizedProblemAllocation']
        assert counts['patientsWithOne'] == counts['problemIdCount'] == (0 if zero else 3)
        assert counts['patientsWithSecond'] == 0
        assert counts['crossSourceClinicalPatients'] == (0 if zero else 1)
        assert built['answerKey']['count'] == (0 if zero else 3)
        # Scenario inventory includes one independent matched control per positive.
        assert built['scenarioEvidence']['count'] == (0 if zero else 2)
        protected = build_protected_invariants_audit(
            built['bundles'], built['crosswalk'], built['answerKey'],
            realization_audit(built['bundles'], spec), spec, repository_root)
        noise = protected['evidence']['NOISE-BUDGET']
        assert 'NOISE-BUDGET' not in protected['failures']
        assert noise['maximumTaggedClinicalRate'] == .025
        assert noise['taggedClinicalNumerator'] == (0 if zero else 5)
        assert noise['taggedClinicalRate'] <= .025
        causality = build_scenario_causality_audit(built['bundles'], built['crosswalk'], built['scenarioEvidence'], spec)
        assert causality['status'] == 'pass'
        if not zero:
            # Counts are derived from emitted markers, not copied from requested targets.
            for bundle in built['bundles'].values():
                for entry in bundle['entry']:
                    entry['resource']['meta']['tag'] = [t for t in entry['resource']['meta'].get('tag', [])
                        if t.get('system') not in {spec['metadata']['problemIdSystem'], spec['metadata']['problemKindSystem']}]
            observed = realized_problem_allocation(built['bundles'], spec)
            assert observed['problemIdCount'] == observed['patientsWithOne'] == 0
            metrics['realizedProblemAllocation'] = observed
            assert any('realized problem' in message for message in metric_failures(metrics, spec))
