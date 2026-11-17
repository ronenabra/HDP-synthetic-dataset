from synthetic_population.detectors import detect_group, detects


def test_resource_local_detectors_use_semantic_inconsistency():
    assert not detects({"code": {"coding": [{"system": "http://snomed.info/sct", "code": "44054006"}]}}, "conflicting-diagnosis")
    assert not detects({"valueQuantity": {"value": 12.5}}, "discordant-lab")
    assert detects({"issued": "2025-01-01T00:00:00Z", "effectiveDateTime": "2025-01-01T01:00:00Z"}, "timestamp-order")
    assert detects({"resourceType": "Condition", "code": {"coding": [{"system": "http://snomed.info/sct", "code": "119297000"}]}}, "semantically-wrong-code")
    assert detects({"resourceType": "Observation", "valueQuantity": {"unit": "mg/dL", "code": "mmol/L"}}, "unit-mismatch")
    assert not detects({"resourceType": "Patient", "address": [{"period": {"end": "2024-01-01"}}, {"period": {"start": "2024-01-02"}}]}, "stale-demographics")
    assert not detects({"identifier": [{"system": "urn:hdp-test-bundle:synthetic:duplicate-candidate"}]}, "probable-duplicate")


def test_group_only_detectors_require_relationship_evidence():
    clinical = {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]}
    type1 = {"resourceType": "Condition", "clinicalStatus": clinical, "code": {"coding": [{"system": "http://snomed.info/sct", "code": "46635009"}]}}
    type2 = {"resourceType": "Condition", "clinicalStatus": clinical, "code": {"coding": [{"system": "http://snomed.info/sct", "code": "44054006"}]}}
    assert detect_group([type1, type2], "conflicting-diagnosis")
    assert not detect_group([type2], "conflicting-diagnosis")


def test_timestamp_variants_have_independent_patient_and_anchor_evidence():
    from synthetic_population.detectors import quality_variant_evidence
    patient = {'resourceType': 'Patient', 'birthDate': '2000-01-01'}
    for effective, issued, variant in [
        ('2025-01-01T12:00:00Z', '2025-01-01T11:00:00Z', 'issued-before-effective'),
        ('2027-01-01T12:00:00Z', '2027-01-01T13:00:00Z', 'future-result'),
        ('1999-01-01T12:00:00Z', '1999-01-01T13:00:00Z', 'pre-birth-event'),
    ]:
        observation = {'resourceType': 'Observation', 'effectiveDateTime': effective, 'issued': issued}
        assert quality_variant_evidence([patient, observation], 'timestamp-order', anchor='2026-01-01') == {variant}
    assert not quality_variant_evidence([patient, {'effectiveDateTime': '2025-01-01T12:00:00Z', 'issued': '2025-01-01T13:00:00Z'}], 'timestamp-order', anchor='2026-01-01')


def _lab(value=13, unit='g/dL', specimen='Specimen/a', when='2025-01-01T12:00:00Z'):
    return {'resourceType': 'Observation', 'status': 'final',
            'code': {'coding': [{'system': 'http://loinc.org', 'code': '718-7'}]},
            'valueQuantity': {'value': value, 'unit': unit, 'code': unit},
            'specimen': {'reference': specimen}, 'effectiveDateTime': when}


def test_unit_alias_and_conversion_controls():
    from synthetic_population.detectors import quality_variant_evidence
    left = _lab()
    left['valueQuantity']['unit'] = 'g per dL'
    assert not detects(left, 'unit-mismatch')
    right = _lab(130, 'g/L')
    right['_fullUrl'] = 'Observation/source'
    left['derivedFrom'] = [{'reference': 'Observation/source'}]
    assert not quality_variant_evidence([left, right], 'unit-mismatch')
    left['valueQuantity']['value'] = 130
    assert quality_variant_evidence([left, right], 'unit-mismatch') == {'scale-mismatch'}
    assert not quality_variant_evidence([left, right], 'discordant-lab')


def test_lab_comparison_preserves_legitimate_repeat_and_correction():
    from synthetic_population.detectors import quality_variant_evidence
    left, right = _lab(13), _lab(11)
    assert quality_variant_evidence([left, right], 'discordant-lab') == {'same-specimen-disagreement'}
    right['specimen']['reference'] = 'Specimen/b'
    assert not quality_variant_evidence([left, right], 'discordant-lab')
    right['specimen']['reference'] = 'Specimen/a'
    right['status'] = 'corrected'
    assert not quality_variant_evidence([left, right], 'discordant-lab')


def test_diagnosis_snapshot_conflicts_do_not_use_diabetes_shortcut():
    import copy
    from synthetic_population.detectors import quality_variant_evidence
    left = {'resourceType': 'Condition', 'identifier': [{'system': 'urn:source:diagnosis', 'value': '12'}],
            'code': {'coding': [{'system': 'http://snomed.info/sct', 'code': '38341003'}]},
            'clinicalStatus': {'coding': [{'system': 'http://terminology.hl7.org/CodeSystem/condition-clinical', 'code': 'active'}]}, 'recordedDate': '2025-01-01', 'onsetDateTime': '2020-01-01'}
    right = copy.deepcopy(left)
    right['clinicalStatus']['coding'][0]['code'] = 'resolved'
    assert quality_variant_evidence([left, right], 'conflicting-diagnosis') == {'status-lifecycle-conflict'}
    right['recordedDate'] = '2026-01-01'
    assert not quality_variant_evidence([left, right], 'conflicting-diagnosis')
    right = copy.deepcopy(left)
    right['onsetDateTime'] = '2022-01-01'
    assert quality_variant_evidence([left, right], 'conflicting-diagnosis') == {'onset-provenance-conflict'}
    right['identifier'][0]['value'] = 'different-episode'
    assert not quality_variant_evidence([left, right], 'conflicting-diagnosis')


def test_empty_truth_counts_observed_kind_as_false_positive():
    from synthetic_population.blind import evaluate_candidates
    candidate = {'kind': 'timestamp-order', 'variant': 'future-result', 'resourceFullUrls': ['Observation/1']}
    result = evaluate_candidates([candidate], {'problems': []}, {'scenarios': []})
    assert result['falsePositiveCount'] == 1
    assert not result['zeroUnexplainedCandidates']
    assert result['byProblemKind']['timestamp-order']['precision'] == 0
    assert result['byProblemVariant']['timestamp-order:future-result']['falsePositiveCount'] == 1
    assert result['byProblemKind']['unit-mismatch']['candidateCount'] == 0


def test_scoring_requires_scenario_and_variant_and_one_to_one_evidence():
    from synthetic_population.blind import evaluate_candidates
    problem = {'problemId': 'p1', 'kind': 'cross-source-clinical-risk', 'scenario': 'allergy-treatment-risk',
               'variant': 'administration-after-allergy', 'participants': [{'fullUrl': 'Observation/1'}]}
    candidate = {'kind': problem['kind'], 'scenario': problem['scenario'], 'variant': 'prescribing-after-allergy',
                 'resourceFullUrls': ['Observation/1']}
    result = evaluate_candidates([candidate], {'problems': [problem]}, {'scenarios': []})
    assert (result['truePositiveCount'], result['falsePositiveCount'], result['falseNegativeCount']) == (0, 1, 1)
    candidate['variant'] = problem['variant']
    result = evaluate_candidates([candidate], {'problems': [problem, {**problem, 'problemId': 'p2'}]}, {'scenarios': []})
    assert (result['truePositiveCount'], result['falseNegativeCount']) == (1, 1)


def _duplicate_oracle(patients):
    from itertools import combinations
    from synthetic_population.detectors import duplicate_linkage_score
    found = []
    for left, right in combinations(range(len(patients)), 2):
        score, evidence = duplicate_linkage_score(patients[left], patients[right])
        if score >= 8 and all(evidence[key] for key in ('family', 'given', 'birthClose')):
            found.append((left, right, score, evidence))
    return found


def test_duplicate_blocking_matches_exhaustive_scores_and_boundaries():
    import copy
    from synthetic_population.detectors import blocked_duplicate_pairs
    base = {'resourceType': 'Patient', 'name': [{'family': 'Cohen', 'given': ['Rina']}],
            'birthDate': '1980-01-02', 'gender': 'female',
            'telecom': [{'system': 'phone', 'value': '+972-50-1234567'}],
            'address': [{'city': 'Haifa', 'line': ['Synthetic Street 1']}]}
    patients = [base]
    for field, value in [('name', [{'family': 'COHEN', 'given': ['Rinah']}]),
                         ('telecom', [{'system': 'phone', 'value': '0509999999'}]),
                         ('birthDate', '1980-01-01'), ('birthDate', '1980-01-03'),
                         ('birthDate', '1980-01-04'), ('birthDate', ''),
                         ('name', [{'family': 'Levi', 'given': ['Rina']}]),
                         ('name', [{'family': 'Cohen', 'given': ['Dana']}]),
                         ('name', [])]:
        row = copy.deepcopy(base)
        row[field] = value
        # Missing names are represented by the same FHIR shape as existing fixtures.
        if field == 'name' and not value:
            row.pop('name')
        patients.append(row)
    stats = {}
    actual = list(blocked_duplicate_pairs(patients, stats))
    assert actual == _duplicate_oracle(patients)
    pairs = {(a, b) for a, b, *_ in actual}
    assert {(0, 1), (0, 2), (0, 3), (0, 4)} <= pairs
    assert not any((0, i) in pairs for i in range(5, 10))
    assert stats['normalizedPatients'] == len(patients)
    assert stats['duplicateScoreComparisons'] < len(patients) * (len(patients) - 1) // 2


def test_duplicate_work_scales_with_unique_name_groups():
    from synthetic_population.detectors import blocked_duplicate_pairs
    counts = []
    for size in (40, 80):
        patients = [{'name': [{'family': f'Family{i // 2}', 'given': ['Rina']}],
                     'birthDate': '1980-01-01', 'gender': 'female',
                     'telecom': [{'value': '0501234567'}]} for i in range(size)]
        stats = {}
        assert list(blocked_duplicate_pairs(patients, stats)) == _duplicate_oracle(patients)
        counts.append(stats)
    assert counts[1] == {key: 2 * value for key, value in counts[0].items()}


def test_corpus_counters_preserve_candidates_and_conclusions(tiny_spec, repository_root, monkeypatch):
    from synthetic_population import blind
    from synthetic_population.bundle import build_population
    built = build_population(tiny_spec, repository_root)
    stats = {}
    candidates = blind.detect_corpus(built['bundles'], tiny_spec, work_stats=stats)
    measured = blind.evaluate_candidates(candidates, built['answerKey'], built['scenarioEvidence'], work_stats=stats)
    monkeypatch.setattr(blind, 'blocked_duplicate_pairs', lambda patients, work_stats=None: iter(_duplicate_oracle(patients)))
    exhaustive = blind.detect_corpus(built['bundles'], tiny_spec)
    assert candidates == exhaustive
    baseline = blind.evaluate_candidates(exhaustive, built['answerKey'], built['scenarioEvidence'])
    assert {k: v for k, v in measured.items() if k not in {'work', 'schemaVersion'}} == {k: v for k, v in baseline.items() if k != 'schemaVersion'}
    assert stats['normalizedPatients'] == sum(entry['resource']['resourceType'] == 'Patient' for bundle in built['bundles'].values() for entry in bundle['entry'])
