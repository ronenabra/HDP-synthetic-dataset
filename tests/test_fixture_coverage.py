import synthetic_population
"""Explicit catalog fixtures; these are never population richness evidence."""
import copy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from synthetic_population.config import load_spec, scale_spec
from synthetic_population.utils import read_json
from synthetic_population.problems import PROBLEM_VARIANTS

ROOT = Path(synthetic_population.__file__).resolve().parent
SPEC = load_spec(ROOT / 'data/specs/native.json')
CAPTURE = read_json(ROOT / SPEC['remediation']['capturePolicyCatalog'])
QUALITY_CASES = [(kind, variant) for kind, variants in PROBLEM_VARIANTS.items() for variant in variants]
CAPTURE_CASES = [(row['source'], row['disposition']) for row in CAPTURE['coverageFixtures']]


def test_fixture_catalog_completeness():
    assert len(QUALITY_CASES) == 21 and len({k for k, v in QUALITY_CASES}) == 7
    assert set(CAPTURE_CASES) == {(source, disposition) for source, row in CAPTURE['sources'].items()
                                 for disposition in row['enabledDispositions']}
    assert len(CAPTURE_CASES) == len(set(CAPTURE_CASES))


@pytest.mark.parametrize('source,disposition', CAPTURE_CASES)
def test_capture_branch_projects_observable_content_and_negative(source, disposition):
    from synthetic_population.capture import capture_coverage_fixtures, CaptureDisposition, observable_origin, metadata_content_leaks
    from synthetic_population.care_pathway import ClinicalEvent, ClinicalFact, PatientPathway
    from synthetic_population.temporal import ClinicalInterval
    from synthetic_population.fhir_projection import project_pathway
    row = next(r for r in capture_coverage_fixtures(SPEC['remediation']['capturePolicyCatalog'], SPEC['seed'])['rows']
               if (r['source'], r['disposition']) == (source, disposition))
    assert row['status'] == 'pass'
    instant = datetime(2025, 1, 1, tzinfo=timezone.utc)
    # Realistic coded facts for the role exercised by the production capture engine.
    role = row['eventRole']
    system, code, display = ('http://loinc.org', '6298-4', 'Potassium') if role == 'result' else (
        ('http://www.nlm.nih.gov/research/umls/rxnorm', '860975', 'Metformin') if role in {'prescription', 'dispense'}
        else ('http://snomed.info/sct', '185349003', 'Clinical consultation'))
    fact = ClinicalFact('fact', 'fixture', system, code, display, 'final', 4.5, 'mmol/L', 3.5, 5.1, 'N')
    if role in {'prescription', 'dispense'}:
        fact = replace(fact, status='active', medication_detail={'doseMg': 500, 'frequencyPerDay': 2,
            'routeCode': '26643006', 'exposureStart': '2025-01-01', 'ingredients': ['metformin']})
    event = ClinicalEvent(row['fixtureId'], 'episode', fact.id, role, 'fixture', row['careSetting'], role,
                          'completed', ClinicalInterval(instant, instant, 'instant'))
    pathway = PatientPathway('fixture', 'cohort-000001', 'chronic-care', 'fixture', 'none', 'none', (fact,), (event,), ())
    capture = CaptureDisposition(event.id, source, disposition, 30, source != row['routedSource'], 'fixture', row['routedSource'])
    positive, mappings = project_pathway(SPEC, {'id': source}, 1, 'urn:uuid:patient', pathway, [capture])
    absent, absent_mappings = project_pathway(SPEC, {'id': source}, 1, 'urn:uuid:patient', pathway, [replace(capture, disposition='absent')])
    assert absent == absent_mappings == []
    if disposition == 'absent':
        assert positive == mappings == []
        # The owner still retains the actual event: absence is source-local.
        owner = row['routedSource']
        owner_rows, _ = project_pathway(SPEC, {'id': owner}, 1, 'urn:uuid:patient', pathway,
            [replace(capture, source=owner, disposition='performed', provenance_required=False)])
        assert owner_rows
        return
    assert len(positive) == len(mappings) == 1
    resource = positive[0][1]
    assert observable_origin(resource, source) == row['routedSource']
    assert mappings[0]['originSource'] == row['routedSource']
    if disposition == 'metadata-only':
        assert resource['resourceType'] == 'DocumentReference'
        assert not metadata_content_leaks(resource)
        assert 'valueQuantity' not in resource and 'code' not in resource
        broken = copy.deepcopy(resource)
        broken['content'][0]['attachment']['data'] = 'NC41'
        assert metadata_content_leaks(broken)
    elif disposition == 'referral-only':
        assert resource['resourceType'] == 'ServiceRequest' and resource['status'] == 'active'
        assert all('reference' not in p for p in resource['performer'])
    elif disposition == 'summarized':
        assert 'issued' not in resource and 'supportingInfo' not in resource
        assert resource.get('code') or resource.get('medicationCodeableConcept') or resource.get('class')
    else:
        assert resource.get('code') or resource.get('medicationCodeableConcept') or resource.get('class')
        if role == 'result':
            assert resource['valueQuantity']['value'] == 4.5


def restore_truth(bundles, answer):
    restored = copy.deepcopy(bundles)
    for problem in answer['problems']:
        for state in problem['originalCleanTruth']['participantStates']:
            entries = restored[state['organization']]['entry']
            if state['operation'] == 'remove':
                entries[:] = [e for e in entries if e['fullUrl'] != state['fullUrl']]
            else:
                entry = next(e for e in entries if e['fullUrl'] == state['fullUrl'])
                entry['resource'] = copy.deepcopy(state['resource'])
    return restored


@pytest.mark.parametrize('kind,variant', QUALITY_CASES)
def test_quality_fixture_complete_blind_corpus_and_reversibility(repository_root, monkeypatch, kind, variant):
    import synthetic_population.bundle as builder
    from synthetic_population.archetypes import primary_archetype
    from synthetic_population.pathways import care_pathway
    from synthetic_population.blind import detect_corpus, evaluate_candidates, blind_resource
    spec = scale_spec(SPEC, 30, '4.1.0-s10-fixture', zero_problems=True)
    spec['problems']['patientsWithOne'] = 1
    index = next(i for i in range(1, 31) if primary_archetype(spec, i) == 'chronic-care'
                 and care_pathway(spec, i, 'chronic-care')['pathwayVariant'] == 'diabetes-stable')
    problem = {'problemId': 'problem-000001', 'class': 'data-quality', 'kind': kind,
               'variant': variant, 'scenario': None, 'severity': 'moderate'}
    # Fixture-only injection: exercises every declared overlay without changing
    # population allocation or pretending a natural 30-person run covers all variants.
    monkeypatch.setattr(builder, 'allocation', lambda cfg, i: [problem] if i == index else [])
    built = builder.build_population(spec, repository_root)
    candidates = detect_corpus(built['bundles'], spec)
    result = evaluate_candidates(candidates, built['answerKey'], built['scenarioEvidence'])
    assert (result['truePositiveCount'], result['falsePositiveCount'], result['falseNegativeCount']) == (1, 0, 0)
    decisive = [row for row in candidates if row.get('conclusionLevel') != 'uncertain-linkage']
    assert {(row['kind'], row.get('variant')) for row in decisive} == {(kind, variant)}
    restored = restore_truth(built['bundles'], built['answerKey'])
    assert detect_corpus(restored, spec) == []
    # All markers are removed before discovery; restoring labels alone cannot
    # resurrect a conclusion whose decisive clinical facts have been repaired.
    stripped = {source: {**bundle, 'entry': [{**entry, 'resource': blind_resource(entry['resource'], spec)}
                for entry in bundle['entry']]} for source, bundle in built['bundles'].items()}
    assert detect_corpus(stripped, spec) == candidates
    empty_truth = evaluate_candidates(candidates, {'problems': []}, {'scenarios': []})
    assert empty_truth['falsePositiveCount'] == len(decisive) and not empty_truth['zeroUnexplainedCandidates']
