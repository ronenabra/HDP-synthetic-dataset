import synthetic_population
"""Catalog clinical proofs are fixture evidence, never population diversity."""
import copy
from pathlib import Path

import pytest

from synthetic_population.config import load_spec
from synthetic_population.care_pathway import INJURY_BRANCH_CONCEPTS
from synthetic_population.utils import read_json, write_json
from synthetic_population.typed_grammar import compile_grammar, evaluate_variant, _observable_graph, _clause_mutation_evidence

ROOT = Path(synthetic_population.__file__).resolve().parent
SPEC = load_spec(ROOT / 'data/specs/native.json')
CATALOG = read_json(ROOT / SPEC['remediation']['pathwayTemplateCatalog'])
PATHWAYS = [(archetype, variant['id']) for archetype, variants in CATALOG['archetypes'].items() for variant in variants]


def clinical_fixture(archetype, variant, birth='1980-01-01', index=1):
    from synthetic_population.care_pathway import build_patient_pathway
    from synthetic_population.capture import project_capture
    from synthetic_population.fhir_projection import project_pathway
    pathway = build_patient_pathway(SPEC, index, archetype, variant, birth,
        SPEC['remediation']['pathwayTemplateCatalog'], SPEC['remediation']['clinicalContentCatalog'])
    captures, _ = project_capture(pathway, SPEC['remediation']['capturePolicyCatalog'], SPEC['seed'])
    resources = []
    for source, rows in captures.items():
        emitted, _ = project_pathway(SPEC, {'id': source}, index, f'urn:uuid:{source}-patient', pathway, rows)
        resources.extend({**resource, '_source': source, '_fullUrl': url} for url, resource in emitted)
    return pathway, resources


@pytest.mark.parametrize('archetype,variant', PATHWAYS)
def test_material_pathway_observable_proof_and_independent_negatives(archetype, variant):
    birth = '2025-05-01' if 'infant' in variant else '2020-01-01' if 'child' in variant or archetype == 'pediatric' and 'adolescent' not in variant else '2010-01-01' if archetype == 'pediatric' else '1980-01-01'
    pathway, raw = clinical_fixture(archetype, variant, birth)
    grammar = compile_grammar(SPEC['remediation']['pathwayTemplateCatalog'])
    graph = _observable_graph(raw)
    clauses = grammar[archetype][variant]
    result = evaluate_variant(clauses, graph, {'birthDate': birth})
    assert all(r['passed'] for r in result), result
    inferred = [v for v, rules in grammar[archetype].items() if all(r['passed'] for r in evaluate_variant(rules, graph, {'birthDate': birth}))]
    assert inferred == [variant]
    from synthetic_population.observable import codings
    for position, clause in enumerate(clauses):
        approved_codes = {(clause.code_system, code) for code in (clause.code, *clause.alternate_codes)}
        approved_codes.update(clause.alternate_codings)
        removed = [r for r in graph if not (r['resourceType'] in clause.resource_types and any(
            code in set(codings(r)) for code in approved_codes
        ))]
        assert not all(r['passed'] for r in evaluate_variant(clauses, removed))
        other = next((c for c in clauses if (c.code_system,c.code) != (clause.code_system,clause.code)), None)
        evidence = _clause_mutation_evidence(clause, clauses[position-1] if position else None,
            (other.code_system,other.code) if other else None)
        assert evidence['status'] == 'pass', evidence
    assert len({e.occurrence.start for e in pathway.events}) >= 3


def test_pathway_fixture_catalog_completeness():
    assert len({a for a,v in PATHWAYS}) == 8
    assert {'hypertension-stable', 'infant-well-child', 'child-well-child', 'adolescent-preventive'} <= {v for a,v in PATHWAYS}
    grammar = compile_grammar(SPEC['remediation']['pathwayTemplateCatalog'])
    assert set(PATHWAYS) == {(a,v) for a, variants in grammar.items() for v in variants}


@pytest.mark.parametrize('archetype,variant', [
    ('generally-healthy', 'minor-injury'),
    ('pediatric', 'minor-injury'),
    ('womens-health', 'ordinary-injury'),
])
def test_injury_branches_change_observable_diagnosis(archetype, variant):
    from synthetic_population.observable import codings
    observed = set()
    for index in range(1, 25):
        pathway, resources = clinical_fixture(archetype, variant, index=index)
        branch = pathway.secondary_branch
        code = INJURY_BRANCH_CONCEPTS[branch][0]
        observed.add(branch)
        injury_events = [event for event in pathway.events
                         if event.role in {'emergency-encounter', 'condition'}
                         and next(fact for fact in pathway.facts if fact.id == event.fact_id).code == code]
        assert len(injury_events) == 2
        assert {event.care_setting for event in injury_events} <= {'regional-emergency', 'tertiary-emergency'}
        assert {next(fact for fact in pathway.facts if fact.id == event.fact_id).code for event in injury_events} == {code}
        assert any(resource['resourceType'] == 'Condition' and ('http://snomed.info/sct', code) in set(codings(resource))
                   for resource in resources)
    assert observed == set(CATALOG['archetypes'][archetype][next(
        i for i, row in enumerate(CATALOG['archetypes'][archetype]) if row['id'] == variant
    )]['secondaryBranches'])


def test_regional_short_admission_has_local_procedure_order():
    from synthetic_population.capture import project_capture
    pathway, _ = clinical_fixture('perioperative-surgical', 'short-admission')
    orders = [event for event in pathway.events if event.role == 'order' and event.care_setting == 'regional-inpatient']
    assert len(orders) == 1
    captures, routes = project_capture(pathway, SPEC['remediation']['capturePolicyCatalog'], SPEC['seed'])
    assert routes[orders[0].id] == 'hospital-b'
    assert any(row.latent_event_id == orders[0].id and row.disposition == 'authored'
               for row in captures['hospital-b'])


@pytest.mark.parametrize('mutation', ['unknown-action', 'wrong-status', 'wrong-ingredient', 'unknown-field'])
def test_workflow_actions_are_executable_and_schema_remains_closed(tmp_path, mutation):
    catalog = copy.deepcopy(CATALOG)
    event = next(e for e in catalog['archetypes']['medication-intensive'][0]['events'] if 'workflowAction' in e)
    if mutation == 'unknown-action': event['workflowAction'] = 'unsupported'
    elif mutation == 'wrong-status': event['status'] = 'stopped'
    elif mutation == 'wrong-ingredient': event['code'] = '860975'
    else: event['decorativePromise'] = True
    path = tmp_path / 'bad.json'
    write_json(path, catalog)
    with pytest.raises(ValueError): compile_grammar(str(path))


def test_infant_spacing_follows_existing_age_compression():
    birth = '2025-12-01'
    pathway, raw = clinical_fixture('pediatric', 'infant-well-child', birth)
    clauses = compile_grammar(SPEC['remediation']['pathwayTemplateCatalog'])['pediatric']['infant-well-child']
    graph = _observable_graph(raw)
    assert all(r['passed'] for r in evaluate_variant(clauses, graph, {'birthDate': birth}, SPEC['timelineAnchor']))
    assert all(e.occurrence.start.date().isoformat() >= birth for e in pathway.events)
    broken = copy.deepcopy(graph)
    for r in broken:
        if r.get('resourceType') == 'Observation': r['effectiveDateTime'] = '1900-01-01T00:00:00Z'
    assert not all(r['passed'] for r in evaluate_variant(clauses, broken, {'birthDate': birth}, SPEC['timelineAnchor']))


from synthetic_population.problems import SCENARIO_VARIANTS
SAFETY = [(family, variant) for family, variants in SCENARIO_VARIANTS.items() for variant in variants]


@pytest.mark.parametrize('family,variant', SAFETY)
def test_safety_controls_and_all_source_subsets_agree_with_blind_discovery(family, variant):
    from itertools import combinations
    from synthetic_population.problems import allocation_table, additional_fixture_conclusions
    from synthetic_population.bundle import _clinical_resources, _scenario_variant_resources, _apply_matched_control
    from synthetic_population.blind import detect_corpus, blind_resource
    from synthetic_population.predicates import subset_truth_table_variant
    from synthetic_population.scenario_audit import _counterfactuals
    from synthetic_population.ids import cohort_id, full_url
    index, problems = next((i, rows) for i, rows in allocation_table(SPEC).items()
        if rows and rows[0].get('scenario') == family and rows[0]['variant'] == variant)
    problem = problems[0]
    for control in (False, True):
        bundles, resources = {}, {}
        for org in SPEC['organizations']:
            source = org['id']
            patient_url = full_url(SPEC['datasetVersion'], source, cohort_id(index), 'Patient')
            clinical = _clinical_resources(SPEC, org, index, patient_url, [] if control else problems, '1980-01-01', full_capture=True)
            if control:
                _apply_matched_control(SPEC, index, family, source, patient_url, clinical, SCENARIO_VARIANTS[family].index(variant))
            _scenario_variant_resources(SPEC, source, index, patient_url, problem, clinical, control=control)
            patient = {'resourceType':'Patient','name':[{'given':['Rina'],'family':'Fixture'}], 'gender':'female','birthDate':'1980-01-01'}
            bundles[source] = {'entry':[{'fullUrl':patient_url,'resource':patient}] + [{'fullUrl':u,'resource':r} for u,r in clinical]}
            resources[source] = [{**blind_resource(r,SPEC),'_source':source,'_fullUrl':u} for u,r in clinical]
        expected = set(additional_fixture_conclusions(variant, control=control)) | (set() if control else {variant})
        table = subset_truth_table_variant(family, variant, resources, require_three_source_context=True)
        assert len(table) == 7
        for size in (1,2,3):
            for sources in combinations(sorted(bundles),size):
                observed = {row['variant'] for row in detect_corpus({s:bundles[s] for s in sources},SPEC)
                    if row.get('conclusionLevel')=='strict-variant' and row.get('scenario')==family}
                assert observed == (expected if size==3 else set())
                assert table['+'.join(sources)] == (not control and size==3)
        if not control:
            mutations = _counterfactuals(family,variant,[r for values in resources.values() for r in values])
            assert mutations and all(r['passed'] for r in mutations)


def test_sensitive_fixture_projections_and_missing_injury_regression(repository_root, tmp_path):
    from synthetic_population.config import scale_spec
    from synthetic_population.bundle import build_population
    from synthetic_population.artifacts import bundle_relative_path
    from synthetic_population.projections import aggregate, challenge, mobility, _references
    from synthetic_population.sensitivity import realization_audit
    from synthetic_population.protected import build_protected_invariants_audit
    from synthetic_population.latent_audit import build_latent_mapping_audit
    spec = scale_spec(SPEC,100,'4.1.1-s11-security',zero_problems=True)
    # Explicit six-fact fixture matrix, independent of naturally rounded counts.
    spec['sensitivity']['factCohorts'] = {key:1 for key in spec['sensitivity']['factCohorts']}
    built = build_population(spec,repository_root)
    audit = realization_audit(built['bundles'],spec)
    assert audit['status']=='pass', audit['errors']
    assert build_latent_mapping_audit(built['bundles'],built['crosswalk'],built['latentTruthLedger'],spec)['status']=='pass'
    wrong_result=copy.deepcopy(built['bundles'])
    observed=next(e['resource'] for e in wrong_result['hmo']['entry']
        if e['resource'].get('valueCodeableConcept',{}).get('coding',[{}])[0].get('code')=='260373001')
    observed['valueCodeableConcept']['coding'][0]['code']='260385009'
    assert build_latent_mapping_audit(wrong_result,built['crosswalk'],built['latentTruthLedger'],spec)['status']=='fail'
    run=tmp_path/'run'
    write_json(run/'dataset-spec.json',spec)
    for source,bundle in built['bundles'].items(): write_json(run/bundle_relative_path(spec,source),bundle)
    conf=spec['metadata']['confidentialitySystem']
    def codes(r,system): return {c['code'] for c in r.get('meta',{}).get('security',[]) if c['system']==system}
    views={}
    for name, fn in [('aggregate',aggregate),('challenge',challenge),('ordinary',mobility),
                     ('authorized',lambda out,dest:mobility(out,dest,True,('MH','HIV')))]:
        path=fn(run,tmp_path/f'{name}.json'); views[name]=read_json(path)
        urls={e['fullUrl'] for e in views[name]['entry']}
        for entry in views[name]['entry']:
            r=entry['resource']
            assert spec['metadata']['testReasonCode'] in codes(r,spec['metadata']['actReasonSystem'])
            assert all(ref in urls for ref in _references(r) if ref.startswith('urn:uuid:'))
            if name in {'ordinary','authorized'}: assert 'V' not in codes(r,conf)
            if name=='ordinary': assert 'R' not in codes(r,conf)
        if name in {'ordinary','authorized'}:
            assert any(any(t.get('code')=='ordinary-injury' for t in e['resource'].get('meta',{}).get('tag',[])) for e in views[name]['entry'])
    assert len(views['authorized']['entry']) > len(views['ordinary']['entry'])
    mh=read_json(mobility(run,tmp_path/'mh-only.json',True,('MH',)))
    assert all(not ('R' in codes(e['resource'],conf) and 'HIV' in codes(e['resource'],spec['metadata']['clinicalSensitivitySystem'])) for e in mh['entry'])
    # Security mutations must fail independently of normal clinical evidence.
    broken=copy.deepcopy(built['bundles'])
    r=next(e['resource'] for e in broken['hmo']['entry'] if e['resource']['resourceType']=='Patient')
    r['meta']['security']=[c for c in r['meta']['security'] if c['system']!=spec['metadata']['actReasonSystem']]
    assert 'SECURITY-HTEST' in build_protected_invariants_audit(broken,built['crosswalk'],built['answerKey'],audit,spec,repository_root)['failures']
    # An ordinary resource referring to a V resource must be removed transitively.
    victim=next(e for b in built['bundles'].values() for e in b['entry'] if 'V' in codes(e['resource'],conf))
    source=next(s for s,b in built['bundles'].items() if victim in b['entry'])
    ordinary=next(e for e in built['bundles'][source]['entry'] if e['resource']['resourceType']=='Observation' and not codes(e['resource'],conf))
    ordinary['resource']['derivedFrom']=[{'reference':victim['fullUrl']}]
    write_json(run/bundle_relative_path(spec,source),built['bundles'][source])
    projected=read_json(mobility(run,tmp_path/'mutated.json',True,('MH','HIV')))
    assert ordinary['fullUrl'] not in {e['fullUrl'] for e in projected['entry']}
    assert projected['_projectionEvidence']['excluded']['referenceClosure']>0
