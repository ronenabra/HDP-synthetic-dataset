import synthetic_population
"""F04/F08/F11 checks use clinical fields rather than assignment labels."""
from copy import deepcopy
from pathlib import Path
import pytest
from synthetic_population.config import load_spec
from synthetic_population.care_pathway import build_patient_pathway
from synthetic_population.capture import project_capture
from synthetic_population.fhir_projection import project_pathway
from synthetic_population.clinical_branch_predicates import material_branch_proof
from synthetic_population.clinical_semantics import meaningful_result, semantic_findings, CONTRACT
from synthetic_population.medication_safety import normalized_dose_mg

ROOT=Path(synthetic_population.__file__).resolve().parent
SPEC=load_spec(ROOT/'data/specs/native.json')


def fixture(archetype,variant,branch):
    birth='2020-01-01' if archetype=='pediatric' else '1980-01-01'
    for index in range(1,40):
        path=build_patient_pathway(SPEC,index,archetype,variant,birth,
            SPEC['remediation']['pathwayTemplateCatalog'],SPEC['remediation']['clinicalContentCatalog'])
        if path.secondary_branch!=branch:continue
        capture,_=project_capture(path,SPEC['remediation']['capturePolicyCatalog'],SPEC['seed'])
        rows=[]
        for source,dispositions in capture.items():
            projected,_=project_pathway(SPEC,{'id':source},index,'patient',path,dispositions)
            rows.extend({**r,'_source':source,'_fullUrl':url} for url,r in projected)
        return path,rows,birth
    raise AssertionError(branch)


@pytest.mark.parametrize('archetype,variant,branch',[
    *[(a,v,b) for a,v in [('chronic-care','medication-adjustment'),('medication-intensive','dose-change')]
      for b in ['dose-increase','dose-decrease','switch-class']],
    ('generally-healthy','preventive-screening','glucose-screen'),
    ('pediatric','respiratory-acute','oxygen-saturation'),
    ('medication-intensive','monitoring-result','glucose-monitor'),
    ('rare-complex','rare-metabolic','glucose-monitor'),
    ('medication-intensive','monitoring-result','cbc-monitor'),
    ('oncology','toxicity-monitoring','hepatic-toxicity'),
    ('oncology','toxicity-monitoring','hematologic-toxicity'),
    *[('perioperative-surgical','procedure-with-medication',b) for b in ['anesthesia','analgesia','antibiotic-prophylaxis']],
])
def test_promised_branches_have_content_and_independent_negative(archetype,variant,branch):
    path,rows,birth=fixture(archetype,variant,branch)
    assert material_branch_proof(branch,rows,birth),branch
    assert not semantic_findings(rows)
    changed=deepcopy(rows)
    if branch in {'dose-increase','dose-decrease','switch-class'}:
        for r in changed:
            if r['resourceType']=='MedicationRequest':r.pop('priorPrescription',None)
    elif branch in CONTRACT['branchTests']:
        code=CONTRACT['branchTests'][branch][0]
        changed=[r for r in changed if not (r['resourceType']=='Observation' and
                 code in {c['code'] for c in r.get('code',{}).get('coding',[])})]
    else:
        for r in changed:r.pop('partOf',None)
    assert not material_branch_proof(branch,changed,birth)


@pytest.mark.parametrize('branch',['dose-increase','dose-decrease'])
def test_unchanged_and_opposite_doses_do_not_prove_direction(branch):
    _,rows,birth=fixture('medication-intensive','dose-change',branch)
    for mutation in ('unchanged','opposite'):
        changed=deepcopy(rows)
        for r in changed:
            if r['resourceType']!='MedicationRequest' or not r.get('priorPrescription'):continue
            prior=next(p for p in changed if p['_fullUrl']==r['priorPrescription'].get('reference'))
            dose=normalized_dose_mg(prior)
            new=dose if mutation=='unchanged' else dose/2 if branch=='dose-increase' else dose*2
            r['dosageInstruction'][0]['doseAndRate'][0]['doseQuantity']['value']=new
        assert not material_branch_proof(branch,changed,birth)


def observation(code='2093-3',**fields):
    return {'resourceType':'Observation','_fullUrl':'result','status':'final','_source':'hmo',
            'code':{'coding':[{'system':'http://loinc.org','code':code}]},'subject':{'reference':'patient'},
            'effectiveDateTime':'2025-01-02T12:00:00Z','valueQuantity':{'value':150,'code':'mg/dL'},**fields}


def order(code):
    return {'resourceType':'ServiceRequest','_fullUrl':'order','status':'completed','_source':'hmo',
            'subject':{'reference':'patient'},'code':{'coding':[{'system':'http://loinc.org','code':code}]},
            'authoredOn':'2025-01-01T12:00:00Z'}


@pytest.mark.parametrize('result_code,order_code',[('2093-3','82675-0'),('718-7','24331-1')])
def test_original_wrong_order_links_fail(result_code,order_code):
    r=observation(result_code,basedOn=[{'reference':'order'}])
    assert any(x['finding']=='F11' for x in semantic_findings([r,order(order_code)]))


def test_panel_external_specimen_and_timing_contract():
    r=observation(basedOn=[{'reference':'order'}]);o=order('24331-1')
    assert semantic_findings([r,o])==[]
    r['code']['coding'][0]['code']='13457-7'
    assert semantic_findings([r,o])==[]
    o['authoredOn']='2025-02-01T00:00:00Z'
    assert semantic_findings([r,o])
    r['basedOn']=[{'type':'ServiceRequest','identifier':{'system':'external','value':'order'}}]
    r['specimen']={'type':'Specimen','identifier':{'system':'external','value':'cervical-looking-name'}}
    assert semantic_findings([r])==[] # identifier is not a specimen type
    r['specimen']={'reference':'specimen'}
    specimen={'resourceType':'Specimen','_fullUrl':'specimen','type':{'text':'Cervical specimen'},
              'collection':{'collectedDateTime':'2025-01-01T12:00:00Z'}}
    assert semantic_findings([r,specimen])


@pytest.mark.parametrize('code',['24606-6','225337009','386661006','160245001'])
def test_placeholder_families_cannot_claim_result_depth(code):
    r=observation(code);r.pop('valueQuantity');r['valueString']='Synthetic result for something'
    assert not meaningful_result(r)
    assert any(x['finding']=='F08' for x in semantic_findings([r]))
    r['valueString']=CONTRACT['narrativeResults'][code]
    assert meaningful_result(r)
    assert not semantic_findings([r])
    r.pop('valueString');r['dataAbsentReason']={'coding':[{'code':'unknown'}]}
    assert not meaningful_result(r)
    assert not semantic_findings([r])


def test_unavailable_results_do_not_inflate_signatures():
    from synthetic_population.variability_v4 import normalized_signature
    core=order('24331-1');unknown=observation();unknown.pop('valueQuantity')
    unknown['dataAbsentReason']={'coding':[{'code':'unknown'}]}
    for view in ('semantic','role-topology'):
        assert normalized_signature([core],view)==normalized_signature([core,unknown],view)


def test_all_configured_branches_have_explicit_disposition():
    from synthetic_population.catalogs import load_catalog
    templates=load_catalog(SPEC['remediation']['pathwayTemplateCatalog'])
    for archetype,variants in templates['archetypes'].items():
        for variant in variants:
            for branch in variant['secondaryBranches']:
                disposition=CONTRACT['branchDispositions'][archetype+'/'+variant['id']+'/'+branch]
                assert disposition['status'] in {'observable-predicate','unsupported-no-coverage-credit'}
                assert disposition['reason']


def test_known_episode_mismatch_is_separate_from_reference_closure():
    r=observation(basedOn=[{'reference':'order'}],encounter={'reference':'visit-a'})
    o=order('24331-1');o['encounter']={'reference':'visit-b'}
    visits=[{'resourceType':'Encounter','_fullUrl':key} for key in ('visit-a','visit-b')]
    assert any('encounters/episodes' in f['reason'] for f in semantic_findings([r,o,*visits]))


def test_class_switch_closes_all_preexisting_same_ingredient_contexts():
    from synthetic_population.medication_safety import medication_findings
    path=build_patient_pathway(SPEC,2,'medication-intensive','dose-change','1980-01-01',
        SPEC['remediation']['pathwayTemplateCatalog'],SPEC['remediation']['clinicalContentCatalog'],
        secondary_trait='medication-exposure')
    assert path.secondary_branch=='switch-class'
    capture,_=project_capture(path,SPEC['remediation']['capturePolicyCatalog'],SPEC['seed'])
    rows=[]
    for source,dispositions in capture.items():
        projected,_=project_pathway(SPEC,{'id':source},2,'patient',path,dispositions)
        rows.extend({**r,'_source':source,'_fullUrl':u} for u,r in projected)
    assert not medication_findings(rows,SPEC)
    assert material_branch_proof('switch-class',rows,'1980-01-01')


def test_missing_or_invalid_dose_units_fail_closed():
    from synthetic_population.medication_safety import dosage_for,regimen_catalog
    resource={'resourceType':'MedicationRequest','dosageInstruction':[dosage_for(regimen_catalog(SPEC)['9997'])]}
    for field in ('code','unit'):
        broken=deepcopy(resource);broken['dosageInstruction'][0]['doseAndRate'][0]['doseQuantity'][field]=None
        assert normalized_dose_mg(broken) is None


@pytest.mark.parametrize('value',[{}, {'value':None}, {'value':float('nan')}, {'value':True}])
def test_empty_or_nonfinite_quantity_is_not_meaningful(value):
    resource=observation(valueQuantity=value)
    assert not meaningful_result(resource)
    assert semantic_findings([resource])


def test_pulse_oximetry_retains_required_fhir_oxygen_saturation_coding():
    _,rows,_=fixture('pediatric','respiratory-acute','oxygen-saturation')
    observations=[r for r in rows if r['resourceType']=='Observation' and
                  '59408-5' in {c['code'] for c in r['code']['coding']}]
    assert observations
    assert all('2708-6' in {c['code'] for c in r['code']['coding']} for r in observations)


@pytest.mark.parametrize('mutation',['episode','collection-before-order'])
def test_specimen_must_match_order_episode_and_chronology(mutation):
    r=observation(basedOn=[{'reference':'order'}],encounter={'reference':'visit-a'},specimen={'reference':'specimen'})
    o=order('24331-1');o['encounter']={'reference':'visit-a'}
    specimen={'resourceType':'Specimen','_source':'hmo','_fullUrl':'specimen',
              'subject':{'reference':'patient'},'encounter':{'reference':'visit-a'},
              'type':{'text':'Blood'},'collection':{'collectedDateTime':'2025-01-01T13:00:00Z'}}
    assert not semantic_findings([r,o,specimen])
    if mutation=='episode':specimen['encounter']['reference']='visit-b'
    else:specimen['collection']['collectedDateTime']='2024-12-01T12:00:00Z'
    assert any(f['finding']=='F11' for f in semantic_findings([r,o,specimen]))


@pytest.mark.parametrize('archetype,variant,branch',[
    ('generally-healthy','preventive-screening','glucose-screen'),
    ('pediatric','respiratory-acute','oxygen-saturation'),
    ('medication-intensive','monitoring-result','glucose-monitor'),
    ('rare-complex','rare-metabolic','glucose-monitor'),
    ('medication-intensive','monitoring-result','cbc-monitor'),
    ('oncology','toxicity-monitoring','hepatic-toxicity'),
    ('oncology','toxicity-monitoring','hematologic-toxicity'),
    ('oncology','hematologic-treatment','cbc-monitoring'),
])
@pytest.mark.parametrize('mutation',['historical','different-episode'])
def test_branch_cannot_borrow_unrelated_result(archetype,variant,branch,mutation):
    _,rows,birth=fixture(archetype,variant,branch)
    assert material_branch_proof(branch,rows,birth)
    code=CONTRACT['branchTests'][branch][0]
    for r in rows:
        if r['resourceType'] not in {'Observation','ServiceRequest'} or code not in {c['code'] for c in r.get('code',{}).get('coding',[])}:continue
        if mutation=='historical':r['effectiveDateTime' if r['resourceType']=='Observation' else 'authoredOn']='2010-01-01T12:00:00Z'
        else:r['encounter']={'reference':'unrelated-episode'}
    assert not material_branch_proof(branch,rows,birth)
