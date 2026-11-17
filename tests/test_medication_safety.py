import synthetic_population
"""Medication defects are checked from emitted clinical fields, without tags."""
from copy import deepcopy
from pathlib import Path
import pytest
from synthetic_population.config import load_spec, scale_spec
from synthetic_population.medication_safety import (
    dosage_for, regimen_catalog, normalized_dose_mg, medication_findings,
    dispense_ledger, population_medication_audit,
)

ROOT = Path(synthetic_population.__file__).resolve().parent
SPEC = load_spec(ROOT / 'data/specs/native.json')


def request(code='9997', key='rx', start='2025-01-01', end=None):
    return {'resourceType': 'MedicationRequest', '_fullUrl': key, 'status': 'active',
            'authoredOn': start, 'medicationCodeableConcept': {'coding': [
                {'system': 'http://www.nlm.nih.gov/research/umls/rxnorm', 'code': code}]},
            'dosageInstruction': [dosage_for(regimen_catalog(SPEC)[code], start=start, end=end)],
            'dispenseRequest': {'quantity': {'value': 30, 'system': 'http://snomed.info/sct', 'code': '428673006'},
                                'numberOfRepeatsAllowed': 1}}


def fill(rx, key, date, quantity=30, kind='FFC'):
    return {'resourceType': 'MedicationDispense', '_fullUrl': key, 'status': 'completed',
            'medicationCodeableConcept': deepcopy(rx['medicationCodeableConcept']),
            'authorizingPrescription': [{'reference': rx['_fullUrl']}],
            'whenHandedOver': date, 'quantity': {**rx['dispenseRequest']['quantity'], 'value': quantity},
            'type': {'coding': [{'code': kind}]}}


def test_mass_and_strength_are_not_interchangeable():
    rx = request()
    assert normalized_dose_mg(rx) == 25
    q = rx['dosageInstruction'][0]['doseAndRate'][0]['doseQuantity']
    q.update(value=12.5, unit='tablet', code='1')
    assert normalized_dose_mg(rx) is None
    assert any(r['finding'] == 'F01' for r in medication_findings([rx], SPEC))
    q.update(value=1)
    rx['medicationCodeableConcept']['text'] = 'Spironolactone 25 mg per tablet'
    assert normalized_dose_mg(rx) == 25
    assert not medication_findings([rx], SPEC)
    q.update(value=.025, unit='g', code='g')
    assert normalized_dose_mg(rx) == 25
    rx['dosageInstruction'][0]['text'] = 'Take 12.5 mg daily'
    assert medication_findings([rx], SPEC)


def test_ingredient_change_requires_complete_regimen():
    rx = request('11289')
    rx['medicationCodeableConcept']['coding'][0]['code'] = '1191'
    assert any(r['finding'] == 'F09' for r in medication_findings([rx], SPEC))
    rx['dosageInstruction'] = [dosage_for(regimen_catalog(SPEC)['1191'])]
    assert not medication_findings([rx], SPEC)


def test_overlap_switch_and_untagged_renal_background():
    ace, arb = request('29046'), request('52175', 'arb', '2025-03-01')
    assert any(r['finding'] == 'F10' for r in medication_findings([ace, arb], SPEC))
    ace['status'] = 'stopped'
    # Status alone cannot conceal an unknown cessation date.
    assert medication_findings([ace, arb], SPEC)
    ace['dosageInstruction'][0]['timing']['repeat']['boundsPeriod']['end'] = arb['authoredOn']
    arb['priorPrescription'] = {'reference': 'rx'}
    assert not medication_findings([ace, arb], SPEC)
    renal = {'resourceType': 'Condition', 'code': {'coding': [{'code': '431857002'}]},
             'onsetDateTime': '2025-02-01'}
    assert any(r['finding'] == 'F13' for r in medication_findings([renal, request('6809')], SPEC))
    assert not medication_findings([renal, request('1100699')], SPEC)


@pytest.mark.parametrize('repeats,count', [(0,2),(1,3)])
def test_every_fill_consumes_authorization(repeats,count):
    rx = request(); rx['dispenseRequest']['numberOfRepeatsAllowed'] = repeats
    fills = [fill(rx,str(i),f'2025-0{i+1}-02',kind='FFC' if i == 0 else 'RFC') for i in range(count)]
    assert dispense_ledger([rx,*fills])['status'] == 'fail'
    rx['dispenseRequest']['numberOfRepeatsAllowed'] = count - 1
    assert dispense_ledger([rx,*fills])['status'] == 'pass'


def test_partial_duplicate_renewal_and_validity():
    rx = request(); rx['dispenseRequest']['numberOfRepeatsAllowed'] = 0
    a,b = fill(rx,'a','2025-01-02',10,'FFP'), fill(rx,'b','2025-01-05',20,'FFP')
    a['identifier'] = [{'system':'event','value':'a'}]
    duplicate = deepcopy(a); duplicate['_fullUrl'] = 'other-source-a'
    assert dispense_ledger([rx,a,b,duplicate])['uniqueFillCount'] == 2
    assert dispense_ledger([rx,a,b,duplicate])['status'] == 'pass'
    renewal = request(key='new',start='2025-02-01'); renewal['priorPrescription']={'reference':'rx'}
    c=fill(renewal,'c','2025-02-02')
    assert dispense_ledger([rx,a,b,renewal,c])['status'] == 'pass'
    rx['dispenseRequest']['validityPeriod'] = {'end':'2025-01-03'}
    assert dispense_ledger([rx,a,b])['status'] == 'fail'
    b['authorizingPrescription']=[{'reference':'outside'}]
    assert dispense_ledger([rx,b])['status'] == 'incomplete'


def test_bounded_generation_including_late_supplemental_paths():
    from synthetic_population.bundle import build_population
    spec=scale_spec(SPEC,30,'4.2.0-s21-test')
    built=build_population(spec,ROOT)
    audit=population_medication_audit(built['bundles'],spec)
    assert audit['findings'] == []
    assert all(row['status']=='pass' for row in audit['dispenseHistories'])


def test_unit_labels_and_conflicting_capture_authorizations_fail():
    rx=request();rx['dosageInstruction'][0]['doseAndRate'][0]['doseQuantity']['unit']='tablet'
    assert medication_findings([rx],SPEC)
    first,second=request(key='first'),request(key='second')
    a=fill(first,'a','2025-01-02');b=fill(second,'b','2025-01-02')
    for row in (a,b):row['identifier']=[{'system':'urn:hdp-test-bundle:clinical-event-key','value':'one-fill'}]
    assert dispense_ledger([first,second,a,b])['status']=='fail'


def test_native_audit_does_not_pass_incomplete_authorization():
    patient={'resourceType':'Patient','identifier':[{'system':SPEC['identity']['cohortIdentifierSystem'],'value':'cohort-000001'}]}
    rx=request();row=fill(rx,'fill','2025-01-02')
    row['subject']={'reference':'patient'}
    row['authorizingPrescription']=[{'reference':'external-rx'}]
    bundles={'hmo':{'entry':[{'fullUrl':'patient','resource':patient},{'fullUrl':'fill','resource':row}]}}
    result=population_medication_audit(bundles,SPEC)
    assert result['status']=='fail'
    assert any('incomplete' in finding['reason'] for finding in result['findings'])


@pytest.mark.parametrize('field,value', [('unit','mL'),('unit','capsule'),('unit','dose'),('system','http://snomed.info/sct')])
def test_all_dose_unit_representations_must_be_consistent(field,value):
    rx=request();rx['dosageInstruction'][0]['doseAndRate'][0]['doseQuantity'][field]=value
    assert any(row['finding']=='F01' for row in medication_findings([rx],SPEC))
