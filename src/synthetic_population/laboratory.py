"""Final laboratory projection and independent F14/F15 observable checks.

Culture detection alone makes no infection, susceptibility or treatment claim.
"""
from __future__ import annotations

from html import escape
import json
import re
from xml.etree import ElementTree as ET


def is_laboratory(r):
    return r.get('resourceType') == 'Observation' and (
        any(c.get('code') in {'LAB', 'laboratory', '108252007'}
            for cat in r.get('category', []) for c in cat.get('coding', []))
        or any(p.endswith('il-hdp-observation-lab') for p in r.get('meta', {}).get('profile', [])))


def is_culture(r):
    return any(c.get('system') == 'http://loinc.org' and c.get('code') == '630-4'
               for c in r.get('code', {}).get('coding', []))


def finalize_laboratory(r):
    """Finalize generated display only; never normalize declared quantity noise.

    Called after clinical mutation and before answer-key capture/export. Structured
    culture metadata has no supported specimen-specific baseline in this generator:
    omit interpretation, range and method rather than assert normal or infection.
    """
    if not is_laboratory(r):
        return
    if is_culture(r) and ('valueCodeableConcept' in r or 'valueString' in r):
        for key in ('valueQuantity', 'referenceRange', 'interpretation', 'method', 'bodySite'):
            r.pop(key, None)
    # Interpretation follows final values, including converted-unit copies;
    # quantity noise itself is retained and independently dispositioned.
    q = r.get('valueQuantity', {})
    ranges = r.get('referenceRange', [])
    conversions = {'mg/dL': 1, 'umol/L': 1 / 88.4, 'g/dL': 1, 'g/L': .1, 'mmol/L': 1, 'meq/L': 1}
    if q and ranges:
        low, high = ranges[0].get('low', {}), ranges[0].get('high', {})
        unit, reference_unit = q.get('code'), low.get('code')
        compatible = unit == reference_unit or {unit, reference_unit} in ({'mg/dL', 'umol/L'}, {'g/dL', 'g/L'}, {'mmol/L', 'meq/L'})
        if compatible and high.get('code') == reference_unit and all(isinstance(n.get('value'), (int, float)) for n in (q, low, high)):
            value = q['value'] * conversions.get(unit, 1) / conversions.get(reference_unit, 1)
            code = 'H' if value > high['value'] else 'L' if value < low['value'] else 'N'
            r['interpretation'] = [{'coding': [{'system': 'http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation', 'code': code}]}]
    fields = ['code', 'status'] + sorted(k for k in r if k.startswith('value'))
    fields += [k for k in ('component', 'dataAbsentReason', 'interpretation', 'referenceRange', 'method') if k in r]
    # JSON strings preserve exact coding labels and numeric precision, safely
    # escaped inside XHTML. Field labels support independently parsed inspection.
    paragraphs = ''.join('<p>' + escape(k + ': ' + json.dumps(r[k], ensure_ascii=False, sort_keys=True)) + '</p>' for k in fields)
    r['text'] = {'status': 'generated', 'div': '<div xmlns="http://www.w3.org/1999/xhtml">' + paragraphs + '</div>'}


def laboratory_findings(r):
    """Parse rendered assertions; compare values, then independently check metadata.

    No renderer call or renderer-byte equality is used. Historical simple numeric
    prose is parsed independently; unsupported forms are actionable, never a pass.
    """
    if not is_laboratory(r):
        return []
    found = []
    def fail(code, field, reason):
        found.append({'finding': code, 'resource': r.get('_fullUrl', r.get('id')),
                      'source': r.get('_source'), 'field': field, 'reason': reason})
    raw = r.get('text', {}).get('div')
    try:
        root = ET.fromstring(raw or '')
        text = ' '.join(root.itertext())
    except ET.ParseError:
        fail('F14', 'text.div', 'missing or malformed required laboratory narrative')
        root, text = None, ''
    assertions = {}
    unsupported = bool(root is not None and root.text and root.text.strip())
    if root is not None:
        for element in root:
            line = ''.join(element.itertext())
            key, sep, value = line.partition(': ')
            if element.tail and element.tail.strip():
                unsupported = True
            if element.tag != '{http://www.w3.org/1999/xhtml}p' or list(element):
                unsupported = True
            if sep:
                try:
                    if key in assertions:
                        fail('F14', 'text.div', 'duplicate narrative field assertion')
                    assertions[key] = json.loads(value)
                except json.JSONDecodeError:
                    unsupported = True
            else:
                unsupported = True
        if assertions:
            if unsupported:
                fail('F14', 'text.div', 'unsupported extra narrative content beside structured assertions')
            expected_keys = {'code', 'status'} | {k for k in r if k.startswith('value')}
            expected_keys |= {k for k in ('component', 'dataAbsentReason', 'interpretation', 'referenceRange', 'method') if k in r}
            for key in sorted(expected_keys | set(assertions)):
                if key not in assertions or key not in r or assertions[key] != r[key]:
                    fail('F14', 'text.div/' + key, 'narrative assertion disagrees with final structured field')
        else:
            match = re.fullmatch(r'Synthetic (.+?) result: (-?\d+(?:\.\d+)?) (.+)\.', text.strip())
            q = r.get('valueQuantity', {})
            labels = {c.get('display') for c in r.get('code', {}).get('coding', [])}
            if not match or match[1] not in labels or float(match[2]) != q.get('value') or match[3] != q.get('unit'):
                fail('F14', 'text.div', 'unsupported narrative or stale analyte/result/unit prose')
            else:
                fail('F14', 'text.div', 'legacy narrative omits final status and metadata')
    q = r.get('valueQuantity', {})
    analytes = {c.get('code') for c in r.get('code', {}).get('coding', [])}
    supported_units = {'2160-0': {'mg/dL', 'umol/L'}, '718-7': {'g/dL', 'g/L'}, '6298-4': {'mmol/L', 'meq/L'}}
    if q:
        if q.get('unit') != q.get('code'):
            fail('F15', 'valueQuantity.unit', 'display unit disagrees with coded unit')
        if any(code in analytes and q.get('code') not in units for code, units in supported_units.items()):
            fail('F15', 'valueQuantity.code', 'unit is incompatible with the quantitative analyte')
        if any(c.get('code') != '396550006' for c in r.get('method', {}).get('coding', [])):
            fail('F15', 'method', 'unsupported method for generated quantitative blood analyte')
        method_text = r.get('method', {}).get('text')
        allowed_methods = {'2160-0': {'IDMS-traceable creatinine; adult CKD-EPI 2021'}, '5018-7': {'HIV-1 RNA nucleic acid amplification', 'NAA with probe detection'}}
        if method_text and not any(method_text in allowed_methods.get(code, set()) for code in analytes):
            fail('F15', 'method', 'method text is incompatible with generated analyte')
        for interval in r.get('referenceRange', []):
            low, high = interval.get('low', {}), interval.get('high', {})
            if not low or not high:
                continue
            if any(n.get('unit') != n.get('code') for n in (low, high)):
                fail('F15', 'referenceRange', 'displayed reference unit disagrees with coded unit')
            pairs = ({'mg/dL', 'umol/L'}, {'g/dL', 'g/L'}, {'mmol/L', 'meq/L'})
            units = {q.get('code'), low.get('code')}
            if low.get('code') != high.get('code') or (len(units) != 1 and units not in pairs):
                fail('F15', 'referenceRange', 'range units incompatible with final result')
                continue
            if low.get('value', 0) > high.get('value', 0):
                fail('F15', 'referenceRange', 'reference interval is reversed')
            factors = {'umol/L': 1 / 88.4, 'g/L': .1}
            value = q.get('value', 0) * factors.get(q.get('code'), 1) / factors.get(low.get('code'), 1)
            expected = 'H' if value > high.get('value', 0) else 'L' if value < low.get('value', 0) else 'N'
            actual = {c.get('code') for item in r.get('interpretation', []) for c in item.get('coding', [])}
            if actual and actual != {expected}:
                fail('F15', 'interpretation', 'interpretation disagrees with value and reference interval')
    if is_culture(r):
        if 'valueQuantity' in r:
            fail('F15', 'valueQuantity', 'qualitative culture carries a quantitative result')
        for key in ('referenceRange', 'interpretation', 'method', 'bodySite'):
            if key in r:
                fail('F15', key, 'culture metadata lacks a supported specimen/result context')
        value = r.get('valueCodeableConcept', {})
        if not value.get('coding') and not r.get('valueString') and not r.get('dataAbsentReason') and r.get('status') not in {'cancelled', 'entered-in-error'}:
            fail('F15', 'value[x]', 'culture lacks a supported qualitative or unavailable result')
    return found
