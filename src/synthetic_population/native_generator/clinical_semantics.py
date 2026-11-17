"""Observable result content and semantic references, independent of labels."""
from __future__ import annotations
from collections import defaultdict
from math import isfinite
from .catalogs import load_catalog
from .clinical_state import codes
from .temporal import occurrence_from_resource

CONTRACT_PATH = 'tools/synthetic_population/data/observable-clinical-contract.v1.0.0.json'
CONTRACT = load_catalog(CONTRACT_PATH)


def order_compatible(result_code: str, order_code: str) -> bool:
    return result_code == order_code or result_code in CONTRACT['panelMembers'].get(order_code, [])


def meaningful_result(resource: dict) -> bool:
    if resource.get('resourceType') != 'Observation':
        return True
    if resource.get('status') not in {'final','amended','corrected'} or resource.get('dataAbsentReason'):
        return False
    text = resource.get('valueString', '').strip()
    if text and any(token in text.lower() for token in ('synthetic result for', 'placeholder', 'result pending')):
        return False
    def has_value(node):
        quantity = node.get('valueQuantity',{}).get('value')
        if isinstance(quantity,(int,float)) and not isinstance(quantity,bool) and isfinite(quantity):
            return True
        if any(row.get('system') and row.get('code') for row in node.get('valueCodeableConcept',{}).get('coding',[])):
            return True
        if isinstance(node.get('valueBoolean'),bool):
            return True
        if isinstance(node.get('valueInteger'),int) and not isinstance(node['valueInteger'],bool):
            return True
        return False
    return bool(text or has_value(resource) or any(has_value(component) for component in resource.get('component', [])))


def semantic_findings(resources: list[dict]) -> list[dict]:
    by_url = {r.get('_fullUrl'): r for r in resources if r.get('_fullUrl')}
    by_identifier = {(r.get('_source'), i.get('system'), i.get('value')):r for r in resources
                     for i in r.get('identifier',[]) if i.get('value')}
    def resolve(reference, source):
        identifier=reference.get('identifier',{})
        return by_url.get(reference.get('reference')) or by_identifier.get((source,identifier.get('system'),identifier.get('value')))
    def context_identity(resource):
        context=resource.get('encounter',{})
        resolved=resolve(context,resource.get('_source'))
        if resolved:return resolved.get('_fullUrl')
        identifier=context.get('identifier',{})
        return context.get('reference') or ((identifier.get('system'),identifier['value']) if identifier.get('value') else None)
    findings = []
    def fail(r, finding, reason):
        findings.append({'finding':finding,'resource':r.get('_fullUrl',r.get('id')),'reason':reason})
    for r in resources:
        if r.get('resourceType') != 'Observation':
            continue
        if r.get('dataAbsentReason') and any(key.startswith('value') for key in r):
            fail(r,'F08','unavailable result also carries a value')
        if r.get('status') in {'final','amended','corrected'} and not meaningful_result(r) and not r.get('dataAbsentReason'):
            fail(r,'F08','final result is empty or placeholder content')
        for reference in r.get('basedOn', []):
            target = resolve(reference, r.get('_source'))
            if target is None:
                # Identifier-only context conveys an unavailable order, never a
                # claim about the order's analyte or specimen.
                continue
            if target.get('resourceType') != 'ServiceRequest' or not any(order_compatible(a,b) for a in codes(r) for b in codes(target)):
                fail(r,'F11','result and local order have incompatible analytes/panel membership')
            result_context,order_context=context_identity(r),context_identity(target)
            if result_context and order_context and result_context != order_context:
                fail(r,'F11','result and order belong to different captured encounters/episodes')
            at, ordered = occurrence_from_resource(r), occurrence_from_resource(target)
            if at and ordered and at.interval.start < ordered.interval.start:
                fail(r,'F11','result precedes its order')
            if r.get('_source') and target.get('_source') != r['_source']:
                fail(r,'F11','local order reference crosses source boundary')
            if r.get('subject') and target.get('subject') and r['subject'].get('reference') != target['subject'].get('reference'):
                fail(r,'F11','order and result have different subjects')
        ref = r.get('specimen', {})
        specimen = resolve(ref, r.get('_source'))
        if specimen is not None:
            specimen_context = context_identity(specimen)
            for peer in [r, *(resolve(ref, r.get('_source')) for ref in r.get('basedOn', []))]:
                if peer is None:continue
                peer_context = context_identity(peer)
                if specimen_context and peer_context and specimen_context != peer_context:
                    fail(r,'F11','specimen and result/order belong to different captured encounters/episodes')
                if peer.get('resourceType') == 'ServiceRequest':
                    ordered, collected = occurrence_from_resource(peer), occurrence_from_resource(specimen)
                    if ordered and collected and collected.interval.start < ordered.interval.start:
                        fail(r,'F11','specimen collection precedes the linked order')
            known = codes(specimen,'type')
            text = specimen.get('type',{}).get('text','').lower()
            cervical = 'cervical' in text or bool(known & {'119395005','258524009'})
            blood = bool(known & {'119297000','119364003','119361006'}) or any(word in text for word in ('blood','serum','plasma'))
            urine = bool(known & {'122575003'}) or 'urine' in text
            blood_analytes = {'2093-3','2085-9','13457-7','718-7','58410-2','2160-0','6298-4','2345-7','1742-6'}
            if urine and codes(r) & blood_analytes:
                fail(r,'F11','urine specimen is incompatible with this blood analyte')
            if specimen.get('_source') and specimen.get('_source') != r.get('_source'):
                fail(r,'F11','local specimen reference crosses source boundary')
            if r.get('subject') and specimen.get('subject') and r['subject'].get('reference') != specimen['subject'].get('reference'):
                fail(r,'F11','specimen and result have different subjects')
            if (cervical and '82675-0' not in codes(r)) or (blood and '82675-0' in codes(r)):
                fail(r,'F11','known specimen type is incompatible with the analyte')
            at, collected = occurrence_from_resource(r), occurrence_from_resource(specimen)
            if at and collected and at.interval.start < collected.interval.start:
                fail(r,'F11','result precedes specimen collection')
    return findings


def _prebirth_fixture_accounting(resources, cohort, spec, allocation):
    """Prove exact declared timestamp mutations against a clean local graph.

    This audit may use the deterministic fixture contract; the blind detector
    still receives observable resources only. Tags alone never grant an exemption.
    """
    import copy
    from .bundle import _clinical_resources, _problem_for_resource, demographic_truth, full_url
    from .utils import sha256_json
    index = int(cohort.removeprefix("cohort-"))
    problems = allocation.get(index, [])
    intended = [p for p in problems if p.get("kind") == "timestamp-order" and p.get("variant") == "pre-birth-event"]
    if not intended:
        return {}, []
    accepted, rejected = {}, []
    truth = demographic_truth(spec, index, {})
    by_url = {r.get("_fullUrl"): r for r in resources}
    expected_reasons = {"result precedes its order", "result precedes specimen collection"}
    for organization in spec["organizations"]:
        source = organization["id"]
        selected, _ = _problem_for_resource(problems, source, "Observation")
        for problem in intended:
            if problem["problemId"] not in selected:
                continue
            url = full_url(spec["datasetVersion"], source, cohort, "Observation", "primary")
            dirty = by_url.get(url)
            patient_url = full_url(spec["datasetVersion"], source, cohort, "Patient")
            kwargs = {"full_capture": True, "gender": truth["gender"]}
            expected = dict(_clinical_resources(spec, organization, index, patient_url, problems, truth["birthDate"], **kwargs))[url]
            clean = dict(_clinical_resources(spec, organization, index, patient_url, [], truth["birthDate"], fixture_context=True, **kwargs))[url]
            fields = {k for k in set(expected) | set(clean) if expected.get(k) != clean.get(k)}
            tag_systems = {spec["metadata"][key] for key in ("problemMarkerSystem", "problemIdSystem", "problemKindSystem")}
            def without_attribution(meta):
                result = copy.deepcopy(meta)
                result["tag"] = [tag for tag in result.get("tag", []) if tag.get("system") not in tag_systems]
                return result
            actual = {k: v for k, v in (dirty or {}).items() if not k.startswith("_")}
            proved = (sum(r.get("_fullUrl") == url for r in resources) == 1
                      and dirty is not None and dirty.get("_source") == source and actual == expected
                      and fields == {"effectiveDateTime", "issued", "meta"}
                      and without_attribution(expected["meta"]) == clean["meta"])
            graph = [({**clean, **{k: v for k, v in r.items() if k.startswith("_")}} if r.get("_fullUrl") == url else r) for r in resources]
            clean_findings = [f for f in semantic_findings(graph) if f["resource"] == url]
            raw = [f for f in semantic_findings(resources) if f["resource"] == url]
            proved = proved and not clean_findings and bool(raw) and all(
                f["finding"] == "F11" and f["reason"] in expected_reasons for f in raw)
            if proved:
                accepted[url] = {"problemId": problem["problemId"], "variant": "pre-birth-event",
                                 "source": source, "fullUrl": url, "dirtySha256": sha256_json(expected),
                                 "cleanSha256": sha256_json(clean), "cleanLocalGraphStatus": "pass",
                                 "allowedChangedFields": sorted(fields)}
            else:
                rejected.append({"finding": "F11", "resource": url,
                                 "reason": "declared pre-birth fixture fails exact mutation/clean local graph proof"})
    return accepted, rejected


def population_semantic_audit(bundles, spec):
    from .observable import patient_graphs
    from .problems import allocation_table
    graphs,_ = patient_graphs(bundles,spec)
    allocation = allocation_table(spec) if spec.get("remediation") else {}
    findings, raw_findings, intentional_findings = [], [], []
    summaries = defaultdict(lambda:{'people':set(),'facts':set(),'sourceCopies':0})
    for cohort,resources in graphs.items():
        raw = semantic_findings(resources)
        accepted, rejected = _prebirth_fixture_accounting(resources, cohort, spec, allocation)
        raw_findings += [{**row, 'cohortId':cohort} for row in raw]
        for row in raw:
            identity = accepted.get(row['resource'])
            if identity:
                intentional_findings.append({**row, 'cohortId':cohort, 'fixtureProof':identity})
            else:
                findings.append({**row, 'cohortId':cohort})
        findings += [{**row, 'cohortId':cohort} for row in rejected]
        for r in resources:
            if r.get('resourceType')!='Observation':continue
            category='meaningful' if meaningful_result(r) else 'unavailable-or-placeholder'
            row=summaries[category];row['people'].add(cohort);row['sourceCopies']+=1
            key=next((i['value'] for i in r.get('identifier',[]) if i.get('system')=='urn:hdp-test-bundle:clinical-event-key'),r.get('_fullUrl'))
            row['facts'].add((cohort,key))
    return {'status':'fail' if findings else 'pass','findings':findings,
            'rawStatus':'fail' if raw_findings else 'pass', 'rawFindings':raw_findings,
            'knownIntentionalFindings':intentional_findings,
            'resultContent':{key:{'people':len(row['people']),'facts':len(row['facts']),'sourceCopies':row['sourceCopies']} for key,row in sorted(summaries.items())}}
