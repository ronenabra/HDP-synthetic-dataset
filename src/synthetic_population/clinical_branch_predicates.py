"""Independent FHIR proofs for the approved S14A clinical branches.

These predicates use resource content, timing and references. They never read
the declared secondary branch or generator tags.
"""
from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any

from .observable import codings
from .condition_coding import semantic_condition_codings


SNOMED = "http://snomed.info/sct"
LOINC = "http://loinc.org"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"


def _coded(resource: dict[str, Any], field: str, system: str, code: str) -> bool:
    observed = (semantic_condition_codings(resource)
                if field == "code" and resource.get("resourceType") == "Condition"
                else set(codings(resource.get(field))))
    return (system, code) in observed


def _at(resource: dict[str, Any]) -> datetime | None:
    value = next((resource[key] for key in (
        "whenHandedOver", "performedDateTime", "effectiveDateTime", "authoredOn",
        "date", "receivedTime", "dateAsserted", "onsetDateTime", "recordedDate",
    ) if resource.get(key)), None)
    if value is None and resource.get("period"):
        value = resource["period"].get("start")
    if value is None and resource.get("collection"):
        value = resource["collection"].get("collectedDateTime")
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def _days(left: dict[str, Any], right: dict[str, Any]) -> float:
    a, b = _at(left), _at(right)
    return (b - a).total_seconds() / 86400 if a and b else -1


def _age(birth_date: str, resource: dict[str, Any]) -> int:
    born, when = date.fromisoformat(birth_date), _at(resource).date()
    return when.year - born.year - ((when.month, when.day) < (born.month, born.day))


def _linked(resource: dict[str, Any], field: str, target: dict[str, Any]) -> bool:
    target_url = target.get("_fullUrl")
    target_keys = {row.get("value") for row in target.get("identifier", [])
                   if row.get("system") == "urn:hdp-test-bundle:clinical-event-key"}
    values = resource.get(field, [])
    if isinstance(values, dict):
        values = [values]
    return any((target_url is not None and row.get("reference") == target_url) or
               row.get("identifier", {}).get("value") in target_keys
               for row in values if isinstance(row, dict))


def _rows(resources: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [row for row in resources if row.get("resourceType") == kind]


def _event_keys(resource: dict[str, Any]) -> set[str]:
    return {row.get("value") for row in resource.get("identifier", [])
            if row.get("system") == "urn:hdp-test-bundle:clinical-event-key"}


def _same_source(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.get("_source") == right.get("_source")


def _components(resource: dict[str, Any]) -> set[str]:
    return {row.get("code", {}).get("text") for row in resource.get("component", [])
            if row.get("valueBoolean") is True}


def _screening(branch: str, resources: list[dict[str, Any]], birth_date: str) -> bool:
    hpv = branch == "cervical-screen" or branch == "result-review" and any(
        _coded(row, "code", LOINC, "82675-0") for row in _rows(resources, "Observation"))
    system, order_code, result_code = ((LOINC, "82675-0", "82675-0") if hpv
                                       else (SNOMED, "71651007", "24606-6"))
    orders = [row for row in _rows(resources, "ServiceRequest")
              if _coded(row, "code", system, order_code) and row.get("status") == "completed"]
    results = [row for row in _rows(resources, "Observation")
               if _coded(row, "code", LOINC, result_code) and row.get("status") == "final"]
    assessments = [row for row in _rows(resources, "Observation")
                   if _coded(row, "code", SNOMED, "225337009")]
    for order in orders:
        if not (30 <= _age(birth_date, order) <= 54 if hpv else 50 <= _age(birth_date, order) <= 74):
            continue
        minimum_interval = 1095 if hpv else 730
        if any(0 < _days(prior, order) < minimum_interval for prior in results):
            continue
        required = ({"Asymptomatic at screening", "No prior abnormal screening result or relevant treatment",
                     "Cervix present", "No cervical screening in prior three years"} if hpv else
                    {"Asymptomatic at screening", "No routine mammogram in prior two years", "Average breast cancer risk"})
        if not any(required <= _components(row) and 0 <= _days(row, order) <= 30 for row in assessments):
            continue
        for result in results:
            if not 0 <= _days(order, result) <= 42 or not _linked(result, "basedOn", order):
                continue
            if hpv:
                specimens = [row for row in _rows(resources, "Specimen")
                             if row.get("type", {}).get("text") == "Cervical specimen"
                             and row.get("collection", {}).get("bodySite", {}).get("text") == "Cervix"
                             and 0 <= _days(order, row) and 7 <= _days(row, result) <= 28
                             and _linked(result, "specimen", row)]
                if not specimens or not _coded(result, "valueCodeableConcept", SNOMED, "260385009") \
                        or result.get("method", {}).get("text") != "NAA with probe detection" \
                        or "valueQuantity" in result:
                    continue
            if branch == "result-review":
                reviews = [row for row in _rows(resources, "Encounter")
                           if 7 <= _days(result, row) <= 42 and _linked(row, "reasonReference", result)]
                if not reviews or any(_at(other) > _at(result) for other in orders):
                    continue
            elif any(7 <= _days(result, row) <= 42 and _linked(row, "reasonReference", result)
                     for row in _rows(resources, "Encounter")):
                continue
            return True
    return False


def _appendectomy(branch: str, resources: list[dict[str, Any]], birth_date: str) -> bool:
    code = "80146002" if branch == "open" else "174041007"
    def performed_at_owner(row: dict[str, Any]) -> bool:
        source = row.get("_source", "")
        actors = [item.get("actor", {}) for item in row.get("performer", [])]
        return (source.startswith("hospital")
                and row.get("meta", {}).get("source") == f"urn:hdp-test-bundle:source:{source}"
                and ("urn:hdp-test-bundle:synthetic:capture-class", f"{source}-performed")
                in codings(row.get("meta", {}).get("tag", []))
                and any(actor.get("reference", "").startswith("urn:uuid:")
                        and actor.get("identifier", {}).get("system", "").endswith(f":{source}:practitioner")
                        and actor.get("identifier", {}).get("value") for actor in actors))
    procedures = [row for row in _rows(resources, "Procedure")
                  if _coded(row, "code", SNOMED, code) and row.get("status") == "completed"
                  and performed_at_owner(row)
                  and (branch != "open" or row.get("code", {}).get("text") == "Open appendectomy")]
    emergencies = [row for row in _rows(resources, "Encounter")
                   if row.get("class", {}).get("code") == "EMER"
                   and _coded(row, "reasonCode", SNOMED, "74400008")]
    diagnoses = [row for row in _rows(resources, "Condition")
                 if _coded(row, "code", SNOMED, "74400008")]
    for procedure in procedures:
        if _age(birth_date, procedure) < 18 or not any(0 <= _days(row, procedure) <= 2 for row in emergencies):
            continue
        if not any(0 <= _days(row, procedure) <= 2 and _linked(procedure, "reasonReference", row)
                   for row in diagnoses):
            continue
        recoveries = [row for row in _rows(resources, "Encounter")
                      if row.get("class", {}).get("code") == ("AMB" if branch == "short-stay" else "IMP")
                      and row.get("_source", "").startswith("hospital")
                      and 0 <= _days(procedure, row) <= 1 and _linked(row, "reasonReference", procedure)]
        for recovery in recoveries:
            discharges = [row for row in _rows(resources, "DocumentReference")
                          if _coded(row, "type", SNOMED, "373942005")
                          and _linked(row.get("context", {}), "encounter", recovery)]
            for discharge in discharges:
                duration = _days(procedure, discharge)
                if not ((0 <= duration < 1) if branch == "short-stay" else
                        (2 <= duration <= 4.5) if branch == "open" else (1 <= duration <= 2.5)):
                    continue
                if branch == "short-stay" and not any(
                    row.get("_source") == "hmo" and row.get("class", {}).get("code") == "AMB"
                    and 3 <= _days(discharge, row) <= 7
                    and any(summary.get("_source") == "hmo" and summary is not procedure
                            and _coded(summary, "code", SNOMED, code)
                            and bool(_event_keys(summary) & _event_keys(procedure))
                            and _linked(row, "reasonReference", summary)
                            for summary in _rows(resources, "Procedure"))
                    for row in _rows(resources, "Encounter")
                ):
                    continue
                if branch == "open":
                    external_summaries = [row for row in _rows(resources, "Procedure")
                                          if row is procedure or (
                                              row.get("_source") == "hmo"
                                              and _coded(row, "code", SNOMED, code)
                                              and abs(_days(procedure, row)) < 1
                                          )]
                    if not any(
                        3 <= _days(discharge, row) <= 14
                        and any(_linked(row, "reasonReference", item) for item in external_summaries)
                        for row in _rows(resources, "Encounter")
                    ):
                        continue
                return True
    return False


def _metformin_regimen(resource: dict[str, Any]) -> tuple[int, int, int] | None:
    if not _coded(resource, "medicationCodeableConcept", RXNORM, "6809"):
        return None
    product = resource.get("medicationCodeableConcept", {}).get("text", "")
    strength = re.fullmatch(r"Metformin (\d+) mg per tablet", product)
    instructions = resource.get("dosageInstruction", [])
    if not strength or len(instructions) != 1:
        return None
    dosage = instructions[0]
    repeat = dosage.get("timing", {}).get("repeat", {})
    if repeat.get("period") != 1 or repeat.get("periodUnit") != "d":
        return None
    frequency = repeat.get("frequency")
    if not isinstance(frequency, int) or frequency < 1:
        return None
    doses = [item.get("doseQuantity", {}) for item in dosage.get("doseAndRate", [])]
    mg = [item.get("value") for item in doses if item.get("system") == "http://unitsofmeasure.org" and item.get("code") == "mg"]
    tablets = [item.get("value") for item in doses if item.get("system") == "http://unitsofmeasure.org" and item.get("code") == "1"]
    if len(mg) != 1 or len(tablets) != 1 or tablets[0] <= 0 or mg[0] != int(strength[1]) * tablets[0]:
        return None
    return int(strength[1]), tablets[0], frequency


def _refill(branch: str, resources: list[dict[str, Any]], birth_date: str) -> bool:
    from .medication_safety import dispense_ledger
    if dispense_ledger(resources)["status"] != "pass":
        return False
    diabetes = [row for row in _rows(resources, "Condition") if row.get("_source") == "hmo"
                and _coded(row, "code", SNOMED, "44054006")]
    if not diabetes:
        return False
    requests = [row for row in _rows(resources, "MedicationRequest")
                if row.get("_source") == "hmo" and row.get("status") in {"active", "completed"}
                and _metformin_regimen(row) is not None
                and any(_linked(row, "reasonReference", condition) for condition in diabetes)]
    candidates = [row for row in _rows(resources, "MedicationRequest")
                  if row.get("_source") == "hmo" and _metformin_regimen(row) is not None]
    if len(candidates) != len(requests):
        return False
    dispenses = [row for row in _rows(resources, "MedicationDispense")
                 if row.get("_source") == "hmo" and _metformin_regimen(row) is not None
                 and row.get("quantity", {}).get("value") == 30
                 and row.get("quantity", {}).get("code") == "428673006"
                 and row.get("daysSupply", {}).get("value") == 30
                 and row.get("daysSupply", {}).get("code") == "d"
                 and row.get("status") == "completed"
                 and any(_linked(row, "authorizingPrescription", request)
                         and 0 <= _days(request, row)
                         and _metformin_regimen(row) == _metformin_regimen(request)
                         and request.get("dispenseRequest", {}).get("quantity", {}).get("value") == 30
                         and request.get("dispenseRequest", {}).get("expectedSupplyDuration", {}).get("value") == 30
                         and 30 * _metformin_regimen(row)[1] * _metformin_regimen(row)[2] == 30
                         for request in requests)]
    dispenses.sort(key=_at)
    # A later valid pair must not conceal a broken renewal earlier in the history.
    ordered_requests = sorted(requests, key=_at)
    for prior, current in zip(ordered_requests, ordered_requests[1:]):
        if not (prior.get("status") == "completed" and current.get("status") == "active"
                and _linked(current, "priorPrescription", prior)
                and prior.get("statusReason", {}).get("text")
                and "exhausted" in current.get("statusReason", {}).get("text", "")
                and prior.get("dispenseRequest", {}).get("validityPeriod", {}).get("end") == current.get("authoredOn")):
            return False
    for first, second in zip(dispenses, dispenses[1:]):
        first_request = next((row for row in requests if _linked(first, "authorizingPrescription", row)), None)
        second_request = next((row for row in requests if _linked(second, "authorizingPrescription", row)), None)
        if first_request is None or second_request is None:
            continue
        if first_request is second_request:
            if (first_request["status"] != "active" or
                    first_request.get("dispenseRequest", {}).get("numberOfRepeatsAllowed", 0) < 1):
                continue
        elif not (
            first_request["status"] == "completed" and second_request["status"] == "active"
            and first_request.get("dispenseRequest", {}).get("numberOfRepeatsAllowed") == 0
            and second_request.get("dispenseRequest", {}).get("numberOfRepeatsAllowed", -1) >= 0
            and _linked(second_request, "priorPrescription", first_request)
            and first_request.get("statusReason", {}).get("text")
            and "exhausted" in second_request.get("statusReason", {}).get("text", "")
            and first_request.get("dispenseRequest", {}).get("validityPeriod", {}).get("end")
            == second_request.get("authoredOn")
            and 0 <= _days(first, second_request) <= _days(first, second)
        ):
            continue
        gap = _days(first, second)
        if not ((28 <= gap <= 30) if branch != "late-refill" else (37 <= gap <= 44)):
            continue
        if _age(birth_date, second) < 18:
            continue
        if branch == "late-refill":
            if not any(0 <= _days(second, row) <= 7 and row.get("resourceType") == "Encounter"
                       and any(_linked(statement, "context", row)
                               for statement in _rows(resources, "MedicationStatement"))
                       for row in resources):
                continue
        if branch == "pharmacist-review":
            if not any(0 <= _days(row, second) <= 7 and row.get("resourceType") == "Encounter"
                       and any(code.get("text") == "Pharmacy medication review" for code in row.get("reasonCode", []))
                       and any(_linked(statement, "context", row)
                               and _coded(statement, "medicationCodeableConcept", RXNORM, "6809")
                               for statement in _rows(resources, "MedicationStatement"))
                       for row in resources):
                continue
        if branch == "on-time-refill" and any(
            statement.get("resourceType") == "MedicationStatement" and
            statement.get("context") for statement in resources
        ):
            continue
        return True
    return False


def _diabetes(branch: str, resources: list[dict[str, Any]], birth_date: str) -> bool:
    if not any(_coded(row, "code", SNOMED, "44054006") and row.get("clinicalStatus", {}).get("coding", [{}])[0].get("code") == "active"
               for row in _rows(resources, "Condition")):
        return False
    results = sorted((row for row in _rows(resources, "Observation")
                      if _coded(row, "code", LOINC, "4548-4") and row.get("status") == "final"
                      and isinstance(row.get("valueQuantity", {}).get("value"), (int, float))), key=_at)
    pairs = [(a, b) for a in results for b in results if 150 <= _days(a, b) <= 210
             and a["valueQuantity"]["value"] != b["valueQuantity"]["value"]]
    if not pairs or _age(birth_date, pairs[-1][1]) < 18:
        return False
    if branch == "hba1c-stable":
        if any(row.get("code", {}).get("text") == "Diabetic eye assessment"
               for row in _rows(resources, "ServiceRequest")) or any(
            any(code.get("text") == "Diet and lifestyle counselling for type 2 diabetes"
                for code in row.get("reasonCode", [])) for row in _rows(resources, "Encounter")):
            return False
        return any(_days(latest, row) >= 0 and _linked(row, "reasonReference", latest)
                   for _, latest in pairs for row in _rows(resources, "Encounter"))
    if branch == "diet-review":
        encounters = _rows(resources, "Encounter")
        orders = [row for row in _rows(resources, "ServiceRequest")
                  if _coded(row, "code", SNOMED, "390906007")]
        return not any(row.get("code", {}).get("text") == "Diabetic eye assessment"
                       for row in _rows(resources, "ServiceRequest")) and any(
                   first.get("_source") == "hmo" and second.get("_source") == "hmo"
                   and 30 <= _days(first, second) <= 90 and _linked(first, "reasonReference", latest)
                   and any(_same_source(first, order) and _same_source(order, second)
                           and 0 <= _days(first, order) < 1
                           and _linked(order, "supportingInfo", latest)
                           and _linked(second, "basedOn", order)
                           for order in orders)
                   and any(_linked(second, "reasonReference", condition) for condition in _rows(resources, "Condition")
                           if _same_source(second, condition) and _coded(condition, "code", SNOMED, "44054006"))
                   and any(code.get("text") == "Diet and lifestyle counselling for type 2 diabetes"
                           for code in first.get("reasonCode", []))
                   for _, latest in pairs for first in encounters for second in encounters)
    referrals = [row for row in _rows(resources, "ServiceRequest")
                 if row.get("code", {}).get("text") == "Diabetic eye assessment"
                 and row.get("performerType", {}).get("text") == "Ophthalmology"
                 and row.get("status") in {"active", "completed"}]
    return any(any(0 <= _days(latest, referral) <= 30
                   and _same_source(latest, referral) and _linked(referral, "supportingInfo", latest)
                   for _, latest in pairs)
               and any(_same_source(referral, condition) and _linked(referral, "reasonReference", condition)
                       for condition in _rows(resources, "Condition") if _coded(condition, "code", SNOMED, "44054006")) and
               (referral["status"] == "active" or any(
                   14 <= _days(referral, review) <= 90
                   and (_linked(review, "basedOn", referral) or any(
                       _event_keys(referral) & _event_keys(local_referral)
                       and _same_source(review, local_referral)
                       and _linked(review, "basedOn", local_referral)
                       for local_referral in referrals
                   ))
                   for review in _rows(resources, "Encounter"))) for referral in referrals)


def branch_proof(branch: str, resources: list[dict[str, Any]], birth_date: str) -> bool:
    """Return whether emitted resources establish one S14A branch contract."""
    if branch in {"mammography", "cervical-screen", "result-review"}:
        return _screening(branch, resources, birth_date)
    if branch in {"laparoscopic", "open", "short-stay"}:
        return _appendectomy(branch, resources, birth_date)
    if branch in {"on-time-refill", "late-refill", "pharmacist-review"}:
        return _refill(branch, resources, birth_date)
    if branch in {"hba1c-stable", "diet-review", "retinal-referral"}:
        return _diabetes(branch, resources, birth_date)
    if branch == "nonpregnant-follow-up":
        return any("No current pregnancy reported" in _components(row)
                   for row in _rows(resources, "Observation")) and any(
            any(code.get("text") == "Contraception and preconception review" for code in row.get("reasonCode", []))
            for row in _rows(resources, "Encounter"))
    if branch == "lisinopril-potassium-exposure":
        conditions = [row for row in _rows(resources, "Condition")
                      if _coded(row, "code", SNOMED, "38341003")]
        requests = [row for row in _rows(resources, "MedicationRequest")
                    if _coded(row, "medicationCodeableConcept", RXNORM, "29046") and row.get("status") == "active"]
        potassium = [row for row in _rows(resources, "Observation")
                     if _coded(row, "code", LOINC, "6298-4")]
        return any(_age(birth_date, observation) >= 18 and
                   any(_days(condition, request) >= 0 and _days(request, observation) >= 0
                       for condition in conditions for request in requests)
                   for observation in potassium)
    raise ValueError(f"unknown S14A branch {branch}")


LEGACY_BRANCHES = {
    'mammography','cervical-screen','result-review','laparoscopic','open','short-stay',
    'on-time-refill','late-refill','pharmacist-review','hba1c-stable','diet-review',
    'retinal-referral','nonpregnant-follow-up','lisinopril-potassium-exposure',
}


def _same_encounter(left, right, resources):
    def identity(resource):
        ref=resource.get('encounter',{})
        target=next((r for r in resources if r.get('resourceType')=='Encounter' and _linked(resource,'encounter',r)),None)
        if target:return target.get('_fullUrl')
        identifier=ref.get('identifier',{})
        return ref.get('reference') or ((identifier.get('system'),identifier['value']) if identifier.get('value') else None)
    return _same_source(left,right) and identity(left) is not None and identity(left)==identity(right)


def _branch_result_context(branch, result, resources):
    # Bind results to independently observable parent content and episode time.
    # Branch labels and generic background observations are never evidence.
    if branch=='oxygen-saturation':
        return any(_linked(result,'encounter',visit) and _same_source(result,visit)
                   and any((SNOMED,'49727002') in set(codings(reason)) for reason in visit.get('reasonCode',[]))
                   and 0<=_days(visit,result)<=1
                   for visit in _rows(resources,'Encounter'))
    if branch=='glucose-monitor' and any(_coded(c,'code',SNOMED,'75934005') for c in _rows(resources,'Condition')):
        if any(_linked(result,'encounter',visit) and _same_source(result,visit)
               and any((SNOMED,'390906007') in set(codings(reason)) for reason in visit.get('reasonCode',[]))
               and 0<=_days(visit,result)<=1 for visit in _rows(resources,'Encounter')):
            return True
    parent_code={'glucose-screen':'2093-3','glucose-monitor':'6298-4','cbc-monitor':'6298-4',
                 'hepatic-toxicity':'2160-0','hematologic-toxicity':'2160-0','cbc-monitoring':'718-7'}[branch]
    from .clinical_semantics import meaningful_result
    return any(_coded(parent,'code',LOINC,parent_code) and meaningful_result(parent)
               and _same_encounter(parent,result,resources) and 0<=_days(parent,result)<=1
               for parent in _rows(resources,'Observation'))


def material_branch_proof(branch: str, resources: list[dict], birth_date: str) -> bool | None:
    """None means explicitly unsupported, never a coverage pass."""
    from .clinical_semantics import CONTRACT, meaningful_result, semantic_findings
    from .medication_safety import normalized_dose_mg, ingredient, exposure
    if branch in LEGACY_BRANCHES:
        return branch_proof(branch, resources, birth_date)
    if branch in {'dose-increase','dose-decrease','switch-class'}:
        requests = _rows(resources,'MedicationRequest')
        for current in requests:
            for prior in requests:
                if not _linked(current,'priorPrescription',prior) or not _same_source(current,prior):continue
                before,after=normalized_dose_mg(prior),normalized_dose_mg(current)
                if before is None or after is None or _days(prior,current)<=0:continue
                old_interval,new_interval=exposure(prior),exposure(current)
                if not old_interval or not new_interval or old_interval[1]!=new_interval[0]:continue
                if prior.get('status') not in {'completed','stopped'} or current.get('status')!='active':continue
                if branch=='switch-class':
                    if {ingredient(prior),ingredient(current)}=={'29046','52175'}:return True
                elif ingredient(prior)==ingredient(current):
                    if branch=='dose-increase' and after>before or branch=='dose-decrease' and after<before:return True
        return False
    if branch in CONTRACT['branchTests']:
        wanted=CONTRACT['branchTests'][branch]
        for code in wanted:
            matches=[r for r in _rows(resources,'Observation') if _coded(r,'code',LOINC,code) and meaningful_result(r)
                     and _branch_result_context(branch,r,resources)]
            if code=='59408-5':
                matches=[r for r in matches if r.get('valueQuantity',{}).get('code')=='%'
                         and 0 < r.get('valueQuantity',{}).get('value',0) <= 100]
            else:
                matches=[r for r in matches if any(_linked(r,'basedOn',order) and _days(order,r)>=0
                            for order in _rows(resources,'ServiceRequest') if _coded(order,'code',LOINC,code))]
            if not matches or any(row['finding']=='F11' for row in semantic_findings(resources)):
                return False
        return True
    if branch in {'anesthesia','analgesia','antibiotic-prophylaxis'}:
        surgeries=[r for r in _rows(resources,'Procedure') if r.get('status')=='completed'
                   and _coded(r,'code',SNOMED,'387713003')]
        if branch=='anesthesia':
            interventions=[r for r in _rows(resources,'Procedure') if r.get('status')=='completed'
                           and _coded(r,'code',SNOMED,'50697003')]
            return any(_linked(r,'partOf',surgery) and 0 <= _days(r,surgery) <= 1/24
                       for r in interventions for surgery in surgeries)
        code='161' if branch=='analgesia' else '2180'
        interventions=[r for r in _rows(resources,'MedicationAdministration')
                       if r.get('status')=='completed' and ingredient(r)==code
                       and normalized_dose_mg(r)==(500 if code=='161' else 2000)]
        return any(_linked(r,'partOf',surgery) and (
                    0<=_days(surgery,r)<=1 if branch=='analgesia' else .5/24<=_days(r,surgery)<=1/24)
                   for r in interventions for surgery in surgeries)
    return None


def branch_coverage_audit(bundles, crosswalk, spec):
    from .observable import patient_graphs, is_supplemental_capture
    from .clinical_semantics import CONTRACT
    graphs,patients=patient_graphs(bundles,spec)
    rows=[]
    for declaration in crosswalk['patients']:
        cohort=declaration['cohortId']
        branch=declaration.get('secondaryBranch')
        if not branch:continue
        resources=[r for r in graphs.get(cohort,[]) if not is_supplemental_capture(r)]
        # Problem/control source overlays do not establish ordinary branch truth.
        if not declaration.get('ordinaryPathwayEligible',True):
            status='scenario-overlay-excluded'
        else:
            proof=material_branch_proof(branch,resources,patients[cohort]['birthDate'])
            status='unsupported' if proof is None else 'proven' if proof else 'failed'
        rows.append({'cohortId':cohort,'branch':branch,'status':status,
                     'coverageCredit':status=='proven',
                     **({'reason':CONTRACT['unsupportedBranchPolicy']} if status=='unsupported' else {})})
    return {'status':'fail' if any(r['status']=='failed' for r in rows) else 'pass',
            'assignedCount':len(rows),'provenCount':sum(r['status']=='proven' for r in rows),
            'unsupportedCount':sum(r['status']=='unsupported' for r in rows),'rows':rows}
