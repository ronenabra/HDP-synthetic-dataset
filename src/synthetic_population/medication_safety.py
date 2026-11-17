"""S21 regimen, compatibility and complete authorization-history contracts."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from math import isfinite, isclose
import re

from .catalogs import load_catalog
from .clinical_state import codes
from .temporal import parse_fhir_time

CONTRACT_PATH = "data/medication-safety-contract.v1.0.0.json"
CONTRACT = load_catalog(CONTRACT_PATH)
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
EVENT_KEY = "urn:hdp-test-bundle:clinical-event-key"


def regimen_catalog(spec: dict) -> dict:
    catalog = load_catalog(spec["remediation"]["clinicalContentCatalog"])
    base = catalog["modelDefinitions"]["medication-concept"]["valueModel"]["regimens"]
    from .clinical_semantics import CONTRACT as observable_contract
    return {**base, **CONTRACT["additionalRegimens"], **observable_contract.get("perioperativeRegimens", {})}


def ingredient(resource: dict) -> str | None:
    return next((row["code"] for row in resource.get("medicationCodeableConcept", {}).get("coding", [])
                 if row.get("system") == RXNORM), None)


def dosage_for(regimen: dict, *, dose_mg: float | None = None,
               start: str | None = None, end: str | None = None) -> dict:
    dose = regimen["doseMg"] if dose_mg is None else dose_mg
    frequency = regimen["frequencyPerDay"]
    repeat = {"frequency": frequency, "period": 1, "periodUnit": "d"}
    if start:
        repeat["boundsPeriod"] = {"start": start, **({"end": end} if end else {})}
    return {"text": f"Take {dose:g} mg by mouth {frequency:g} time(s) daily",
            "timing": {"repeat": repeat},
            "route": {"coding": [{"system": "http://snomed.info/sct", "code": regimen["routeCode"], "display": "Oral route"}]},
            "doseAndRate": [{"doseQuantity": {"value": dose, "unit": "mg", "system": "http://unitsofmeasure.org", "code": "mg"}}]}


def instructions(resource: dict) -> list[dict]:
    values = resource.get("dosageInstruction", resource.get("dosage", []))
    if isinstance(values, dict):
        return [{**values, "doseAndRate": [{"doseQuantity": values.get("dose", {})}]}]
    return values


def normalized_dose_mg(resource: dict) -> float | None:
    """All provided dose representations must agree; a tablet is not a mass unit."""
    rows = instructions(resource)
    if len(rows) != 1:
        return None
    quantities = [row.get("doseQuantity", {}) for row in rows[0].get("doseAndRate", [])]
    if not quantities:
        return None
    product = resource.get("medicationCodeableConcept", {}).get("text", "")
    strength_match = re.search(r"\b(\d+(?:\.\d+)?)\s*mg\s+per\s+tablet\b", product, re.I)
    strength = float(strength_match[1]) if strength_match else None
    amounts = []
    for quantity in quantities:
        value, unit = quantity.get("value"), quantity.get("code")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value) or value <= 0:
            return None
        if not isinstance(unit, str) or not isinstance(quantity.get("unit", unit), str):
            return None
        labels = {"mg": {"mg", "milligram", "milligrams"}, "g": {"g", "gram", "grams"},
                  "ug": {"ug", "mcg", "microgram", "micrograms"},
                  "1": {"1", "tablet", "tablets"}, "{tbl}": {"{tbl}", "tablet", "tablets"},
                  "428673006": {"tablet", "tablets", "428673006"}}
        expected_system = "http://snomed.info/sct" if unit == "428673006" else "http://unitsofmeasure.org"
        if quantity.get("system") != expected_system or quantity.get("unit", unit).lower() not in labels.get(unit, set()):
            return None
        if unit in CONTRACT["massUnitMg"]:
            amounts.append(value * CONTRACT["massUnitMg"][unit])
        elif unit in {"1", "{tbl}", "428673006"} and strength is not None:
            amounts.append(value * strength)
        else:
            return None
    return amounts[0] if all(isclose(amounts[0], value, rel_tol=1e-6, abs_tol=1e-6) for value in amounts) else None


def exposure(resource: dict) -> tuple[datetime, datetime] | None:
    rows = instructions(resource)
    bounds = rows[0].get("timing", {}).get("repeat", {}).get("boundsPeriod", {}) if rows else {}
    period = resource.get("effectivePeriod", bounds)
    start = period.get("start") or resource.get("authoredOn") or resource.get("effectiveDateTime")
    end = period.get("end")
    if not start:
        return None
    # Missing cessation is unknown for stopped/completed therapy, never an
    # invented same-day end that could hide a historical unsafe overlap.
    last = parse_fhir_time(end).start if end else datetime.max.replace(tzinfo=timezone.utc)
    return parse_fhir_time(start).start, last


def overlap(a, b) -> bool:
    return bool(a and b and max(a[0], b[0]) < min(a[1], b[1]))


def compatible_ingredient(candidate: str, existing: set[str], *, stage4: bool = False) -> bool:
    if stage4 and candidate == "6809":
        return False
    return not any(candidate in pair and (set(pair) - {candidate}) & existing
                   for pair in CONTRACT["incompatibleIngredients"])


def _identity(resource: dict):
    event = next((i.get("value") for i in resource.get("identifier", []) if i.get("system") == EVENT_KEY), None)
    if event:
        return ("event", event)
    strong = tuple(sorted((i.get("system", ""), i["value"]) for i in resource.get("identifier", []) if i.get("value")))
    return ("identifier", strong) if strong else ("url", resource.get("_fullUrl", resource.get("id", id(resource))))


def medication_findings(resources: list[dict], spec: dict) -> list[dict]:
    findings = []
    models = regimen_catalog(spec)
    medications = [r for r in resources if r.get("resourceType") in {"MedicationRequest", "MedicationStatement", "MedicationDispense", "MedicationAdministration"}]
    def fail(finding, resource, reason):
        findings.append({"finding": finding, "resource": resource.get("_fullUrl", resource.get("id")), "reason": reason})
    for r in medications:
        code = ingredient(r)
        if code not in models:
            continue
        dose = normalized_dose_mg(r)
        if dose is None:
            fail("F01", r, "dose is not a consistent mass quantity or explicitly strength-backed tablet count")
            continue
        model = models[code]
        low, high = model.get("doseRangeMg", [model["doseMg"] / 2, model["doseMg"] * 2])
        if not low <= dose <= high:
            fail("F09", r, "regimen is outside the ingredient-specific synthetic contract")
        row = instructions(r)[0]
        if r["resourceType"] != "MedicationAdministration":
            repeat = row.get("timing", {}).get("repeat", {})
            if repeat.get("frequency") != model["frequencyPerDay"] or repeat.get("period") != 1 or repeat.get("periodUnit") != "d":
                # Acetaminophen twice daily is an explicit legacy background
                # regimen; this is an ingredient-specific alternative.
                if not (code == "161" and repeat.get("frequency") == 2 and repeat.get("period") == 1 and repeat.get("periodUnit") == "d"):
                    fail("F09", r, "frequency does not match the selected ingredient regimen")
        if model["routeCode"] not in codes(row, "route"):
            fail("F09", r, "route does not match the selected ingredient regimen")
        text_doses = re.findall(r"\b(\d+(?:\.\d+)?)\s*(mg|ug|g)\b", row.get("text", ""))
        if not text_doses or any(not isclose(float(value) * CONTRACT["massUnitMg"][unit], dose, rel_tol=1e-6)
                                 for value, unit in text_doses):
            fail("F01", r, "dose narrative and structured mass disagree or narrative lacks a dose")
    therapies = list({_identity(r): r for r in medications if r["resourceType"] in {"MedicationRequest", "MedicationStatement"}
                     and r.get("status") in {"active", "completed", "stopped"}}.values())
    for first, second in CONTRACT["incompatibleIngredients"]:
        for a in [r for r in therapies if ingredient(r) == first]:
            for b in [r for r in therapies if ingredient(r) == second]:
                if overlap(exposure(a), exposure(b)):
                    fail("F10", b, "ACE-inhibitor/ARB exposure overlaps without an explicitly timed switch")
    renal = [r for r in resources if r.get("resourceType") == "Condition" and "431857002" in codes(r)]
    for r in therapies:
        if ingredient(r) != "6809":
            continue
        for condition in renal:
            start = condition.get("onsetDateTime") or condition.get("recordedDate")
            end = condition.get("abatementDateTime")
            interval = (parse_fhir_time(start).start, parse_fhir_time(end).start if end else datetime.max.replace(tzinfo=timezone.utc)) if start else None
            if interval is None or overlap(exposure(r), interval):
                fail("F13", r, "metformin exposure conflicts with recorded stage-4 CKD; participant tags do not exempt background therapy")
                break
    return findings


def dispense_ledger(resources: list[dict]) -> dict:
    """Account every unique completed fill, including partial fills and renewals."""
    requests = [r for r in resources if r.get("resourceType") == "MedicationRequest"]
    lookup = {}
    for r in requests:
        if r.get("_fullUrl"):
            lookup[r["_fullUrl"]] = r
        if r.get("id"):
            lookup["MedicationRequest/" + r["id"]] = r
        for identifier in r.get("identifier", []):
            if identifier.get("value"):
                lookup[(identifier.get("system"), identifier["value"])] = r
    findings, unresolved = [], []
    grouped = defaultdict(dict)
    fill_authorizations = {}
    def fail(r, reason):
        findings.append({"finding": "F12", "resource": r.get("_fullUrl", r.get("id")), "reason": reason})
    completed = [r for r in resources if r.get("resourceType") == "MedicationDispense" and r.get("status") == "completed"]
    for fill in completed:
        matches = []
        for ref in fill.get("authorizingPrescription", []):
            identifier = ref.get("identifier", {})
            request = lookup.get(ref.get("reference")) or lookup.get((identifier.get("system"), identifier.get("value")))
            if request is not None:
                matches.append(request)
        if len({_identity(r) for r in matches}) != 1:
            if not fill.get("authorizingPrescription"):
                fail(fill, "completed fill has no authorizing prescription")
            else:
                unresolved.append(fill.get("_fullUrl", fill.get("id")))
            continue
        request = matches[0]
        key, fill_key = _identity(request), _identity(fill)
        if fill_key in fill_authorizations and fill_authorizations[fill_key] != key:
            fail(fill, "copies of one clinical fill disagree on the authorizing prescription")
            continue
        fill_authorizations[fill_key] = key
        if fill_key in grouped[key]:
            old = grouped[key][fill_key][1]
            if any(old.get(field) != fill.get(field) for field in ("whenHandedOver", "quantity", "status", "type", "medicationCodeableConcept")):
                fail(fill, "repeated capture of the same fill disagrees")
        else:
            grouped[key][fill_key] = (request, fill)
    ledgers = []
    for key, pairs in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        request = next(iter(pairs.values()))[0]
        authorization = request.get("dispenseRequest", {})
        quantity = authorization.get("quantity", {})
        per_fill = quantity.get("value")
        repeats = authorization.get("numberOfRepeatsAllowed", 0)
        fills = sorted((pair[1] for pair in pairs.values()), key=lambda r: r.get("whenHandedOver", ""))
        if not isinstance(per_fill, (int, float)) or not isfinite(per_fill) or per_fill <= 0 or not isinstance(repeats, int) or repeats < 0:
            fail(request, "authorization requires positive quantity and a nonnegative repeat allowance")
            continue
        allowed = per_fill * (1 + repeats)
        used = 0
        start_value = authorization.get("validityPeriod", {}).get("start") or request.get("authoredOn")
        end_value = authorization.get("validityPeriod", {}).get("end")
        start = parse_fhir_time(start_value).start if start_value else None
        end = parse_fhir_time(end_value).end if end_value else None
        if request.get("status") in {"revoked", "stopped", "cancelled", "entered-in-error", "draft"} and not end:
            fail(request, "authorization status lacks a verifiable cessation boundary")
        for fill in fills:
            q = fill.get("quantity", {})
            value = q.get("value")
            when = fill.get("whenHandedOver")
            at = parse_fhir_time(when).start if when else None
            if at is None or start is None or at < start or end is not None and at > end:
                fail(fill, "fill is outside authorization validity or has no event time")
            if ingredient(fill) != ingredient(request):
                fail(fill, "dispensed ingredient differs from authorization")
            if not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                fail(fill, "fill quantity is not positive and finite")
                continue
            if (q.get("code"), q.get("system")) != (quantity.get("code"), quantity.get("system")):
                fail(fill, "fill and authorization quantities have incompatible units")
            if value > per_fill or (used % per_fill) + value > per_fill + 1e-6:
                fail(fill, "fill exceeds the remaining quantity of one authorized supply")
            expected_type = ("FF" if used < per_fill else "RF") + ("C" if isclose(value, per_fill) else "P")
            if expected_type not in codes(fill, "type"):
                fail(fill, "first/refill and complete/partial coding disagrees with the full history")
            used += value
            if used > allowed + 1e-6:
                fail(fill, "cumulative dispensing exceeds initial allowance plus repeats")
        ledgers.append({"authorization": request.get("_fullUrl", request.get("id")), "uniqueFills": len(fills),
                        "allowedQuantity": allowed, "dispensedQuantity": used})
    return {"findings": findings, "unresolvedAuthorizationFills": unresolved, "authorizations": ledgers,
            "sourceCopyCount": len(completed), "uniqueFillCount": sum(len(v) for v in grouped.values()),
            "status": "fail" if findings else "incomplete" if unresolved else "pass"}


def population_medication_audit(bundles: dict, spec: dict) -> dict:
    from .observable import patient_graphs
    graphs, _ = patient_graphs(bundles, spec)
    findings, ledgers = [], []
    for cohort, resources in sorted(graphs.items()):
        ledger = dispense_ledger(resources)
        findings.extend({**row, "cohortId": cohort} for row in medication_findings(resources, spec) + ledger["findings"])
        findings.extend({"finding": "F12", "resource": url, "cohortId": cohort,
                         "reason": "authorization history is incomplete: referenced prescription is unavailable"}
                        for url in ledger["unresolvedAuthorizationFills"])
        ledgers.append({"cohortId": cohort, **ledger})
    return {"status": "fail" if findings else "pass", "findings": findings,
            "people": len(graphs), "dispenseHistories": ledgers}
