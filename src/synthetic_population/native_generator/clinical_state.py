"""S20 clinical-state contract v1.0.0, independent of scenario participation.

Synthetic ordinary growth is a generation model, not a diagnostic reference.
CKD-EPI 2021 is restricted to adults; special methods must be represented explicitly.
"""
from __future__ import annotations

from datetime import date, timedelta
from math import isfinite

from .temporal import parse_fhir_time
from .catalogs import load_catalog

CONTRACT_PATH = "tools/synthetic_population/data/clinical-state-contract.v1.0.0.json"
CONTRACT = load_catalog(CONTRACT_PATH)

CONTRACT_VERSION = "1.0.0"
GROWTH = tuple((r["ageYears"], r["heightCm"], r["weightKg"]) for r in CONTRACT["anthropometry"]["growthAnchors"])


def age_years(birth: str, when: str) -> float:
    return (date.fromisoformat(when[:10]) - date.fromisoformat(birth)).days / 365.2425


def under_18_months(birth: str, when: str) -> bool:
    from calendar import monthrange
    born = date.fromisoformat(birth)
    month = born.month - 1 + 18
    year, month = born.year + month // 12, month % 12 + 1
    boundary = date(year, month, min(born.day, monthrange(year, month)[1]))
    return date.fromisoformat(when[:10]) < boundary


def sensitivity_first_sample(birth: str, anchor: str, count: int = 2):
    """Shared event-time contract; absence checks cannot substitute snapshot age."""
    born = parse_fhir_time(birth).start
    horizon = parse_fhir_time(anchor).start - timedelta(microseconds=1)
    first = max(born + timedelta(days=30), horizon - timedelta(days=60))
    if under_18_months(birth, first.date().isoformat()):
        first = min(first, horizon - timedelta(days=count - 1))
    if first < born:
        raise ValueError("independent sensitive samples cannot fit in this life interval")
    return first


def egfr_2021(creatinine: float, age: int, sex: str) -> float:
    if age < 18 or sex not in {"male", "female"} or not isfinite(creatinine) or creatinine <= 0:
        raise ValueError("CKD-EPI 2021 requires adult age, modeled sex and positive creatinine")
    k, alpha, factor = (.7, -.241, 1.012) if sex == "female" else (.9, -.302, 1)
    return 142 * min(creatinine / k, 1) ** alpha * max(creatinine / k, 1) ** -1.2 * .9938 ** age * factor


def stage4_creatinine(age: int, sex: str, target: float) -> float:
    egfr_2021(1, age, sex)  # Validate model eligibility before inversion.
    if not 15 <= target < 30:
        raise ValueError("stage 4 requires eGFR 15 to <30")
    k, factor = (.7, 1.012) if sex == "female" else (.9, 1)
    return round(k * (target / (142 * .9938 ** age * factor)) ** (-1 / 1.2), 2)


def body_measurements(age: float, fraction: float = .5) -> tuple[float, float]:
    if age < 0:
        raise ValueError("measurement before birth")
    if age >= 18:
        height = 150 + 40 * fraction
        bmi = 19 + 9 * fraction
        return round(height, 2), round(bmi * (height / 100) ** 2, 2)
    lower = max((row for row in GROWTH if row[0] <= age), key=lambda row: row[0])
    upper = min((row for row in GROWTH if row[0] >= age), key=lambda row: row[0])
    part = (age - lower[0]) / (upper[0] - lower[0]) if upper != lower else 0
    scale = .95 + .1 * fraction
    return tuple(round((lower[i] + (upper[i] - lower[i]) * part) * scale, 2) for i in (1, 2))


def codes(resource: dict, field: str = "code") -> set[str]:
    return {c.get("code") for c in resource.get(field, {}).get("coding", [])}


def acute_snapshot(condition: dict, anchor: str) -> None:
    """A seven-day synthetic episode: future recovery is not completed abatement."""
    if not codes(condition) & {"74400008", "34014006", "195662009"}:
        return
    start = condition.get("onsetDateTime") or condition.get("recordedDate")
    if not start:
        return
    end = parse_fhir_time(start).start + timedelta(days=7)
    resolved = end <= parse_fhir_time(anchor).start
    condition["clinicalStatus"]["coding"][0]["code"] = "resolved" if resolved else "active"
    if resolved:
        condition["abatementDateTime"] = end.isoformat().replace("+00:00", "Z")
        if "74400008" in codes(condition):
            condition["note"] = [{"text": "Historical acute appendicitis episode resolved; treatment details are outside this source capture."}]
    else:
        condition.pop("abatementDateTime", None)


def state_findings(resources: list[dict], birth: str, sex: str, anchor: str,
                   death: str | None = None, *, required_infant_hiv: bool = False) -> list[dict]:
    findings = []
    def fail(finding, resource, reason):
        findings.append({"finding": finding, "resource": resource.get("_fullUrl", resource.get("id")), "reason": reason})
    horizon = parse_fhir_time(anchor).start
    born = parse_fhir_time(birth).start
    died = parse_fhir_time(death).end if death else horizon
    by_ref = {r.get("_fullUrl", r.get("id")): r for r in resources if r.get("_fullUrl", r.get("id"))}

    def instant(value):
        try:
            return parse_fhir_time(value).start if value else None
        except (ValueError, TypeError):
            return None

    def sample_key(r):
        specimen = r.get("specimen", {})
        identifier = specimen.get("identifier", {})
        if identifier.get("value"):
            return (identifier.get("system"), identifier["value"])
        target = by_ref.get(specimen.get("reference"), {})
        identifiers = target.get("identifier", [])
        return tuple(sorted((i.get("system", ""), i["value"]) for i in identifiers if i.get("value"))) or specimen.get("reference")

    def final_sample(r):
        at = instant(r.get("effectiveDateTime") or r.get("effectivePeriod", {}).get("start"))
        return (r.get("status") in {"final", "amended", "corrected"}
                and not r.get("derivedFrom") and sample_key(r)
                and at is not None and born <= at <= min(horizon, died))

    renal = [r for r in resources if r.get("resourceType") == "Condition" and "431857002" in codes(r)]
    renal_evidence = {}
    evidence_refs = {e.get("reference") for r in renal for group in r.get("evidence", []) for e in group.get("detail", [])}
    measured = {}
    for ref in evidence_refs:
        r = by_ref.get(ref, {})
        q = r.get("valueQuantity", {})
        if (r.get("resourceType") == "Observation" and final_sample(r)
                and "measured-gfr" in codes(r) and r.get("method", {}).get("text") == CONTRACT["renal"]["exceptionMethod"]
                and q.get("code") == "mL/min/{1.73_m2}" and isinstance(q.get("value"), (int, float))
                and 15 <= q["value"] < 30):
            measured[sample_key(r)] = instant(r.get("effectiveDateTime") or r.get("effectivePeriod", {}).get("start"))
    supported_measured = len(measured) >= 2 and (max(measured.values()) - min(measured.values())).days >= 90
    if supported_measured:
        renal_evidence.update(measured)
    infant_hiv, infant_diagnoses = [], []
    for r in resources:
        kind = r.get("resourceType")
        if kind == "Condition":
            for field in ("onsetPeriod", "abatementPeriod"):
                period = r.get(field, {})
                low, high = instant(period.get("start")), instant(period.get("end"))
                if (period and not (low or high)
                        or low is not None and not born <= low <= min(horizon, died)
                        or high is not None and not born <= high <= min(horizon, died)
                        or low is not None and high is not None and high < low):
                    fail("F07", r, "condition Period endpoints violate life/snapshot bounds or interval order")
            documented = instant(r.get("recordedDate"))
            if documented is not None and documented > horizon:
                fail("F07", r, "condition documentation is after the saved snapshot")
            onset = instant(r.get("onsetDateTime") or r.get("onsetPeriod", {}).get("start") or r.get("recordedDate"))
            end_value = r.get("abatementDateTime") or r.get("abatementPeriod", {}).get("end")
            end = instant(end_value)
            resolved = "resolved" in codes(r, "clinicalStatus")
            if (onset is not None and (onset < born or onset > min(horizon, died))) or (
                    end_value and (end is None or end > min(horizon, died) or end < born or onset is not None and end < onset)):
                fail("F07", r, "condition state lacks completed abatement or violates snapshot/life interval")
            if "74400008" in codes(r) and "active" in codes(r, "clinicalStatus") and onset is not None:
                ongoing = any(
                    by_ref.get(e.get("reference"), {}).get("resourceType") == "Observation"
                    and "74400008" in codes(by_ref[e["reference"]])
                    and by_ref[e["reference"]].get("status") == "final"
                    and by_ref[e["reference"]].get("valueString")
                    for group in r.get("evidence", []) for e in group.get("detail", []))
                if (horizon - onset).days > 90 and not ongoing:
                    fail("F05", r, "persistent appendicitis lacks ongoing episode evidence")
            if codes(r) & {"86406008", "62479008"} and "confirmed" in codes(r, "verificationStatus"):
                if onset is None or under_18_months(birth, onset.date().isoformat()):
                    infant_diagnoses.append(r)
        if kind != "Observation":
            continue
        cs = codes(r)
        when = r.get("effectiveDateTime") or r.get("effectivePeriod", {}).get("start")
        at = instant(when)
        if at is None:
            if cs & {"8302-2", "29463-7", "19868-9", "2160-0", "68961-2", "5018-7"}:
                fail("F03" if cs & {"68961-2", "5018-7"} else "F02" if "2160-0" in cs and renal else "F06", r, "clinical event date is unresolved")
            continue
        age = age_years(birth, when)
        q = r.get("valueQuantity", {})
        value = q.get("value")
        unit = q.get("code")
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value) and value > 0
        if cs & {"8302-2", "29463-7"} and value is not None:
            height = "8302-2" in cs
            expected_unit = {"cm", "m", "mm"} if height else {"kg", "g"}
            converted = value * (100 if unit == "m" else .1 if unit == "mm" else 1) if height and numeric else value / (1000 if unit == "g" else 1) if numeric else 0
            typical = body_measurements(max(0, age))[0 if height else 1]
            supported = False
            for ref in r.get("derivedFrom", []):
                assessment = by_ref.get(ref.get("reference"), {})
                assessed_at = instant(assessment.get("effectiveDateTime"))
                exception = next((c for c in codes(assessment, "valueCodeableConcept")
                                  if c in CONTRACT["anthropometry"]["supportedExceptions"]), None)
                policy = CONTRACT["anthropometry"]["supportedExceptions"].get(exception, {})
                same_subject = bool(r.get("subject")) and assessment.get("subject") == r["subject"]
                diagnosis = any(
                    exception in codes(by_ref.get(detail.get("reference"), {}))
                    and by_ref[detail["reference"]].get("subject") == r.get("subject")
                    and "confirmed" in codes(by_ref[detail["reference"]], "verificationStatus")
                    for detail in assessment.get("derivedFrom", [])) if exception else False
                supported |= bool(
                    "anthropometry-assessment" in codes(assessment) and assessment.get("valueString")
                    and assessment.get("status") == "final" and same_subject and assessed_at
                    and born <= assessed_at <= at and (at - assessed_at).days <= 30
                    and diagnosis and cs & set(policy.get("codes", []))
                    and age >= policy.get("minimumAgeYears", 0)
                    and policy["ratioBounds"][0] <= converted / typical <= policy["ratioBounds"][1])
            if (age < 0 or at > min(horizon, died) or unit not in expected_unit or not numeric
                    or r.get("status") not in {"final", "amended", "corrected"}
                    or not .35 <= converted / typical <= 3 and not supported
                    or r.get("referenceRange") or r.get("interpretation")):
                fail("F06", r, "unsupported event-age anthropometry, units or unvalidated reference metadata")
        if "19868-9" in cs and age < 6 and "valueQuantity" in r:
            fail("F06", r, "unsupported spirometry below the modeled age; result must be unavailable")
        if "2160-0" in cs and renal and value is not None and not supported_measured:
            completed = date.fromisoformat(when[:10]).year - date.fromisoformat(birth).year - (when[5:10] < birth[5:10])
            if unit not in {"mg/dL", "umol/L"} or completed < 18 or not numeric:
                fail("F02", r, "unsupported renal method or units")
            else:
                scr = value / 88.4 if unit == "umol/L" else value
                estimated = egfr_2021(scr, completed, sex)
                if not 15 <= estimated < 30:
                    fail("F02", r, "stage-4 creatinine/eGFR discordance requires independent measured-GFR evidence")
                elif final_sample(r):
                    renal_evidence[sample_key(r)] = at
        if cs & {"68961-2", "5018-7"} and "260373001" in codes(r, "valueCodeableConcept") and under_18_months(birth, when):
            infant_hiv.append(r)
    if renal and (len(renal_evidence) < 2 or (max(renal_evidence.values()) - min(renal_evidence.values())).days < 90):
        fail("F02", renal[0], "stage-4 diagnosis lacks independent final longitudinal renal assessments within the life interval")
    if infant_hiv or infant_diagnoses or required_infant_hiv:
        specimens = {sample_key(r) for r in infant_hiv if "5018-7" in codes(r) and final_sample(r)}
        if len(specimens) < 2 or any("68961-2" in codes(r) for r in infant_hiv):
            fail("F03", (infant_hiv or infant_diagnoses or [{}])[0], "infant positive status requires two independent final virologic samples")
    return findings


def population_state_audit(bundles: dict, spec: dict) -> dict:
    from collections import defaultdict
    patients, rows = {}, defaultdict(list)
    system = spec["identity"]["cohortIdentifierSystem"]
    for source, bundle in bundles.items():
        anchors = {}
        for entry in bundle["entry"]:
            r = entry["resource"]
            if r.get("resourceType") == "Patient":
                cohort = next((i.get("value") for i in r.get("identifier", []) if i.get("system") == system), None)
                if cohort:
                    anchors[entry["fullUrl"]] = cohort
                    patients.setdefault(cohort, r)
        for entry in bundle["entry"]:
            r = entry["resource"]
            cohort = anchors.get(r.get("subject", r.get("patient", {})).get("reference"))
            if cohort:
                rows[cohort].append({**r, "_fullUrl": entry["fullUrl"], "_source": source})
    supported_renal = []
    findings, strata = [], defaultdict(lambda: {"people": set(), "facts": set(), "sourceCopies": 0, "values": []})
    for cohort, p in sorted(patients.items()):
        resources = rows.get(cohort, [])
        if not p.get("birthDate") or p.get("gender") not in {"male", "female"}:
            findings.append({"cohortId": cohort, "finding": "F06", "resource": None,
                             "reason": "canonical demographic context missing; clinical age/model unresolved"})
            continue
        from .sensitivity import fact_for_patient
        designated_positive = fact_for_patient(spec, int(cohort.rsplit("-", 1)[-1])) == "minor-hiv-positive"
        required_infant = designated_positive and under_18_months(
            p["birthDate"], sensitivity_first_sample(p["birthDate"], spec["timelineAnchor"]).date().isoformat())
        patient_findings = state_findings(resources, p["birthDate"], p["gender"], spec["timelineAnchor"], p.get("deceasedDateTime"), required_infant_hiv=required_infant)
        for finding in patient_findings:
            findings.append({"cohortId": cohort, **finding})
        if any(r.get("resourceType") == "Condition" and "431857002" in codes(r) for r in resources) and not any(f["finding"] == "F02" for f in patient_findings):
            supported_renal.append(cohort)
        for r in resources:
            if r.get("resourceType") != "Observation" or not codes(r) & {"29463-7", "8302-2", "19868-9"} or not r.get("valueQuantity"):
                continue
            when = r.get("effectiveDateTime") or r.get("effectivePeriod", {}).get("start")
            if not when:
                continue
            age = age_years(p["birthDate"], when)
            key = next(iter(codes(r))) + ":" + ("infant" if age < 1 else "child" if age < 18 else "adult")
            group = strata[key]
            group["people"].add(cohort)
            fact = next((i["value"] for i in r.get("identifier", []) if i.get("system") == "urn:hdp-test-bundle:clinical-event-key"), r["_fullUrl"])
            if fact not in group["facts"]:
                group["values"].append(r["valueQuantity"]["value"])
            group["facts"].add(fact)
            group["sourceCopies"] += 1
    summary = {k: {"people": len(v["people"]), "facts": len(v["facts"]), "sourceCopies": v["sourceCopies"],
                   "minimum": min(v["values"]), "maximum": max(v["values"])} for k, v in sorted(strata.items())}
    return {"contractVersion": CONTRACT_VERSION, "findings": findings, "eventAgeMeasurements": summary,
            "supportedStage4Cohorts": supported_renal,
            "status": "fail" if findings else "pass", "reviewDisposition": "engineering screen; exact-case clinical review required"}
