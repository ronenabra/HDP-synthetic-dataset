from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any

from .catalogs import load_catalog
from .clinical_state import age_years, body_measurements, under_18_months, sensitivity_first_sample
from .clinical_data import quantitative_result
from .medication_safety import regimen_catalog, compatible_ingredient
from .clinical_semantics import CONTRACT as RESULT_CONTRACT, order_compatible
from .clinical_policy import age_applicability, completed_age_years
from .ids import cohort_id
from .temporal import ClinicalInterval, strictly_before


INJURY_BRANCH_CONCEPTS = {
    "burn-care": ("125666000", "Burn"),
    "minor-burn": ("125666000", "Burn"),
    "community-review": ("125666000", "Burn"),
    "sprain-care": ("44465007", "Sprain of ankle"),
    "sprain": ("44465007", "Sprain of ankle"),
    "wound-care": ("125643001", "Open wound"),
    "wound-review": ("125643001", "Open wound"),
}


@dataclass(frozen=True)
class ClinicalFact:
    id: str
    family: str
    system: str
    code: str
    display: str
    status: str
    value: float | None = None
    unit: str | None = None
    reference_low: float | None = None
    reference_high: float | None = None
    interpretation: str | None = None
    blood_pressure_pair: tuple[float, float] | None = None
    medication_detail: dict[str, Any] | None = None
    coded_value: dict[str, str] | None = None
    result_text: str | None = None


@dataclass(frozen=True)
class EventLink:
    relationship: str
    target_event_id: str


@dataclass(frozen=True)
class TemporalConstraint:
    predecessor_event_id: str
    successor_event_id: str
    minimum_days: int = 0
    maximum_days: int | None = None


@dataclass(frozen=True)
class ClinicalEvent:
    id: str
    episode_id: str
    fact_id: str
    role: str
    clinical_domain: str
    care_setting: str
    capability: str
    lifecycle: str
    occurrence: ClinicalInterval
    indication_fact_id: str | None = None
    links: tuple[EventLink, ...] = ()
    permitted_dispositions: tuple[str, ...] = (
        "performed", "authored", "received", "summarized", "referral-only",
        "metadata-only", "absent",
    )
    provenance_required: bool = False
    sensitivity_fact: str | None = None
    subject_id: str | None = None
    participant_roles: tuple[str, ...] = ("responsible-clinician",)


@dataclass(frozen=True)
class CareEpisode:
    id: str
    setting: str
    required_capability: str
    event_ids: tuple[str, ...]
    routed_source: str | None = None


@dataclass(frozen=True)
class PatientPathway:
    id: str
    cohort_id: str
    archetype: str
    variant: str
    secondary_branch: str
    secondary_trait: str
    facts: tuple[ClinicalFact, ...]
    events: tuple[ClinicalEvent, ...]
    episodes: tuple[CareEpisode, ...]
    constraints: tuple[TemporalConstraint, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for event in value["events"]:
            interval = event["occurrence"]
            interval["start"] = interval["start"].isoformat().replace("+00:00", "Z")
            interval["end"] = interval["end"].isoformat().replace("+00:00", "Z")
        return value


def _choice(seed: int, cohort: str, namespace: str, size: int) -> int:
    digest = hashlib.sha256(f"{seed}|{cohort}|{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % size


def _stable_id(seed: int, cohort: str, namespace: str) -> str:
    return hashlib.sha256(f"{seed}|{cohort}|{namespace}".encode()).hexdigest()[:24]


def _instant(anchor: datetime, days: int, hour: int) -> ClinicalInterval:
    value = anchor + timedelta(days=days, hours=hour)
    return ClinicalInterval(value, value, "instant")


def build_patient_pathway(
    spec: dict[str, Any], index: int, archetype: str, variant: str,
    birth_date: str, catalog_path: str, content_catalog_path: str | None = None,
    secondary_trait: str = "none", *, renal_stage4: bool = False, gender: str | None = None,
) -> PatientPathway:
    """Build organization-neutral clinical truth from a versioned template row."""
    catalog = load_catalog(catalog_path)
    content = load_catalog(content_catalog_path) if content_catalog_path else {"analytes": [], "medications": []}
    analytes = {(row["system"], row["code"]): row for row in content["analytes"]}
    templates = catalog["archetypes"][archetype]
    template = next((row for row in templates if row["id"] == variant), None)
    if template is None:
        raise ValueError(f"unknown pathway template {archetype}/{variant}")
    cohort = cohort_id(index)
    secondary_branch = template["secondaryBranches"][
        _choice(spec["seed"], cohort, "secondary-branch", len(template["secondaryBranches"]))
    ]
    anchor = datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00"))
    clinical_horizon = anchor - timedelta(microseconds=1)
    born = datetime.fromisoformat(birth_date + "T00:00:00+00:00")
    history_days = template.get("historyDays", 30)
    jitter = _choice(spec["seed"], cohort, "history-jitter", 121)
    earliest_first = born + timedelta(days=7)
    if variant == "dispense-follow-up":
        try:
            adult = born.replace(year=born.year + 18)
        except ValueError:
            adult = born.replace(year=born.year + 18, month=3, day=1)
        # The established diabetes assertion precedes the first prescription
        # by 30 days; the complete adult refill episode must fit in that life.
        earliest_first = max(earliest_first, adult + timedelta(days=30))
    if renal_stage4:
        try:
            adult = born.replace(year=born.year + 18)
        except ValueError:
            # completed_age_years counts a leap-day birthday on March 1.
            adult = born.replace(year=born.year + 18, month=3, day=1)
        # A secondary diagnosis may precede first by 30 days; its independent
        # renal confirmation precedes onset by another 100 days. Repeated
        # creatinine branches also reach back by up to 119 days.
        earliest_first = adult + timedelta(days=130)
        if earliest_first > clinical_horizon:
            raise ValueError("adult longitudinal renal evidence cannot fit before the clinical horizon")
    first = max(earliest_first, clinical_horizon - timedelta(days=history_days + jitter))
    spacing = list(template.get("eventSpacingDays", [0, 30, 90]))
    maximum_spacing = max(spacing, default=0)
    if first + timedelta(days=maximum_spacing) > clinical_horizon:
        first = max(earliest_first, clinical_horizon - timedelta(days=maximum_spacing))
    available_days = max(0, (clinical_horizon - first).days)
    if maximum_spacing > available_days:
        spacing = [round(offset * available_days / maximum_spacing) for offset in spacing]
    if variant == "diabetes-stable":
        tail_days = 60 if secondary_branch == "diet-review" else 0
        if secondary_branch == "retinal-referral":
            tail_days = 2 + 30 * bool(_choice(spec["seed"], cohort, "eye-referral-completion", 2))
        latest = max(spacing[2], spacing[1] + 180) + tail_days
        if (spacing[2] - spacing[1] != 180 or
                first + timedelta(days=latest, hours=10) > clinical_horizon):
            # Reserve the fixed six-month repeat and selected follow-up before
            # compressing the preceding history. Event hours need a day of slack.
            result_day = min(spacing[1], available_days - 1 - 180 - tail_days)
            if result_day < spacing[0]:
                raise ValueError("diabetes repeat and follow-up cannot fit before the clinical horizon")
            spacing[1:] = [result_day, result_day + 180]
    age_at_first = completed_age_years(born.date(), first.date())
    if variant == "age-eligible-screening":
        if age_at_first < 30:
            try:
                birthday = born.replace(year=born.year + 30)
            except ValueError:
                birthday = born.replace(year=born.year + 30, day=28)
            if birthday + timedelta(days=49, hours=12) > clinical_horizon:
                raise ValueError("screening result and review cannot fit after thirtieth birthday")
            first = birthday
            age_at_first = 30
        # The assessment precedes the actual order. A birthday in that gap can
        # change eligibility, so choose the test at the order event itself.
        order_age = completed_age_years(born.date(), (first + timedelta(days=spacing[1])).date())
        eligible = []
        if 50 <= order_age <= 74:
            eligible.append("mammography")
        if 30 <= order_age <= 54:
            eligible.append("cervical-screen")
        if not eligible:
            raise ValueError("routine screening template selected for an ineligible age")
        if secondary_branch not in eligible and secondary_branch != "result-review":
            secondary_branch = eligible[_choice(spec["seed"], cohort, "eligible-screen", len(eligible))]
        screen_choice = (secondary_branch if secondary_branch != "result-review" else
                         "cervical-screen" if "cervical-screen" in eligible and order_age < 50 else "mammography")
    facts = []
    events = []
    links: list[TemporalConstraint] = []
    episode_id = f"episode-{_stable_id(spec['seed'], cohort, 'primary-episode')}"
    historical_episode_id = f"episode-{_stable_id(spec['seed'], cohort, 'historical-episode')}"
    secondary_episode_id = f"episode-{_stable_id(spec['seed'], cohort, 'secondary-episode')}"
    primary_events: list[ClinicalEvent] = []
    primary_fact_ids: list[str] = []

    def fact_from_row(fact_id: str, row: dict[str, Any]) -> ClinicalFact:
        analyte = analytes.get((row["system"], row["code"]))
        if not analyte:
            return ClinicalFact(
                fact_id, row["family"], row["system"], row["code"], row["display"],
                row.get("status", "active"),
            )
        low, high = float(analyte["low"]), float(analyte["high"])
        guard = load_catalog(spec["clinicalDistributions"]["catalog"])["guardrails"][row["code"]]
        treated_cancer = archetype == "oncology" and any(
            event.role == "oncology-treatment" for event in events
        ) and any(fact.family in {"breast-cancer", "colon-cancer", "leukemia"} for fact in facts)
        modeled = quantitative_result(
            spec, index, "pathway", archetype, row["code"], low, high,
            guard["abnormalLow"], guard["abnormalHigh"], birth_date, None,
            secondary_trait=secondary_trait, pathway_variant=variant,
            occurrence_date=first.isoformat(),
            supported_oncology_treatment=treated_cancer,
            # The medication-exposure branch below emits active hypertension
            # and lisinopril 30 and 29 days before this pathway's first result.
            # Require adult eligibility and reject acute/safety variants here;
            # the observable branch audit independently verifies the emitted
            # condition, active regimen and observation order.
            supported_lisinopril_exposure=(
                secondary_trait == "medication-exposure"
                and completed_age_years(born.date(), first.date()) >= 18
                and not any(token in variant for token in ("acute", "emergency", "toxicity"))
                and not any(fact.code == "52175" for fact in facts)
                and not (
                    archetype == "medication-intensive" and content["medications"]
                    and content["medications"][(index - 1) % len(content["medications"])]["code"] == "52175"
                )  # Exclude concurrent or planned losartan exposure.
            ),
        )
        value = modeled["value"]
        return ClinicalFact(
            fact_id, row["family"], row["system"], row["code"], row["display"],
            row.get("status", "final"), round(value, 2), analyte["unit"], low, high,
            "H" if value > high else "L" if value < low else "N",
        )
    baseline_time = first - timedelta(days=200)
    if baseline_time >= born + timedelta(days=7):
        history_reasons = catalog["historicalEncounterReasons"]
        baseline_concept = history_reasons[
            _choice(spec["seed"], cohort, "historical-encounter-reason", len(history_reasons))
        ]
        baseline_fact_id = f"fact-{_stable_id(spec['seed'], cohort, 'fact-baseline-preventive')}"
        baseline_event_id = f"event-{_stable_id(spec['seed'], cohort, 'event-baseline-preventive')}"
        facts.append(ClinicalFact(
            baseline_fact_id, baseline_concept["family"],
            baseline_concept["system"], baseline_concept["code"],
            baseline_concept["display"], "completed",
        ))
        events.append(ClinicalEvent(
            baseline_event_id, historical_episode_id, baseline_fact_id, "encounter", "preventive",
            "community", "encounter", "finished", _instant(baseline_time, 0, 8),
        ))
    injury_variant = variant in {"minor-injury", "ordinary-injury"}
    for position, source_row in enumerate(template["events"]):
        row = source_row
        if variant == "dose-change" and position == 1 and secondary_branch == "switch-class":
            row = {**source_row, "code": "52175", "display": "Losartan"}
        if variant == "age-eligible-screening" and position in {1, 2}:
            screen = screen_choice
            if screen == "cervical-screen":
                concept = next(row for row in content["branchConcepts"] if row["id"] == "cervical-hpv-test")
                row = {**source_row, "family": "cervical-hpv-order" if position == 1 else "cervical-hpv-result",
                       "system": concept["system"], "code": concept["code"], "display": concept["display"]}
        if variant == "age-eligible-screening" and position == 0 and screen_choice == "cervical-screen":
            row = {**source_row, "status": "cervix-present"}
        elif variant == "age-eligible-screening" and position == 0:
            row = {**source_row, "status": "mammography-eligible"}
        if variant == "emergency-appendectomy" and position == 1 and secondary_branch != "open":
            concept = next(row for row in content["branchConcepts"] if row["id"] == "laparoscopic-appendectomy")
            row = {**source_row, "family": "laparoscopic-appendectomy", "system": concept["system"],
                   "code": concept["code"], "display": concept["display"]}
        if variant == "emergency-appendectomy" and position == 1 and secondary_branch == "open":
            row = {**source_row, "family": "open-appendectomy"}
        if variant == "diabetes-stable" and position == 2 and secondary_branch == "diet-review":
            row = {**source_row, "family": "diet-counselling"}
        if variant == "reproductive-follow-up" and secondary_branch == "nonpregnant-follow-up":
            if position == 0:
                row = {**source_row, "status": "contraception-preconception-review"}
            elif position == 1:
                row = {**source_row, "status": "nonpregnant"}
        if injury_variant and position < 2:
            code, display = INJURY_BRANCH_CONCEPTS[secondary_branch]
            setting = (
                "tertiary-emergency" if _choice(spec["seed"], cohort, "injury-emergency-setting", 5) < 2
                else "regional-emergency"
            )
            row = {**source_row, "code": code, "display": display, "setting": setting}
        elif archetype == "generally-healthy" and variant == "emergency-evaluation" and position < 2:
            setting = (
                "tertiary-emergency" if _choice(spec["seed"], cohort, "evaluation-emergency-setting", 3) == 0
                else "regional-emergency"
            )
            row = {**source_row, "setting": setting}
        fact_id = f"fact-{_stable_id(spec['seed'], cohort, f'fact-{position}-{row["family"]}')}"
        event_id = f"event-{_stable_id(spec['seed'], cohort, f'event-{position}-{row["role"]}')}"
        facts.append(fact_from_row(fact_id, row))
        offset = spacing[min(position, len(spacing) - 1)]
        occurrence = _instant(first, offset, 8 + position % 8)
        event_links = ()
        indication = None
        if position:
            prior_event = primary_events[-1]
            event_links = (EventLink(row.get("relationship", "follows"), prior_event.id),)
            links.append(TemporalConstraint(prior_event.id, event_id, 0, None))
            indication = prior_event.fact_id
        event = ClinicalEvent(
            event_id, episode_id, fact_id, row["role"], row.get("domain", "general"),
            row["setting"], row.get("capability", row["role"]), row.get("lifecycle", "completed"),
            occurrence, indication, event_links,
            provenance_required=row.get("shared", False),
        )
        events.append(event)
        primary_events.append(event)
        primary_fact_ids.append(fact_id)
    def add_branch_event(key: str, row: dict[str, Any], role: str, when: datetime,
                         predecessor: ClinicalEvent | None = None, *, episode: str = episode_id,
                         setting: str | None = None, lifecycle: str = "completed",
                         minimum: int = 0, maximum: int | None = None) -> ClinicalEvent:
        fid = f"fact-{_stable_id(spec['seed'], cohort, key)}"
        eid = f"event-{_stable_id(spec['seed'], cohort, key)}"
        facts.append(fact_from_row(fid, row))
        event = ClinicalEvent(
            eid, episode, fid, role, row.get("domain", "general"),
            setting or row["setting"], role, lifecycle,
            ClinicalInterval(when, when, "instant"),
            predecessor.fact_id if predecessor else None,
            (EventLink("follows", predecessor.id),) if predecessor else (),
        )
        events.append(event)
        if predecessor:
            links.append(TemporalConstraint(predecessor.id, eid, minimum, maximum))
        return event

    if variant == "medication-adjustment":
        previous = next(row for row in primary_events if row.role == "prescription")
        current = add_branch_event("adjusted-treatment", {
            "family": "adjusted-treatment", "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
            "code": "52175" if secondary_branch == "switch-class" else "29046",
            "display": "Losartan" if secondary_branch == "switch-class" else "Lisinopril", "setting": "community",
        }, "prescription", previous.occurrence.start + timedelta(days=30), previous, lifecycle="supersedes")
    promised = RESULT_CONTRACT["branchTests"].get(secondary_branch, [])
    for code in promised:
        predecessor = primary_events[-1]
        when = predecessor.occurrence.start + timedelta(hours=1)
        analyte = next((row for row in content["analytes"] if row["code"] == code), None)
        row = ({**analyte, "family": "branch-" + code, "setting": predecessor.care_setting}
               if analyte else {"family": "branch-" + code, "system": "http://loinc.org", "code": code,
                               "display": "CBC panel - Blood by Automated count" if code == "58410-2" else "Oxygen saturation by pulse oximetry",
                               "setting": predecessor.care_setting})
        if code == "59408-5":
            measured = add_branch_event("branch-oximetry", row, "result", primary_events[0].occurrence.start + timedelta(hours=1), primary_events[0])
            i = next(i for i,fact in enumerate(facts) if fact.id == measured.fact_id)
            facts[i] = replace(facts[i], value=float(97 + index % 3), unit="%", status="final")
        else:
            order = add_branch_event("branch-order-"+code, row, "order", when, predecessor)
            add_branch_event("branch-result-"+code, row, "result", when + timedelta(hours=2), order)
    if variant == "procedure-with-medication":
        surgery = next(row for row in primary_events if row.role == "procedure")
        administration = next(row for row in primary_events if row.role == "medication-administration")
        # The original month-late acetaminophen is an explicit postoperative
        # analgesic record, never evidence of prophylaxis or anesthesia.
        pos = events.index(administration)
        events[pos] = replace(administration, occurrence=ClinicalInterval(
            surgery.occurrence.start + timedelta(hours=2), surgery.occurrence.start + timedelta(hours=2), "instant"),
            links=(EventLink("part-of", surgery.id),))
        if secondary_branch == "anesthesia":
            anesthesia = add_branch_event("general-anesthesia", {
                "family":"general-anesthesia", "system":"http://snomed.info/sct", "code":"50697003",
                "display":"General anesthesia", "setting":"operating-room",
            }, "procedure", surgery.occurrence.start - timedelta(minutes=15))
            events[events.index(anesthesia)] = replace(anesthesia, links=(EventLink("part-of", surgery.id),))
        elif secondary_branch == "antibiotic-prophylaxis":
            prophylaxis = add_branch_event("perioperative-antibiotic", {
                "family":"perioperative-antibiotic", "system":"http://www.nlm.nih.gov/research/umls/rxnorm", "code":"2180",
                "display":"Cefazolin", "setting":"operating-room",
            }, "medication-administration", surgery.occurrence.start - timedelta(minutes=45))
            events[events.index(prophylaxis)] = replace(prophylaxis, links=(EventLink("part-of", surgery.id),))


    if variant == "age-eligible-screening":
        assessment, order, result = primary_events
        screen = "cervical-screen" if next(f for f in facts if f.id == result.fact_id).code == "82675-0" else "mammography"
        if screen == "cervical-screen":
            result_index = next(i for i, fact in enumerate(facts) if fact.id == result.fact_id)
            facts[result_index] = replace(facts[result_index], coded_value={
                "system": "http://snomed.info/sct", "code": "260385009", "display": "Negative",
            })
            specimen = add_branch_event("cervical-specimen", {
                "family": "cervical-specimen", "system": "", "code": "",
                "display": "Cervical specimen", "setting": "tertiary-ambulatory",
            }, "collection", order.occurrence.start + timedelta(days=1), order, minimum=0, maximum=2)
            events[events.index(result)] = replace(result, links=(EventLink("specimen", specimen.id),))
            links.append(TemporalConstraint(specimen.id, result.id, 7, 28))
        if secondary_branch == "result-review":
            add_branch_event("screen-result-review", {
                "family": "screen-result-review", "system": "http://snomed.info/sct",
                "code": "390906007", "display": "Follow-up encounter", "setting": "community-specialty",
            }, "follow-up", result.occurrence.start + timedelta(days=14), result, minimum=7, maximum=42)
        # The imaging center can assess its result and document a further risk
        # assessment. These are clinical actions that the HMO may receive as
        # summaries, distinct from its own community screening order.
        screening_mask = 1 + (index // 6) % 3
        if screening_mask & 1:
            add_branch_event("screen-center-result-assessment", {
                "family": "screen-center-result-assessment", "system": "http://snomed.info/sct",
                "code": "225337009", "display": "Risk assessment", "setting": "tertiary-ambulatory",
            }, "assessment", result.occurrence.start + timedelta(days=15), result, minimum=7, maximum=42)
        if screening_mask & 2:
            add_branch_event("screen-center-risk-assessment", {
                "family": "screen-center-risk-assessment", "system": "http://snomed.info/sct",
                "code": "225337009", "display": "Risk assessment", "setting": "tertiary-ambulatory",
            }, "assessment", result.occurrence.start + timedelta(days=16), result, minimum=7, maximum=42)
    if variant == "specialty-referral":
        referral = next(event for event in primary_events if event.role == "referral")
        if (index // 6) % 2:
            add_branch_event("hypertension-preconsult-creatinine", {
                "family": "hypertension-preconsult-creatinine", "system": "http://loinc.org",
                "code": "2160-0", "display": "Creatinine [Mass/volume] in Serum or Plasma",
                "setting": "community", "status": "final",
            }, "result", referral.occurrence.start + timedelta(hours=1), referral,
                minimum=0, maximum=1)
    if variant == "renal-monitoring" and (index // 6) % 2:
        creatinine = next(event for event in primary_events if event.role == "result")
        kidney_condition = next(event for event in primary_events if event.role == "condition")
        potassium_order = add_branch_event("renal-followup-potassium-order", {
            "family": "renal-followup-potassium-order", "system": "http://loinc.org",
            "code": "6298-4", "display": "Potassium [Moles/volume] in Blood",
            "setting": "community",
        }, "order", creatinine.occurrence.start + timedelta(days=1), kidney_condition,
            minimum=0)
        links.append(TemporalConstraint(creatinine.id, potassium_order.id, 1, 2))
        add_branch_event("renal-followup-potassium-result", {
            "family": "renal-followup-potassium-result", "system": "http://loinc.org",
            "code": "6298-4", "display": "Potassium [Moles/volume] in Blood",
            "setting": "community", "status": "final",
        }, "result", creatinine.occurrence.start + timedelta(days=2), potassium_order,
            minimum=0, maximum=2)
    if archetype == "perioperative-surgical":
        community_referral = next((event for event in primary_events
                                   if event.role == "referral" and event.care_setting == "community"), None)
        if community_referral is not None:
            # Distinct preoperative workups are linked to the actual referral.
            # The receiving surgical center may retain assessment summaries
            # and laboratory results without claiming it performed HMO care.
            workup_mask = 1 + index % 7
            if workup_mask & 1:
                add_branch_event("preoperative-risk-assessment", {
                    "family": "preoperative-risk-assessment", "system": "http://snomed.info/sct",
                    "code": "225337009", "display": "Risk assessment", "setting": "community",
                }, "assessment", community_referral.occurrence.start + timedelta(hours=1),
                    community_referral, minimum=0, maximum=1)
            if workup_mask & 2:
                hemoglobin_order = add_branch_event("preoperative-hemoglobin-order", {
                    "family": "preoperative-hemoglobin-order", "system": "http://loinc.org",
                    "code": "718-7", "display": "Hemoglobin [Mass/volume] in Blood",
                    "setting": "community",
                }, "order", community_referral.occurrence.start + timedelta(hours=1),
                    community_referral, minimum=0, maximum=1)
                add_branch_event("preoperative-hemoglobin", {
                    "family": "preoperative-hemoglobin", "system": "http://loinc.org",
                    "code": "718-7", "display": "Hemoglobin [Mass/volume] in Blood",
                    "setting": "community", "status": "final",
                }, "result", community_referral.occurrence.start + timedelta(hours=2),
                    hemoglobin_order, minimum=0, maximum=1)
            if workup_mask & 4:
                add_branch_event("preoperative-anesthesia-assessment", {
                    "family": "preoperative-anesthesia-assessment", "system": "http://snomed.info/sct",
                    "code": "225337009", "display": "Risk assessment", "setting": "community",
                }, "assessment", community_referral.occurrence.start + timedelta(hours=3),
                    community_referral, minimum=0, maximum=1)
    if variant == "diabetes-stable":
        condition, first_result, review = primary_events
        prior_fact = next(f for f in facts if f.id == first_result.fact_id)
        repeat = add_branch_event("diabetes-six-month-hba1c", {
            "family": "serial-hba1c", "system": prior_fact.system, "code": prior_fact.code,
            "display": prior_fact.display, "setting": "community", "status": "final",
        }, "result", first_result.occurrence.start + timedelta(days=180), first_result, minimum=150, maximum=210)
        repeat_index = next(i for i, fact in enumerate(facts) if fact.id == repeat.fact_id)
        repeated_fact = facts[repeat_index]
        if repeated_fact.value is not None and repeated_fact.reference_low is not None and repeated_fact.reference_high is not None:
            changed = min(repeated_fact.reference_high - .05,
                          max(repeated_fact.reference_low + .05,
                              repeated_fact.value + (.12 if index % 2 else -.12)))
            facts[repeat_index] = replace(repeated_fact, value=round(changed, 2))
        events[events.index(review)] = replace(review, links=(EventLink("reviews-result", repeat.id),),
                                                indication_fact_id=repeat.fact_id)
        links.append(TemporalConstraint(repeat.id, review.id, 0, 2))
        if secondary_branch == "diet-review":
            follow_up_order = add_branch_event("diet-follow-up-order", {
                "family": "diet-follow-up-order", "system": "http://snomed.info/sct",
                "code": "390906007", "display": "Follow-up encounter", "setting": "community-specialty",
            }, "order", review.occurrence.start + timedelta(hours=1), review)
            add_branch_event("diet-follow-up", {
                "family": "diet-follow-up", "system": "http://snomed.info/sct",
                "code": "390906007", "display": "Follow-up encounter", "setting": "community-specialty",
            }, "follow-up", review.occurrence.start + timedelta(days=60), follow_up_order, minimum=30, maximum=90)
        elif secondary_branch == "retinal-referral":
            eye_completed = bool(_choice(spec["seed"], cohort, "eye-referral-completion", 2))
            referral = add_branch_event("diabetic-eye-referral", {
                "family": "diabetic-eye-referral", "system": "", "code": "",
                "display": "Diabetic eye assessment", "setting": "community-specialty",
                "status": "completed" if eye_completed else "active",
            }, "referral", repeat.occurrence.start + timedelta(days=2), repeat,
                lifecycle="completed" if eye_completed else "active", minimum=0, maximum=30)
            if eye_completed:
                add_branch_event("diabetic-eye-review", {
                    "family": "diabetic-eye-review", "system": "http://snomed.info/sct",
                    "code": "390906007", "display": "Follow-up encounter", "setting": "tertiary-ambulatory",
                }, "follow-up", referral.occurrence.start + timedelta(days=30), referral, minimum=14, maximum=90)
    if variant == "dispense-follow-up":
        prescription, first_dispense, review = primary_events
        diabetes = add_branch_event("refill-established-diabetes", {
            "family": "type-2-diabetes", "system": "http://snomed.info/sct",
            "code": "44054006", "display": "Type 2 diabetes mellitus", "setting": "community",
        }, "condition", prescription.occurrence.start - timedelta(days=30))
        events[events.index(prescription)] = replace(prescription, links=(EventLink("reason-for", diabetes.id),),
                                                     indication_fact_id=diabetes.fact_id)
        links.append(TemporalConstraint(diabetes.id, prescription.id, 0, 60))
        second_day = 29 if secondary_branch != "late-refill" else 40
        authorization = prescription
        if _choice(spec["seed"], cohort, "refill-renewal", 2):
            authorization = add_branch_event("metformin-renewal", {
                "family": "metformin-renewal", "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                "code": "6809", "display": "Metformin", "setting": "community",
            }, "prescription", first_dispense.occurrence.start + timedelta(days=second_day - 3), first_dispense)
        if secondary_branch == "pharmacist-review":
            pharmacy_review = add_branch_event("pharmacist-encounter", {
                "family": "pharmacist-review", "system": "http://snomed.info/sct",
                "code": "390906007", "display": "Follow-up encounter", "setting": "pharmacy",
            }, "encounter", first_dispense.occurrence.start + timedelta(days=26), first_dispense)
            add_branch_event("pharmacist-reconciliation", {
                "family": "pharmacist-reconciliation", "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                "code": "6809", "display": "Metformin", "setting": "pharmacy",
            }, "medication-review", pharmacy_review.occurrence.start + timedelta(hours=1), pharmacy_review)
        second = add_branch_event("second-metformin-dispense", {
            "family": "metformin-refill", "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
            "code": "6809", "display": "Metformin", "setting": "pharmacy",
        }, "dispense", first_dispense.occurrence.start + timedelta(days=second_day), authorization)
        if secondary_branch == "late-refill":
            adherence = add_branch_event("late-adherence-encounter", {
                "family": "late-adherence-review", "system": "http://snomed.info/sct",
                "code": "390906007", "display": "Follow-up encounter", "setting": "pharmacy",
            }, "follow-up", second.occurrence.start + timedelta(days=1), second)
            add_branch_event("late-resumption-statement", {
                "family": "late-resumption", "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                "code": "6809", "display": "Metformin", "setting": "pharmacy",
            }, "medication-review", adherence.occurrence.start + timedelta(hours=1), adherence)
        pharmacy_actions = _choice(spec["seed"], cohort, "u03-pharmacy-actions", 8)
        if pharmacy_actions & 1:
            follow_up_order = add_branch_event("pharmacy-linked-review-order", {
                "family": "refill-follow-up-order", "system": "http://snomed.info/sct",
                "code": "390906007", "display": "Follow-up encounter", "setting": "pharmacy",
            }, "order", second.occurrence.start + timedelta(days=14), second,
                minimum=7, maximum=30)
            add_branch_event("pharmacy-linked-review", {
                "family": "refill-follow-up", "system": "http://snomed.info/sct",
                "code": "390906007", "display": "Follow-up encounter", "setting": "pharmacy",
            }, "follow-up", follow_up_order.occurrence.start + timedelta(days=1),
                follow_up_order, minimum=0, maximum=7)
        if pharmacy_actions & 2:
            add_branch_event("third-metformin-dispense", {
                "family": "metformin-repeat-fill", "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                "code": "6809", "display": "Metformin", "setting": "pharmacy",
            }, "dispense", second.occurrence.start + timedelta(days=30), authorization,
                minimum=25, maximum=80)
        if pharmacy_actions & 4:
            add_branch_event("pharmacy-reconciliation-after-fill", {
                "family": "metformin-reconciliation", "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                "code": "6809", "display": "Metformin", "setting": "pharmacy",
            }, "medication-review", second.occurrence.start + timedelta(days=2), second,
                minimum=1, maximum=7)
    if variant == "emergency-appendectomy":
        emergency, procedure, review = primary_events
        diagnosis = add_branch_event("documented-appendicitis", {
            "family": "documented-appendicitis", "system": "http://snomed.info/sct",
            "code": "74400008", "display": "Appendicitis", "setting": "tertiary-emergency",
        }, "condition", emergency.occurrence.start + timedelta(hours=1), emergency)
        events[events.index(procedure)] = replace(procedure, links=(EventLink("reason-for", diagnosis.id),),
                                                     indication_fact_id=diagnosis.fact_id)
        links.append(TemporalConstraint(diagnosis.id, procedure.id, 0, 2))
        recovery = add_branch_event("appendectomy-recovery", {
            "family": "uncomplicated-recovery", "system": "http://snomed.info/sct",
            "code": "390906007", "display": "Follow-up encounter",
            "setting": "tertiary-inpatient",
        }, "encounter" if secondary_branch == "short-stay" else "inpatient-encounter",
            procedure.occurrence.start + timedelta(hours=3), procedure)
        days = 0 if secondary_branch == "short-stay" else 3 if secondary_branch == "open" else 1
        discharge = add_branch_event("appendectomy-discharge", {
            "family": "appendectomy-discharge", "system": "http://snomed.info/sct",
            "code": "373942005", "display": "Discharge summary", "setting": recovery.care_setting,
        }, "discharge", procedure.occurrence.start + timedelta(days=days, hours=8), recovery)
        if secondary_branch == "short-stay":
            add_branch_event("short-stay-community-review", {
                "family": "short-stay-community-review", "system": "http://snomed.info/sct",
                "code": "185389009", "display": "Follow-up visit", "setting": "community",
            }, "follow-up", discharge.occurrence.start + timedelta(days=5), discharge, minimum=3, maximum=7)
    if archetype == "perioperative-surgical" and variant == "short-admission":
        # A regional admission followed by a day procedure has a locally
        # authored procedure order.  The order is linked to the admission and
        # precedes the procedure, rather than being inferred from its result.
        admission, procedure = primary_events[:2]
        procedure_fact = next(fact for fact in facts if fact.id == procedure.fact_id)
        fact_id = f"fact-{_stable_id(spec['seed'], cohort, 'regional-procedure-order')}"
        event_id = f"event-{_stable_id(spec['seed'], cohort, 'regional-procedure-order')}"
        facts.append(ClinicalFact(fact_id, "regional-procedure-order", procedure_fact.system,
                                  procedure_fact.code, procedure_fact.display, "active"))
        order_time = admission.occurrence.start + timedelta(hours=4)
        events.append(ClinicalEvent(event_id, episode_id, fact_id, "order", "surgical-care",
                                    "regional-inpatient", "order", "active",
                                    ClinicalInterval(order_time, order_time, "instant"),
                                    admission.fact_id, (EventLink("follows", admission.id),)))
        links.append(TemporalConstraint(admission.id, event_id, 0, 1))
        links.append(TemporalConstraint(event_id, procedure.id, 0, 1))

    # U03: these are optional, clinically linked episode actions. Two independent
    # decisions create observable graph combinations within each eligible care
    # setting; no value, label, or generator tag counts as a new signature.
    episode_settings = {
        "chronic-care": {"community-specialty", "tertiary-ambulatory"},
        "generally-healthy": {"regional-emergency", "tertiary-emergency"},
        "oncology": {"community", "tertiary-ambulatory"},
        "pediatric": {"regional-emergency", "tertiary-emergency"},
        "perioperative-surgical": {"day-procedure", "operating-room", "tertiary-ambulatory",
                                   "tertiary-emergency", "tertiary-inpatient", "regional-ambulatory"},
        "rare-complex": {"community", "tertiary-ambulatory", "regional-ambulatory"},
        "womens-health": {"community", "community-specialty", "tertiary-ambulatory", "tertiary-emergency", "regional-emergency"},
    }.get(archetype, set())
    for setting in sorted({event.care_setting for event in events if event.episode_id == episode_id} & episode_settings):
        setting_events = [event for event in events if event.episode_id == episode_id and event.care_setting == setting]
        anchor_event = max(setting_events, key=lambda event: event.occurrence.start)
        # Adjacent eligible cohorts cycle through distinct clinical action
        # combinations. Sparse source-local strata then retain actual graph
        # diversity without depending on a favorable random draw.
        ordinal = (index // 6 if archetype in {"perioperative-surgical", "pediatric"}
                   else index // 2 if archetype == "rare-complex" else index)
        mask = (ordinal + _choice(spec["seed"], 0, f"u03-episode-offset|{archetype}|{setting}", 16)) % 16
        if mask == 0:
            continue
        first_time = min(clinical_horizon, anchor_event.occurrence.start + timedelta(hours=1))
        second_time = min(clinical_horizon, anchor_event.occurrence.start + timedelta(hours=2))
        emergency = setting.endswith("emergency")
        surgery = archetype == "perioperative-surgical" and setting != "tertiary-ambulatory"
        if emergency:
            observation_code, observation_display = (
                ("390906007", "Follow-up encounter")
                if mask & 2 and archetype != "womens-health"
                else ("185389009", "Follow-up visit")
            )
            first_row = {
                "family": "brief-emergency-observation", "system": "http://snomed.info/sct",
                "code": observation_code, "display": observation_display, "setting": setting,
            }
            second_row = {
                "family": "emergency-discharge-summary", "system": "http://snomed.info/sct",
                "code": "373942005", "display": "Discharge summary", "setting": setting,
                "status": "current",
            }
            first_role, second_role = "emergency-encounter", "discharge"
        elif surgery:
            first_row = {
                "family": "post-procedure-recovery", "system": "http://snomed.info/sct",
                "code": "390906007", "display": "Follow-up encounter", "setting": setting,
            }
            second_row = {
                "family": "post-procedure-assessment", "system": "http://snomed.info/sct",
                "code": "225337009", "display": "Risk assessment", "setting": setting,
            }
            first_role, second_role = (
                "follow-up" if setting == "tertiary-inpatient" else "encounter"
            ), "assessment"
        else:
            review_code, review_display = (
                ("390906007", "Follow-up encounter") if archetype == "oncology"
                else ("185349003", "Encounter for check up") if archetype == "rare-complex"
                else ("185389009", "Follow-up visit")
            )
            first_row = {
                "family": "linked-clinical-review", "system": "http://snomed.info/sct",
                "code": review_code, "display": review_display, "setting": setting,
            }
            second_row = {
                "family": "linked-clinical-assessment", "system": "http://snomed.info/sct",
                "code": "225337009", "display": "Risk assessment", "setting": setting,
            }
            first_role, second_role = "follow-up", "assessment"
        first_extra = None
        if mask & 1:
            first_extra = add_branch_event(
                f"u03-{setting}-review", first_row, first_role,
                first_time, anchor_event, minimum=0, maximum=1,
            )
        if mask & 2:
            predecessor = first_extra or anchor_event
            add_branch_event(
                f"u03-{setting}-assessment", second_row, second_role,
                second_time, predecessor, minimum=0, maximum=1,
            )
        if mask & 4:
            inpatient_discharge = surgery and setting == "tertiary-inpatient"
            third_role = "discharge" if inpatient_discharge else "assessment"
            third_row = ({
                "family": "inpatient-recovery-discharge", "system": "http://snomed.info/sct",
                "code": "373942005", "display": "Discharge summary", "setting": setting,
                "status": "current",
            } if inpatient_discharge else {
                "family": "emergency-disposition-assessment", "system": "http://snomed.info/sct",
                "code": "225337009", "display": "Risk assessment", "setting": setting,
            } if emergency else {
                "family": "linked-review-reassessment", "system": "http://snomed.info/sct",
                "code": "225337009", "display": "Risk assessment", "setting": setting,
            })
            add_branch_event(
                f"u03-{setting}-order-or-reassessment", third_row, third_role,
                min(clinical_horizon, anchor_event.occurrence.start + timedelta(hours=3)),
                anchor_event, minimum=0, maximum=1,
            )
        if mask & 8:
            if emergency or archetype == "perioperative-surgical":
                later_code, later_display = "390906007", "Follow-up encounter"
            elif archetype == "chronic-care":
                # The observable action mix distinguishes a scheduled checkup,
                # assessment-led visit and fuller follow-up encounter.
                later_code, later_display = (
                    ("390906007", "Follow-up encounter")
                    if mask & 7 in {0, 1} or mask & 7 == 7 and variant == "renal-monitoring"
                    else ("185389009", "Follow-up visit") if mask & 7 in {2, 4, 6, 7}
                    else ("185349003", "Encounter for check up")
                )
            else:
                later_code, later_display = "185389009", "Follow-up visit"
            add_branch_event(
                f"u03-{setting}-later-review", {
                    "family": "linked-later-review", "system": "http://snomed.info/sct",
                    "code": later_code, "display": later_display, "setting": setting,
                }, "emergency-encounter" if emergency else "follow-up",
                min(clinical_horizon, anchor_event.occurrence.start + timedelta(hours=4)),
                anchor_event, minimum=0, maximum=1,
            )
    if archetype == "pediatric":
        bmi_event = next((event for event in events if next(fact for fact in facts if fact.id == event.fact_id).code == "39156-5"), None)
        if bmi_event is not None:
            paired: dict[str, ClinicalFact] = {}
            paired_height = round(150 + _choice(spec["seed"], cohort, "adolescent-height", 300) / 10, 1)
            paired_bmi = round(18.5 + _choice(spec["seed"], cohort, "adolescent-bmi", 650) / 100, 2)
            paired_weight = round(paired_bmi * (paired_height / 100) ** 2, 2)
            for code in ("29463-7", "8302-2"):
                analyte = analytes[("http://loinc.org", code)]
                fact_id = f"fact-{_stable_id(spec['seed'], cohort, f'bmi-input-{code}')}"
                event_id = f"event-{_stable_id(spec['seed'], cohort, f'bmi-input-{code}')}"
                value = paired_weight if code == "29463-7" else paired_height
                paired[code] = ClinicalFact(
                    fact_id, f"bmi-input-{code}", analyte["system"], code,
                    analyte["display"], "final", value, analyte["unit"],
                    analyte["low"], analyte["high"], "N",
                )
                facts.append(paired[code])
                events.append(ClinicalEvent(
                    event_id, episode_id, fact_id, "growth-measurement", "pediatric-growth",
                    "community", "growth-measurement", "final", bmi_event.occurrence,
                ))
            height_m = float(paired["8302-2"].value) / 100
            bmi_value = round(float(paired["29463-7"].value) / (height_m * height_m), 2)
            bmi_index = next(index for index, fact in enumerate(facts) if fact.id == bmi_event.fact_id)
            old = facts[bmi_index]
            bmi_age_years = completed_age_years(born.date(), bmi_event.occurrence.start.date())
            bmi_policy = age_applicability(content, "39156-5", bmi_age_years)
            reference_low = old.reference_low if bmi_policy["referenceRangeHandling"] == "required" else None
            reference_high = old.reference_high if bmi_policy["referenceRangeHandling"] == "required" else None
            interpretation = None
            if bmi_policy["interpretationHandling"] == "required" and reference_low is not None and reference_high is not None:
                interpretation = "H" if bmi_value > float(reference_high) else "L" if bmi_value < float(reference_low) else "N"
            facts[bmi_index] = ClinicalFact(
                old.id, old.family, old.system, old.code, old.display, old.status,
                bmi_value, old.unit, reference_low, reference_high, interpretation,
            )
    if archetype == "pediatric":
        vaccine_schedule = tuple(
            (row["code"], row["display"], row["minimumAgeDays"])
            for row in content.get("immunizations", [])
        ) or (
            ("871822003", "Hepatitis B vaccine", 1),
            ("871831003", "Diphtheria, tetanus and pertussis vaccine", 60),
            ("871839001", "Measles, mumps and rubella vaccine", 365),
        )
        existing_codes = {fact.code for fact in facts}
        for code, display, age_days in vaccine_schedule:
            occurrence_value = born + timedelta(days=age_days, hours=10)
            if code in existing_codes or occurrence_value > clinical_horizon:
                continue
            fact_id = f"fact-{_stable_id(spec['seed'], cohort, f'pediatric-vaccine-{code}')}"
            event_id = f"event-{_stable_id(spec['seed'], cohort, f'pediatric-vaccine-{code}')}"
            facts.append(ClinicalFact(
                fact_id, "routine-pediatric-immunization", "http://snomed.info/sct",
                code, display, "completed",
            ))
            events.append(ClinicalEvent(
                event_id, episode_id, fact_id, "immunization", "preventive",
                "community", "immunization", "completed",
                ClinicalInterval(occurrence_value, occurrence_value, "instant"),
            ))
        latest_growth = max(
            born + timedelta(days=7, hours=9),
            clinical_horizon - timedelta(days=30 + index % 60),
        )
        growth_dates = [latest_growth]
        earlier_growth = latest_growth - timedelta(days=180 + index % 90)
        if earlier_growth >= born + timedelta(days=30):
            growth_dates.insert(0, earlier_growth)
        for date_position, growth_date in enumerate(growth_dates):
            for code in ("29463-7", "8302-2"):
                analyte = analytes.get(('http://loinc.org', code))
                if not analyte:
                    continue
                fact_id = f"fact-{_stable_id(spec['seed'], cohort, f'pediatric-growth-{date_position}-{code}')}"
                event_id = f"event-{_stable_id(spec['seed'], cohort, f'pediatric-growth-{date_position}-{code}')}"
                row = {**analyte, "family": f"pediatric-growth-{code}", "status": "final"}
                facts.append(fact_from_row(fact_id, row))
                events.append(ClinicalEvent(
                    event_id, episode_id, fact_id, "growth-measurement", "pediatric-growth",
                    "community", "growth-measurement", "final",
                    ClinicalInterval(growth_date, growth_date, "instant"),
                ))
    if archetype in {"chronic-care", "medication-intensive", "oncology", "rare-complex"}:
        fact_by_id = {fact.id: fact for fact in facts}
        monitored = next((
            event for event in events
            if fact_by_id[event.fact_id].system == "http://loinc.org"
        ), None)
        if monitored:
            original = fact_by_id[monitored.fact_id]
            fact_id = f"fact-{_stable_id(spec['seed'], cohort, f'repeat-{original.code}')}"
            event_id = f"event-{_stable_id(spec['seed'], cohort, f'repeat-{original.code}')}"
            repeated_value = original.value
            if repeated_value is not None and original.reference_high is not None:
                repeated_value = round(min(original.reference_high, repeated_value * 1.01), 2)
            facts.append(ClinicalFact(
                fact_id, f"repeat-{original.family}", original.system, original.code,
                original.display, original.status, repeated_value, original.unit,
                original.reference_low, original.reference_high, original.interpretation,
            ))
            repeat_time = max(
                born + timedelta(days=7, hours=7),
                monitored.occurrence.start - timedelta(days=90 + index % 30),
            )
            events.append(ClinicalEvent(
                event_id, monitored.episode_id, fact_id, monitored.role,
                monitored.clinical_domain, monitored.care_setting, monitored.capability,
                monitored.lifecycle, ClinicalInterval(repeat_time, repeat_time, "instant"),
            ))
    if content["analytes"]:
        analyte = content["analytes"][(index - 1) % len(content["analytes"])]
        if not any(fact.code == analyte["code"] for fact in facts):
            fact_id = f"fact-{_stable_id(spec['seed'], cohort, f'supplemental-analyte-{analyte["code"]}')}"
            event_id = f"event-{_stable_id(spec['seed'], cohort, f'supplemental-analyte-{analyte["code"]}')}"
            row = {**analyte, "family": f"routine-{analyte['code']}", "status": "final"}
            occurrence_value = max(born + timedelta(days=7, hours=11), clinical_horizon - timedelta(days=15 + index % 120))
            facts.append(fact_from_row(fact_id, row))
            events.append(ClinicalEvent(
                event_id, episode_id, fact_id, "result", "routine-monitoring",
                "community", "result", "final",
                ClinicalInterval(occurrence_value, occurrence_value, "instant"),
                permitted_dispositions=("performed", "authored", "absent"),
            ))
        if archetype == "generally-healthy":
            background = content["analytes"][_choice(
                spec["seed"], cohort, "second-ordinary-background-analyte", len(content["analytes"])
            )]
            if not any(fact.code == background["code"] for fact in facts):
                fact_id = f"fact-{_stable_id(spec['seed'], cohort, f'second-background-{background["code"]}')}"
                event_id = f"event-{_stable_id(spec['seed'], cohort, f'second-background-{background["code"]}')}"
                row = {**background, "family": f"targeted-{background['code']}", "status": "final"}
                occurrence_value = max(
                    born + timedelta(days=7, hours=13),
                    clinical_horizon - timedelta(days=150 + _choice(spec["seed"], cohort, "second-background-offset", 181)),
                )
                facts.append(fact_from_row(fact_id, row))
                events.append(ClinicalEvent(
                    event_id, episode_id, fact_id, "result", "targeted-monitoring",
                    "community", "result", "final",
                    ClinicalInterval(occurrence_value, occurrence_value, "instant"),
                ))
        # A renal panel is indicated for ongoing cardiometabolic or complex
        # medication monitoring. Complete an existing potassium/creatinine pair
        # in the same clinical episode rather than adding an unrelated test.
        if archetype in {"chronic-care", "rare-complex"}:
            renal_codes = {"6298-4", "2160-0"}
            present = {fact.code for fact in facts}
            missing = renal_codes - present
            if len(missing) == 1:
                code = missing.pop()
                companion = next(row for row in content["analytes"] if row["code"] == code)
                fact_id = f"fact-{_stable_id(spec['seed'], cohort, f'renal-panel-{code}')}"
                event_id = f"event-{_stable_id(spec['seed'], cohort, f'renal-panel-{code}')}"
                occurrence_value = max(
                    born + timedelta(days=7, hours=11),
                    clinical_horizon - timedelta(days=15 + index % 120),
                )
                facts.append(fact_from_row(fact_id, {
                    **companion, "family": f"renal-panel-{code}", "status": "final",
                }))
                events.append(ClinicalEvent(
                    event_id, episode_id, fact_id, "result", "renal-monitoring",
                    "community", "result", "final",
                    ClinicalInterval(occurrence_value, occurrence_value, "instant"),
                    permitted_dispositions=("performed", "authored", "absent"),
                ))
    if archetype == "medication-intensive" and content["medications"]:
        candidates = content["medications"]
        existing = {fact.code for fact in facts}
        eligible = [row for row in candidates if compatible_ingredient(row["code"], existing, stage4=renal_stage4)]
        medication = eligible[(index - 1) % len(eligible)]
        if not any(fact.code == medication["code"] for fact in facts):
            fact_id = f"fact-{_stable_id(spec['seed'], cohort, f'supplemental-medication-{medication["code"]}')}"
            event_id = f"event-{_stable_id(spec['seed'], cohort, f'supplemental-medication-{medication["code"]}')}"
            occurrence_value = max(born + timedelta(days=7, hours=12), clinical_horizon - timedelta(days=45 + index % 180))
            facts.append(ClinicalFact(
                fact_id, f"reviewed-{medication['code']}", medication["system"],
                medication["code"], medication["display"], "active",
            ))
            events.append(ClinicalEvent(
                event_id, episode_id, fact_id, "prescription", "medication-management",
                "community", "prescription", "active",
                ClinicalInterval(occurrence_value, occurrence_value, "instant"),
            ))
    secondary_rows = {
        row["trait"]: row for row in content.get("secondaryClinicalBranches", [])
    }
    if (
        secondary_trait != "none"
        and archetype in {"chronic-care", "medication-intensive", "oncology", "rare-complex"}
    ):
        branch = secondary_rows.get(secondary_trait)
        if branch is None or archetype not in branch["eligibleArchetypes"]:
            raise ValueError(f"unsupported secondary clinical branch {archetype}/{secondary_trait}")
        condition = branch["condition"]
        condition_fact_id = f"fact-{_stable_id(spec['seed'], cohort, f'secondary-{secondary_trait}-condition')}"
        condition_event_id = f"event-{_stable_id(spec['seed'], cohort, f'secondary-{secondary_trait}-condition')}"
        secondary_time = max(
            born + timedelta(days=7, hours=10),
            min(first - timedelta(days=30), clinical_horizon - timedelta(days=30)),
        )
        facts.append(ClinicalFact(
            condition_fact_id, f"secondary-{secondary_trait}", condition["system"],
            condition["code"], condition["display"], condition["status"],
        ))
        condition_event = ClinicalEvent(
            condition_event_id, secondary_episode_id, condition_fact_id,
            "condition", "secondary-comorbidity", "community", "condition",
            condition["status"], ClinicalInterval(secondary_time, secondary_time, "instant"),
        )
        events.append(condition_event)
        medication_key = branch["medication"]
        if medication_key is not None:
            medication = next((
                row for row in content["medications"]
                if row["system"] == medication_key["system"] and row["code"] == medication_key["code"]
            ), None)
            if medication is None:
                raise ValueError(f"secondary branch {secondary_trait} references unknown medication")
            medication_fact_id = f"fact-{_stable_id(spec['seed'], cohort, f'secondary-{secondary_trait}-medication')}"
            medication_event_id = f"event-{_stable_id(spec['seed'], cohort, f'secondary-{secondary_trait}-medication')}"
            medication_time = secondary_time + timedelta(days=1)
            facts.append(ClinicalFact(
                medication_fact_id, f"secondary-treatment-{secondary_trait}", medication["system"],
                medication["code"], medication["display"], "active",
            ))
            events.append(ClinicalEvent(
                medication_event_id, secondary_episode_id, medication_fact_id,
                "prescription", "secondary-treatment", "community", "prescription",
                "active", ClinicalInterval(medication_time, medication_time, "instant"),
                condition_fact_id, (EventLink("reason-for", condition_event_id),),
            ))
            links.append(TemporalConstraint(condition_event_id, medication_event_id, 0, 30))
            if archetype == "oncology" and secondary_trait == "medication-exposure":
                # Antihypertensive treatment is reconciled before oncology
                # care. The treating hospital may receive these HMO-authored
                # reviews as summaries; each remains linked to the real Rx.
                review_mask = index % 4
                prescription_event = events[-1]
                if review_mask & 1:
                    add_branch_event("oncology-antihypertensive-follow-up", {
                        "family": "oncology-antihypertensive-follow-up",
                        "system": "http://snomed.info/sct", "code": "185389009",
                        "display": "Follow-up visit", "setting": "community",
                    }, "follow-up", medication_time + timedelta(days=1),
                        prescription_event, episode=secondary_episode_id, minimum=0, maximum=7)
                if review_mask & 2:
                    add_branch_event("oncology-antihypertensive-assessment", {
                        "family": "oncology-antihypertensive-assessment",
                        "system": "http://snomed.info/sct", "code": "225337009",
                        "display": "Risk assessment", "setting": "community",
                    }, "assessment", medication_time + timedelta(days=2),
                        prescription_event, episode=secondary_episode_id, minimum=0, maximum=7)
    # Every emitted BMI needs observable measurements on that same clinical day,
    # including BMI events added by a secondary branch after the pediatric setup.
    bmi_facts = {fact.id: fact for fact in facts}
    growth_anchors = content.get("modelDefinitions", {}).get("quantitative-observation", {}).get("valueModel", {}).get("syntheticGrowthAnchors", [])
    for bmi_event in list(events):
        bmi_fact = bmi_facts[bmi_event.fact_id]
        if bmi_fact.code != "39156-5":
            continue
        observed = {bmi_facts[event.fact_id].code for event in events
                    if event.occurrence.start.date() == bmi_event.occurrence.start.date()}
        if {"29463-7", "8302-2"} <= observed:
            continue
        age = (bmi_event.occurrence.start.date() - born.date()).days / 365.2425
        if archetype == "pediatric" and growth_anchors:
            lower = max((row for row in growth_anchors if row["ageYears"] <= age), key=lambda row: row["ageYears"], default=growth_anchors[0])
            upper = min((row for row in growth_anchors if row["ageYears"] >= age), key=lambda row: row["ageYears"], default=growth_anchors[-1])
            fraction = (age - lower["ageYears"]) / (upper["ageYears"] - lower["ageYears"]) if upper != lower else 0
            height_cm = round(lower["heightCm"] + (upper["heightCm"] - lower["heightCm"]) * fraction, 2)
        else:
            height_cm = round(150 + _choice(spec["seed"], cohort, f"bmi-height-{bmi_event.id}", 300) / 10, 1)
        bmi_value = float(bmi_fact.value) if bmi_fact.value is not None else 21.0
        values = {"8302-2": height_cm, "29463-7": round(bmi_value * (height_cm / 100) ** 2, 2)}
        frame = _choice(spec["seed"], cohort, "adult-body-frame", 100)
        if age >= 20 and frame < 4:
            # Uncommon large frames vary weight. The emitted paired values still
            # determine BMI and remain inside the reviewed physiologic bounds.
            values = {
                "8302-2": round(195 + _choice(spec["seed"], cohort, "large-height", 80) / 10, 1),
                "29463-7": round(121 + _choice(spec["seed"], cohort, "large-weight", 80) / 10, 1),
            }
        elif age >= 20 and frame < 12:
            height = round(201 + _choice(spec["seed"], cohort, "tall-height", 80) / 10, 1)
            values = {
                "8302-2": height,
                "29463-7": round(23.5 * (height / 100) ** 2, 2),
            }
        for code, value in values.items():
            analyte = analytes[("http://loinc.org", code)]
            fact_id = f"fact-{_stable_id(spec['seed'], cohort, f'{bmi_event.id}-input-{code}')}"
            event_id = f"event-{_stable_id(spec['seed'], cohort, f'{bmi_event.id}-input-{code}')}"
            fact = ClinicalFact(fact_id, f"bmi-input-{code}", analyte["system"], code,
                                analyte["display"], "final", value, analyte["unit"],
                                analyte["low"], analyte["high"], "N")
            facts.append(fact)
            bmi_facts[fact_id] = fact
            events.append(ClinicalEvent(event_id, bmi_event.episode_id, fact_id,
                                        "result", "body-measurement", bmi_event.care_setting,
                                        "result", "final", bmi_event.occurrence))
    # The source-local projection resolves links only to previously emitted events.
    # Branch builders may append retrospective facts, so order the graph by occurrence.
    events.sort(key=lambda row: (row.occurrence.start, row.id))
    # Resolve workflow actions against the preceding ingredient before projection.
    regimens = regimen_catalog(spec)
    if renal_stage4:
        facts = [replace(fact, code="1100699", display="Linagliptin")
                 if fact.system == "http://www.nlm.nih.gov/research/umls/rxnorm" and fact.code == "6809" else fact
                 for fact in facts]
    bp_policy = content.get("modelDefinitions", {}).get("quantitative-observation", {}).get("valueModel", {}).get("bloodPressurePair")
    fact_map = {fact.id: fact for fact in facts}
    previous_medication = {}
    previous_ingredient = {}
    workflow_actions = {row["family"]: row.get("workflowAction") for row in template["events"]}
    for position, event in enumerate(events):
        fact = fact_map[event.fact_id]
        if event.role in {"prescription", "dispense", "medication-review", "medication-administration"} and regimens:
            previous = previous_medication.get(event.episode_id)
            action = workflow_actions.get(fact.family) or (fact.code if fact.code in {"182840001", "430193006"} else None)
            if action:
                intended = [prior for prior in events[:position]
                            if prior.id in {link.target_event_id for link in event.links}
                            and prior.role in {"prescription", "dispense", "medication-review"}]
                if len(intended) != 1:
                    raise ValueError("medication workflow action requires one explicitly linked preceding ingredient")
                previous = intended[0]
                if previous is None:
                    raise ValueError("medication workflow action has no preceding ingredient")
                ingredient = fact_map[previous.fact_id]
                fact = replace(fact, system=ingredient.system, code=ingredient.code, display=ingredient.display)
            regimen = regimens[fact.code]
            detail = dict(regimen, exposureStart=event.occurrence.start.isoformat().replace("+00:00", "Z"), workflowAction=action)
            if "high-dose" in fact.family:
                detail["doseMg"] *= 2
            status = "stopped" if event.lifecycle == "stopped" else "completed" if event.role in {"dispense", "medication-administration"} else "active"
            if previous and fact_map[previous.fact_id].code == fact.code:
                prior_fact = fact_map[previous.fact_id]
                if status == "stopped" or event.lifecycle == "supersedes":
                    prior_detail = dict(prior_fact.medication_detail)
                    prior_detail["exposureEnd"] = detail["exposureStart"]
                    fact_map[previous.fact_id] = replace(prior_fact, status="stopped" if status == "stopped" else "completed", medication_detail=prior_detail)
                if status == "stopped":
                    detail["exposureStart"] = prior_fact.medication_detail["exposureStart"]
                    detail["exposureEnd"] = event.occurrence.start.isoformat().replace("+00:00", "Z")
                    detail["reason"] = "Temporary therapy hold documented at medication review"
                if event.lifecycle == "supersedes" or "restart" in fact.family:
                    events[position] = replace(event, links=(EventLink("supersedes", previous.id),))
            prior_ingredient = previous_ingredient.get(fact.code)
            if (status == "active" and prior_ingredient is not None
                    and fact_map[prior_ingredient.fact_id].status == "stopped"
                    and not any(link.relationship == "supersedes" for link in events[position].links)):
                events[position] = replace(
                    events[position],
                    links=events[position].links + (EventLink("supersedes", prior_ingredient.id),),
                )
            # Switching RAS class ends the preceding therapy across episodes.
            for other_event in events[:position]:
                other = fact_map[other_event.fact_id]
                if (status == "active" and other.medication_detail and other.status == "active"
                        and not compatible_ingredient(fact.code, {other.code})):
                    fact_map[other.id] = replace(other, status="stopped", medication_detail={
                        **other.medication_detail, "exposureEnd": detail["exposureStart"],
                        "reason": "Explicit switch to the subsequent RAS therapy"})
                    events[position] = replace(events[position], links=events[position].links +
                                               (EventLink("supersedes", other_event.id),))
            fact_map[fact.id] = replace(fact, status=status, medication_detail=detail)
            previous_medication[event.episode_id] = events[position]
            previous_ingredient[fact.code] = events[position]
        elif fact.value is not None:
            value = fact.value
            pair = None
            age = (event.occurrence.start.date() - born.date()).days / 365.2425
            if fact.code in {"8480-6", "8462-4"} and bp_policy:
                fraction = _choice(spec["seed"], cohort, "joint-blood-pressure", 1000) / 999
                child = age < bp_policy["childMaximumAgeYears"]
                sys_range = bp_policy["childSystolic" if child else "normalSystolic"]
                dia_range = bp_policy["childDiastolic" if child else "normalDiastolic"]
                pair = tuple(round(bounds[0] + (bounds[1] - bounds[0]) * fraction, 2) for bounds in (sys_range, dia_range))
                if not child and fact.interpretation == "H":
                    pair = tuple(bp_policy["mildHigh"])
                value = pair[0 if fact.code == "8480-6" else 1]
            growth = content.get("modelDefinitions", {}).get("quantitative-observation", {}).get("valueModel", {}).get("syntheticGrowthAnchors", [])
            if age < 18 and fact.code in {"29463-7", "8302-2"} and growth:
                lower = max((row for row in growth if row["ageYears"] <= age), key=lambda row: row["ageYears"], default=growth[0])
                upper = min((row for row in growth if row["ageYears"] >= age), key=lambda row: row["ageYears"], default=growth[-1])
                fraction = (age - lower["ageYears"]) / (upper["ageYears"] - lower["ageYears"]) if upper != lower else 0
                field_name = "weightKg" if fact.code == "29463-7" else "heightCm"
                value = round(lower[field_name] + (upper[field_name] - lower[field_name]) * fraction, 2)
            if age >= 18 and fact.code in {"29463-7", "8302-2"} and not fact.family.startswith("bmi-input"):
                height, weight = body_measurements(age, _choice(spec["seed"], cohort, "body-frame", 1000) / 999)
                value = weight if fact.code == "29463-7" else height
            policy = age_applicability(content, fact.code, completed_age_years(born.date(), event.occurrence.start.date()))
            if policy["referenceRangeHandling"] == "omit" or fact.code in {"29463-7", "8302-2"}:
                interpretation = None
                reference_low = None
                reference_high = None
            else:
                reference_low = fact.reference_low
                reference_high = fact.reference_high
                interpretation = None
                if policy["interpretationHandling"] == "required" and reference_low is not None and reference_high is not None:
                    interpretation = "H" if value > reference_high else "L" if value < reference_low else "N"
            fact_map[fact.id] = replace(
                fact, value=value, reference_low=reference_low, reference_high=reference_high,
                interpretation=interpretation, blood_pressure_pair=pair,
            )
    for event in events:
        fact = fact_map[event.fact_id]
        if fact.code == "39156-5":
            # A derived BMI is tied to the final measurements on its clinical
            # date.  The source observations may be documented at different
            # times (the infant defect had height/weight at 09:00 and BMI at
            # 11:00), so comparing complete intervals leaves an inconsistent
            # same-date triple untouched.  Keep the latest observation for
            # each input code, with event order as a deterministic tie-break.
            latest_same_date: dict[str, tuple[datetime, int, float]] = {}
            for position, row in enumerate(events):
                row_fact = fact_map[row.fact_id]
                if row_fact.code not in {"29463-7", "8302-2"} or row_fact.value is None:
                    continue
                if row.occurrence.start.date() != event.occurrence.start.date():
                    continue
                candidate = (row.occurrence.start, position, float(row_fact.value))
                previous = latest_same_date.get(row_fact.code)
                if previous is None or candidate[:2] >= previous[:2]:
                    latest_same_date[row_fact.code] = candidate
            paired = {code: candidate[2] for code, candidate in latest_same_date.items()}
            if all(code in paired for code in ("29463-7", "8302-2")):
                value = round(paired["29463-7"] / (paired["8302-2"] / 100) ** 2, 2)
                policy = age_applicability(content, fact.code, completed_age_years(born.date(), event.occurrence.start.date()))
                if policy["referenceRangeHandling"] == "omit" or fact.code in {"29463-7", "8302-2"}:
                    reference_low = None
                    reference_high = None
                    interpretation = None
                else:
                    reference_low = fact.reference_low
                    reference_high = fact.reference_high
                    interpretation = None
                    if policy["interpretationHandling"] == "required" and reference_low is not None and reference_high is not None:
                        interpretation = "H" if value > reference_high else "L" if value < reference_low else "N"
                fact_map[fact.id] = replace(
                    fact, value=value, reference_low=reference_low, reference_high=reference_high,
                    interpretation=interpretation,
                )
    if variant == "dispense-follow-up":
        original = next(row for row in events if row.role == "prescription" and
                        fact_map[row.fact_id].family == "metformin")
        renewal = next((row for row in events if row.role == "prescription" and
                        fact_map[row.fact_id].family == "metformin-renewal"), None)
        if renewal:
            prior = fact_map[original.fact_id]
            detail = dict(prior.medication_detail,
                          authorizationEnd=renewal.occurrence.start.isoformat().replace("+00:00", "Z"))
            fact_map[original.fact_id] = replace(prior, status="completed", medication_detail=detail)
            current = fact_map[renewal.fact_id]
            fact_map[renewal.fact_id] = replace(current, medication_detail={
                **current.medication_detail,
                "reason": "Original 30-tablet authorization exhausted after the first fill; stable therapy continues",
            })
    if variant in {"dose-change", "medication-adjustment"}:
        orders = [event for event in events if event.role == "prescription" and event.episode_id == episode_id]
        if len(orders) >= 2:
            prior, current = orders[:2]
            old, new = fact_map[prior.fact_id], fact_map[current.fact_id]
            start = current.occurrence.start.isoformat().replace("+00:00", "Z")
            old_detail = {**old.medication_detail, "doseMg": 10 if secondary_branch != "dose-decrease" else 20, "exposureEnd": start}
            new_detail = {**new.medication_detail, "doseMg": 20 if secondary_branch == "dose-increase" else 10}
            if secondary_branch == "switch-class":
                new = replace(new, code="52175", display="Losartan")
                new_detail = {**new_detail, **regimens["52175"]}
            fact_map[old.id] = replace(old, status="completed", medication_detail=old_detail)
            fact_map[new.id] = replace(new, status="active", medication_detail=new_detail)
            pos = events.index(current)
            events[pos] = replace(current, links=(EventLink("supersedes", prior.id),))
    # Plan every fill against its explicit authorization before source projection.
    event_map = {event.id: event for event in events}
    fills = {}
    for event in events:
        if event.role == "dispense":
            authorization = next((event_map[link.target_event_id] for link in event.links
                                  if event_map[link.target_event_id].role == "prescription"), None)
            if authorization:
                fills.setdefault(authorization.id, []).append(event)
    for authorization_id, history in fills.items():
        authorization = event_map[authorization_id]
        fact = fact_map[authorization.fact_id]
        fact_map[fact.id] = replace(fact, medication_detail={**fact.medication_detail,
            "plannedRepeats": len(history) - 1})
        for ordinal, event in enumerate(history):
            fact = fact_map[event.fact_id]
            fact_map[fact.id] = replace(fact, medication_detail={**fact.medication_detail,
                "fillType": "FFC" if ordinal == 0 else "RFC"})
    for event in events:
        fact = fact_map[event.fact_id]
        if renal_stage4 and fact.code == "2160-0":
            from .clinical_state import stage4_creatinine
            age = completed_age_years(born.date(), event.occurrence.start.date())
            value = stage4_creatinine(age, gender, 20 + index % 9)
            fact_map[fact.id] = replace(fact, value=value, interpretation="H")
        if fact.code == "19868-9" and age_years(birth_date, event.occurrence.start.isoformat()) < 6:
            fact_map[fact.id] = replace(fact, status="unsupported-age-method", value=None,
                                        reference_low=None, reference_high=None, interpretation=None)
    # Bind semantic order/result edges in the clinical model, not by source-local
    # nearest-event heuristics. An omitted compatible order stays external.
    for position, event in enumerate(events):
        fact = fact_map[event.fact_id]
        if event.role in {"result", "assessment", "growth-measurement"}:
            if fact.value is None and fact.coded_value is None:
                fact_map[fact.id] = replace(fact, result_text=RESULT_CONTRACT["narrativeResults"].get(fact.code))
            compatible = [candidate for candidate in events if candidate.role == "order"
                          and candidate.episode_id == event.episode_id
                          and candidate.occurrence.start <= event.occurrence.start
                          and order_compatible(fact.code, fact_map[candidate.fact_id].code)]
            retained = tuple(link for link in event.links if
                             next(row for row in events if row.id == link.target_event_id).role not in {"order", "referral"})
            if compatible:
                order = max(compatible, key=lambda row: row.occurrence.start)
                retained += (EventLink("result-of", order.id),)
            events[position] = replace(event, links=retained)
    facts = [fact_map[fact.id] for fact in facts]
    # Required sensitivity facts are explicit latent events, independent of
    # ordinary optional utilization and intentional-quality overlay selection.
    from .sensitivity import fact_for_patient, fact_rule, classify, owning_source
    sensitivity_fact = fact_for_patient(spec, index)
    if sensitivity_fact and spec.get("remediation"):
        rule = fact_rule(spec, sensitivity_fact)
        fact_time = sensitivity_first_sample(birth_date, spec["timelineAnchor"], rule.get("expectedResourceMultiplicity", 2))
        role = "result" if "hiv" in sensitivity_fact else "condition"
        semantic_role = "hiv-result" if role == "result" else {
            "mental-health-diagnosis": "mental-health-condition",
            "protected-domestic-violence-indicator": "violence-flag", "ordinary-injury": "ordinary-injury",
        }[sensitivity_fact]
        classify(spec, sensitivity_fact, semantic_role, birth_date, fact_time.date().isoformat())
        concept = ("http://loinc.org", "68961-2", "HIV 1 Ab [Presence] in Serum or Plasma by Immunoblot") if role == "result" else {
            "mental-health-diagnosis": ("http://snomed.info/sct", "35489007", "Depressive disorder"),
            "protected-domestic-violence-indicator": ("http://snomed.info/sct", "707087005", "At risk of domestic abuse"),
            "ordinary-injury": ("http://snomed.info/sct", "125666000", "Burn"),
        }[sensitivity_fact]
        infant_hiv = role == "result" and under_18_months(birth_date, fact_time.date().isoformat())
        if infant_hiv:
            concept = ("http://loinc.org", "5018-7", "HIV 1 RNA [Presence] in Specimen by NAA with probe detection")
        owner = owning_source(spec, sensitivity_fact)
        from .capture import load_capabilities
        setting = load_capabilities(spec['remediation']['capturePolicyCatalog'])[owner].care_settings[0]
        count = rule.get("expectedResourceMultiplicity", 2 if rule['confidentiality'] else 1)
        if infant_hiv:
            fact_time = min(fact_time, clinical_horizon - timedelta(days=count - 1))
            if fact_time < born:
                raise ValueError("independent infant HIV samples cannot fit in this life interval")
        episode = f"episode-{_stable_id(spec['seed'], cohort, sensitivity_fact)}"
        for offset in range(count):
            fid = f"fact-{_stable_id(spec['seed'], cohort, sensitivity_fact + str(offset))}"
            eid = f"event-{_stable_id(spec['seed'], cohort, sensitivity_fact + str(offset))}"
            value = ({"system": "http://snomed.info/sct", "code": "260385009" if sensitivity_fact.endswith("negative") else "260373001",
                      "display": "Negative" if sensitivity_fact.endswith("negative") else "Detected"} if role == "result" else None)
            facts.append(ClinicalFact(fid, sensitivity_fact, *concept, "final" if role == "result" else "active", coded_value=value))
            events.append(ClinicalEvent(eid, episode, fid, role, "sensitivity-fact", setting, role,
                "final" if role == "result" else "active", ClinicalInterval(
                    fact_time + timedelta(days=offset if infant_hiv else 0),
                    fact_time + timedelta(days=offset if infant_hiv else 0), "instant"),
                links=(EventLink("derived-from", events[-1].id),) if offset and not infant_hiv else (),
                sensitivity_fact=sensitivity_fact))
        if sensitivity_fact == "protected-domestic-violence-indicator":
            # Ordinary emergency triage is independent of the restricted
            # indicator: it does not reference either protected Condition.
            # The two tagged Conditions retain their required multiplicity.
            attendance = add_branch_event("concurrent-emergency-triage", {
                "family": "concurrent-emergency-triage", "system": "http://snomed.info/sct",
                "code": "50849002", "display": "Emergency care", "setting": setting,
            }, "emergency-encounter", fact_time - timedelta(hours=1),
                episode=episode)
            events[-1] = replace(attendance, clinical_domain="ordinary-triage",
                                 permitted_dispositions=("performed", "authored", "absent"))
            support_mask = index % 4
            if support_mask & 1:
                assessment = add_branch_event("concurrent-triage-risk-assessment", {
                    "family": "concurrent-triage-risk-assessment", "system": "http://snomed.info/sct",
                    "code": "225337009", "display": "Risk assessment", "setting": setting,
                }, "assessment", fact_time - timedelta(minutes=30), attendance,
                    episode=episode, minimum=0, maximum=1)
                events[-1] = replace(assessment, clinical_domain="ordinary-triage",
                                     permitted_dispositions=("performed", "authored", "absent"))
            if support_mask & 2:
                discharge = add_branch_event("concurrent-triage-discharge", {
                    "family": "concurrent-triage-discharge", "system": "http://snomed.info/sct",
                    "code": "373942005", "display": "Discharge summary", "setting": setting,
                    "status": "current",
                }, "discharge", fact_time + timedelta(hours=1), attendance,
                    episode=episode, minimum=0, maximum=1)
                events[-1] = replace(discharge, clinical_domain="ordinary-triage",
                                     permitted_dispositions=("performed", "authored", "absent"))
        if sensitivity_fact == "ordinary-injury":
            # The ordinary injury is a real regional emergency episode. Its
            # clinical assessment/disposition choices are separate from the
            # one required classification fact and vary by stable cohort index.
            injury = events[-1]
            attendance = add_branch_event("ordinary-injury-attendance", {
                "family": "ordinary-injury-attendance", "system": "http://snomed.info/sct",
                "code": "125666000", "display": "Burn", "setting": setting,
            }, "emergency-encounter", fact_time + timedelta(hours=1), injury,
                episode=episode, minimum=0, maximum=1)
            injury_mask = (index - 1 + _choice(spec["seed"], 0, "ordinary-injury-actions-offset", 8)) % 8
            if injury_mask & 1:
                add_branch_event("ordinary-injury-assessment", {
                    "family": "burn-severity-assessment", "system": "http://snomed.info/sct",
                    "code": "225337009", "display": "Risk assessment", "setting": setting,
                }, "assessment", fact_time + timedelta(hours=2), attendance,
                    episode=episode, minimum=0, maximum=1)
            if injury_mask & 2:
                add_branch_event("ordinary-injury-observation", {
                    "family": "brief-burn-observation", "system": "http://snomed.info/sct",
                    "code": "390906007", "display": "Follow-up encounter", "setting": setting,
                }, "emergency-encounter", fact_time + timedelta(hours=3), attendance,
                    episode=episode, minimum=0, maximum=1)
            if injury_mask & 4:
                add_branch_event("ordinary-injury-discharge", {
                    "family": "burn-discharge-summary", "system": "http://snomed.info/sct",
                    "code": "373942005", "display": "Discharge summary", "setting": setting,
                    "status": "current",
                }, "discharge", fact_time + timedelta(hours=4), attendance,
                    episode=episode, minimum=0, maximum=1)
    events = [replace(event, subject_id=cohort) for event in events]
    episode_ids = []
    for event in events:
        if event.episode_id not in episode_ids:
            episode_ids.append(event.episode_id)
    episodes = tuple(
        CareEpisode(
            current_episode,
            next(event for event in events if event.episode_id == current_episode).care_setting,
            next(event for event in events if event.episode_id == current_episode).capability,
            tuple(event.id for event in events if event.episode_id == current_episode),
            None,
        )
        for current_episode in episode_ids
    )
    pathway = PatientPathway(
        f"pathway-{_stable_id(spec['seed'], cohort, archetype + '|' + variant)}",
        cohort, archetype, variant,
        secondary_branch,
        secondary_trait, tuple(facts), tuple(events), episodes, tuple(links),
    )
    validate_pathway(pathway, born, anchor)
    return pathway


def validate_pathway(pathway: PatientPathway, birth: datetime, horizon: datetime) -> None:
    events = {row.id: row for row in pathway.events}
    facts = {row.id for row in pathway.facts}
    if len(events) != len(pathway.events) or len(facts) != len(pathway.facts):
        raise ValueError("duplicate latent pathway identifiers")
    for event in pathway.events:
        if event.fact_id not in facts:
            raise ValueError(f"latent event {event.id} references unknown fact")
        if event.subject_id != pathway.cohort_id:
            raise ValueError(f"latent event {event.id} has an invalid subject")
        if not event.participant_roles:
            raise ValueError(f"latent event {event.id} has no clinical participant role")
        if event.occurrence.start < birth:
            raise ValueError(f"latent event {event.id} predates birth")
        if event.occurrence.end > horizon:
            raise ValueError(f"latent event {event.id} exceeds timeline horizon")
        for link in event.links:
            if link.target_event_id not in events:
                raise ValueError(f"latent event {event.id} has an orphan link")
        if event.indication_fact_id is not None and event.indication_fact_id not in facts:
            raise ValueError(f"latent event {event.id} has an orphan indication")
    episode_ids = {row.id for row in pathway.episodes}
    if {event.episode_id for event in pathway.events} != episode_ids:
        raise ValueError("latent episode registry does not match event episode references")
    for episode in pathway.episodes:
        actual = tuple(event.id for event in pathway.events if event.episode_id == episode.id)
        if actual != episode.event_ids:
            raise ValueError(f"latent episode {episode.id} event registry mismatch")
    for constraint in pathway.constraints:
        left = events[constraint.predecessor_event_id].occurrence
        right = events[constraint.successor_event_id].occurrence
        if not strictly_before(left, right):
            raise ValueError(f"unsatisfied temporal constraint {constraint}")
        gap = (right.start - left.end).days
        if gap < constraint.minimum_days or (
            constraint.maximum_days is not None and gap > constraint.maximum_days
        ):
            raise ValueError(f"temporal constraint outside window {constraint}")
