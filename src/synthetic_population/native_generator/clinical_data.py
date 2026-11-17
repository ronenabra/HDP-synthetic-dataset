from __future__ import annotations

import hashlib
from datetime import date
from typing import Any


def _unit(seed: int, index: int, namespace: str) -> float:
    digest = hashlib.sha256(f"{seed}|clinical-data|{namespace}|cohort-{index:06d}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1)


def quantitative_result(spec: dict[str, Any], index: int, organization: str, archetype: str,
                        loinc: str, reference_low: float, reference_high: float,
                        ordinary_min: float, ordinary_max: float,
                        birth_date: str | None, gender: str | None,
                        secondary_trait: str = "none", pathway_variant: str = "",
                        occurrence_date: str | None = None,
                        supported_oncology_treatment: bool = False,
                        supported_lisinopril_exposure: bool = False) -> dict[str, Any]:
    """Seeded mixture with patient/subgroup effects and continuous abnormalities."""
    center = (reference_low + reference_high) / 2
    half = (reference_high - reference_low) / 2
    person = _unit(spec["seed"], index, f"person-effect|{loinc}") - .5
    # Related renal analytes share this between-person liability. It is keyed by
    # patient identity rather than source/analyte, so shuffling patient identity
    # destroys the observed joint distribution as required by R33.
    renal_liability = _unit(spec["seed"], index, "joint-effect|renal-potassium") - .5
    hematology_liability = _unit(spec["seed"], index, "joint-effect|hematology") - .5
    analytic = _unit(spec["seed"], index, f"analytic-effect|{organization}|{loinc}") - .5
    abnormal_draw = _unit(spec["seed"], index, f"abnormal-class|{organization}|{loinc}")
    direction_draw = _unit(spec["seed"], index, f"abnormal-direction|{loinc}")
    age = None
    if birth_date:
        anchor = date.fromisoformat((occurrence_date or spec["timelineAnchor"])[:10])
        born = date.fromisoformat(birth_date)
        age = anchor.year - born.year - ((anchor.month, anchor.day) < (born.month, born.day))

    if loinc in {"29463-7", "8302-2"} and birth_date:
        from .clinical_state import age_years, body_measurements
        event_age = age_years(birth_date, occurrence_date or spec["timelineAnchor"])
        height, weight = body_measurements(event_age, _unit(spec["seed"], index, "body-frame"))
        return {"value": weight if loinc == "29463-7" else height,
                "interpretation": "N", "abnormal": False, "effects": ["event-age-growth"],
                "trajectoryClass": "stable-normal", "latentEffects": {}}

    abnormal_probability = {
        "13457-7": .12, "1742-6": .12,
        "718-7": .055,
        "2345-7": .08, "4548-4": .08,
    }.get(loinc, .065)
    preferred = "high" if direction_draw >= .5 else "low"
    effects = []
    if loinc == "718-7" and archetype == "oncology" and (supported_oncology_treatment or not spec.get("remediation")):
        abnormal_probability += .24
        preferred = "low"
        effects.append("oncology-low-hemoglobin")
    if loinc == "718-7" and gender == "female":
        abnormal_probability += .035
        preferred = "low"
        effects.append("female-reference-stratum")
    if loinc == "6298-4" and archetype in {"chronic-care", "medication-intensive", "rare-complex"}:
        abnormal_probability += .08
        preferred = "high"
        effects.append("chronic-medication-potassium")
    if loinc == "6298-4" and secondary_trait == "medication-exposure" and (supported_lisinopril_exposure or not spec.get("remediation")):
        abnormal_probability += .16
        preferred = "high"
        effects.append("documented-medication-exposure")
    if loinc == "2160-0" and (age or 0) >= 65:
        abnormal_probability += .16
        preferred = "high"
        effects.append("older-adult-creatinine")
    if loinc == "2160-0" and archetype in {"chronic-care", "rare-complex"}:
        abnormal_probability += .035
        preferred = "high"
        effects.append("renal-condition-archetype")
    if loinc in {"6298-4", "2160-0"}:
        abnormal_probability += max(0.0, renal_liability) * .035
    if loinc == "718-7":
        abnormal_probability += max(0.0, -hematology_liability) * .02

    abnormal = abnormal_draw < abnormal_probability
    if abnormal and preferred == "high":
        headroom = max(0.0, ordinary_max - reference_high)
        excursion = (.15 + .75 * _unit(spec["seed"], index, f"high-excursion|{organization}|{loinc}")) * min(headroom, max(half, 1.0) * .30)
        value = reference_high + excursion
        interpretation = "H"
    elif abnormal:
        headroom = max(0.0, reference_low - ordinary_min)
        excursion = (.15 + .75 * _unit(spec["seed"], index, f"low-excursion|{organization}|{loinc}")) * min(headroom, max(half, 1.0) * .30)
        value = reference_low - excursion
        interpretation = "L"
    else:
        joint_shift = 0.0
        if loinc in {"6298-4", "2160-0"}:
            joint_shift = renal_liability * half * 1.1
        elif loinc == "718-7":
            joint_shift = hematology_liability * half * .35
        value = center + person * half * .85 + joint_shift + analytic * half * .20
        value = min(reference_high - .01, max(reference_low + .01, value))
        interpretation = "N"
    value = round(min(ordinary_max, max(ordinary_min, value)), 2)
    interpretation = "H" if value > reference_high else "L" if value < reference_low else "N"
    abnormal = interpretation != "N"
    acute_pathway = any(token in pathway_variant for token in ("acute", "emergency", "toxicity"))
    if abnormal and index % 3 == 0:
        trajectory_class = "persistent-mild-abnormal"
    elif abnormal:
        trajectory_class = "isolated-mild-abnormal-resolves"
    elif secondary_trait == "medication-exposure" and loinc in {"6298-4", "2160-0"}:
        trajectory_class = "medication-or-condition-affected"
    elif acute_pathway:
        trajectory_class = "acute-episode-recovery"
    elif index % 11 == 0:
        trajectory_class = "gradual-worsening"
    elif index % 5 == 0:
        trajectory_class = "gradual-improvement"
    else:
        trajectory_class = "stable-normal"
    return {
        "value": value,
        "interpretation": interpretation,
        "abnormal": abnormal,
        "effects": effects,
        "trajectoryClass": trajectory_class,
        "latentEffects": {
            "renalPotassium": round(renal_liability, 6),
            "hematology": round(hematology_liability, 6),
            "patientAnalyte": round(person, 6),
            "sourceAnalytic": round(analytic, 6),
        },
    }


def repeated_value(initial: float, interpretation: str, index: int, trajectory_class: str) -> float:
    magnitude = .03 + (index % 9) * .01
    if trajectory_class == "isolated-mild-abnormal-resolves":
        return round(initial * (1 - magnitude) if interpretation == "H" else initial * (1 + magnitude), 2)
    if trajectory_class == "gradual-improvement":
        return round(initial * (1 - magnitude / 2), 2)
    if trajectory_class == "gradual-worsening":
        return round(initial * (1 + magnitude / 2), 2)
    if trajectory_class == "acute-episode-recovery":
        return round(initial * (1 - magnitude) if interpretation == "H" else initial * (1 + magnitude), 2)
    if trajectory_class == "medication-or-condition-affected":
        return round(initial * (1 + ((index % 5) - 2) * .004), 2)
    if trajectory_class == "persistent-mild-abnormal":
        return round(initial * (1 + magnitude / 5), 2)
    return round(initial * (1 + ((index % 3) - 1) * .005), 2)
