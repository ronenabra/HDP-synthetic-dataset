import synthetic_population
"""Adversarial evidence for the S14C grammar and episode reconciliation."""

import copy
from pathlib import Path

import pytest

from synthetic_population.bundle import build_population
from synthetic_population.config import load_spec, scale_spec
from synthetic_population.grammar import PREDICATES, build_grammar_audit
from synthetic_population.observable import codings, is_supplemental_capture, patient_graphs
from synthetic_population.typed_grammar import ROLE_TYPES, compile_grammar
from synthetic_population.variability_v4 import build_variability_audit


ROOT = Path(synthetic_population.__file__).resolve().parent


@pytest.fixture(scope="module")
def small_population():
    spec = scale_spec(
        load_spec(ROOT / "data/specs/native.json"),
        100, "4.1.29-s14c-adversarial", zero_problems=True,
    )
    return spec, build_population(spec, ROOT)


RULE_CASES = (
    # Legacy name, selected typed variant, decisive event role, code, mode.
    ("resolved-minor-condition", "generally-healthy", "emergency-evaluation", "assessment", "386661006", "normal"),
    ("serial-monitoring", "medication-intensive", "polypharmacy-review", "medication-review", "83367", "normal"),
    ("chronic-medication", "chronic-care", "diabetes-stable", "result", "4548-4", "normal"),
    ("multiple-immunizations", "pediatric", "gastrointestinal-acute", "condition", "25374005", "normal"),
    ("medication-request-and-statement", "medication-intensive", "monitoring-result", "result", "2160-0", "normal"),
    ("medication-lifecycle", "medication-intensive", "monitoring-result", "result", "6298-4", "normal"),
    ("surgical-referral", "perioperative-surgical", "elective-procedure", "referral", "306206005", "normal"),
    ("surgical-procedure", "perioperative-surgical", "elective-procedure", "procedure", "38102005", "normal"),
    ("postoperative-follow-up", "perioperative-surgical", "elective-procedure", "follow-up", "185389009", "normal"),
    ("oncology-diagnosis", "oncology", "solid-tumor-systemic", "condition", "254837009", "normal"),
    ("oncology-confirmation", "oncology", "solid-tumor-surgical", "procedure", "387713003", "normal"),
    ("oncology-treatment", "oncology", "solid-tumor-surgical", "procedure", "387713003", "normal"),
    ("oncology-monitoring", "oncology", "solid-tumor-surgical", "follow-up", "390906007", "zero"),
    ("oncology-surveillance", "oncology", "solid-tumor-surgical", "follow-up", "390906007", "normal"),
    ("complex-follow-up", "rare-complex", "rare-metabolic", "follow-up", "390906007", "normal"),
)


@pytest.fixture(scope="module")
def canonical_zero_population():
    spec = scale_spec(
        load_spec(ROOT / "data/specs/native.json"),
        1000, "4.1.41-s14c-adversarial-zero", zero_problems=True,
    )
    return spec, build_population(spec, ROOT)


@pytest.mark.parametrize("name,archetype,variant,role,code,mode", RULE_CASES)
def test_each_named_typed_successor_rejects_missing_decisive_evidence(
    name, archetype, variant, role, code, mode, small_population, canonical_zero_population,
):
    spec, built = canonical_zero_population if mode == "zero" else small_population
    original = build_grammar_audit(built["bundles"], built["crosswalk"], spec)
    successor = f"{archetype}/{variant}/{name} -> typed:{archetype}.{variant}"
    assert original["failureCount"] == 0
    assert original["typedSuccessorReconciliations"].get(successor, 0) > 0
    clause = next(row for row in compile_grammar(spec["remediation"]["pathwayTemplateCatalog"])[archetype][variant]
                  if row.event_role == role and row.code == code)
    assert clause.minimum_distinct_events == 1
    graphs, _patients = patient_graphs(built["bundles"], spec)
    selected = next(row["cohortId"] for row in built["crosswalk"]["patients"]
                    if row["primaryArchetype"] == archetype
                    and row["pathwayVariant"] == variant
                    and not PREDICATES[name](graphs[row["cohortId"]], {})
                    and any(not is_supplemental_capture(resource)
                            and resource["resourceType"] in ROLE_TYPES[role]
                            and (clause.code_system, code) in set(codings(resource))
                            for resource in graphs[row["cohortId"]]))
    target_urls = {
        row["_fullUrl"] for row in graphs[selected]
        if not is_supplemental_capture(row)
        and row["resourceType"] in ROLE_TYPES[role]
        and (clause.code_system, code) in set(codings(row))
    }
    assert target_urls, (name, selected, role, code)
    mutated = copy.deepcopy(built["bundles"])
    for bundle in mutated.values():
        bundle["entry"] = [entry for entry in bundle["entry"] if entry["fullUrl"] not in target_urls]
    failed = build_grammar_audit(mutated, built["crosswalk"], spec)
    assert any(row["cohortId"] == selected and row["clause"] == name
               for row in failed["failures"])
    assert failed["typedSuccessorReconciliations"].get(successor, 0) < original["typedSuccessorReconciliations"][successor]


def test_supplemental_capture_enters_source_local_episode_grain(small_population):
    spec, built = small_population
    supplemental = [row for row in built["latentTruthLedger"]
                    if row["rowType"] == "supplemental-source-capture"]
    assert supplemental
    audited = build_variability_audit(
        built["bundles"], built["crosswalk"], spec, built["latentTruthLedger"],
    )
    episodes = {row["grainKey"] for row in audited["signatureTable"]
                if row["grain"] == "source-local-episode"
                and row["view"] == "role-topology"}
    assert all(f"{row['source']}|{row['episodeId']}|{row['careSetting']}" in episodes
               for row in supplemental)
    assert not any("observable-unassigned-setting" in episode for episode in episodes)
