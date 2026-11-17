import copy
import json
from dataclasses import asdict, replace
from datetime import date, datetime, timezone

import pytest

from synthetic_population.archetypes import primary_archetype
from synthetic_population.bundle import (
    _clinical_resources, _mark_scenario_evidence, _scenario_variant_resources,
)
from synthetic_population.capture import (
    capture_coverage_fixtures, capture_ledger_rows, load_capabilities, project_capture,
)
from synthetic_population.capture_class import capture_class_coding, CODE_SYSTEM_PATH
from synthetic_population.catalogs import load_catalog
from synthetic_population.care_pathway import EventLink, build_patient_pathway, validate_pathway
from synthetic_population.config import load_spec
from synthetic_population.pathways import care_pathway
from synthetic_population.predicates import evaluate_variant, subset_truth_table_variant
from synthetic_population.problems import allocation_table
from synthetic_population.ids import cohort_id, full_url
from synthetic_population.temporal import clinical_occurrences, history_decision, parse_fhir_time, strictly_before
from synthetic_population.typed_grammar import (
    SUPPORTED_PREDICATES, compile_grammar, evaluate_supported_predicate,
)
from synthetic_population.variability_v4 import _cosmetic_mutation, normalized_signature


def test_capture_class_codes_are_exact_catalog_pairs(repository_root):
    spec = load_spec(repository_root / "data/specs/native.json")
    catalog = load_catalog(spec["remediation"]["capturePolicyCatalog"])
    codes = {item["code"] for item in load_catalog(CODE_SYSTEM_PATH)["concept"]}
    expected = {f"{source}-{disposition}" for source, row in catalog["sources"].items()
                for disposition in row["enabledDispositions"]}
    assert codes == expected
    for source, row in catalog["sources"].items():
        for disposition in row["enabledDispositions"]:
            assert capture_class_coding(spec, source, disposition) == {
                "system": "urn:hdp-test-bundle:synthetic:capture-class",
                "code": f"{source}-{disposition}",
            }
    with pytest.raises(ValueError, match="invalid capture-class"):
        capture_class_coding(spec, "hmo", "referral-only")
    with pytest.raises(ValueError, match="invalid capture-class"):
        capture_class_coding(spec, "unknown", "performed")


def test_v4_latent_pathway_is_organization_neutral_and_capture_is_total(repository_root):
    spec = load_spec(repository_root / "data/specs/native.json")
    index = 1
    archetype = primary_archetype(spec, index)
    legacy = care_pathway(spec, index, archetype)
    pathway = build_patient_pathway(
        spec, index, archetype, legacy["pathwayVariant"], "1980-01-01",
        spec["remediation"]["pathwayTemplateCatalog"],
        spec["remediation"]["clinicalContentCatalog"],
    )
    captures, routes = project_capture(
        pathway, spec["remediation"]["capturePolicyCatalog"], spec["seed"],
    )
    assert set(captures) == {"hmo", "hospital-a", "hospital-b"}
    assert set(routes) == {event.id for event in pathway.events}
    for event in pathway.events:
        rows = [
            row for source_rows in captures.values() for row in source_rows
            if row.latent_event_id == event.id
        ]
        assert len(rows) == 3
        assert sum(row.disposition != "absent" for row in rows) >= 1


def test_latent_history_is_a_separate_episode_and_never_becomes_primary_indication(repository_root):
    spec = load_spec(repository_root / "data/specs/native.json")
    catalog = json.loads((repository_root / spec["remediation"]["pathwayTemplateCatalog"]).read_text())
    variant = catalog["archetypes"]["chronic-care"][0]["id"]
    pathway = build_patient_pathway(
        spec, 1, "chronic-care", variant, "1980-01-01",
        spec["remediation"]["pathwayTemplateCatalog"],
        spec["remediation"]["clinicalContentCatalog"], "none",
    )
    history = next(event for event in pathway.events if event.clinical_domain == "preventive")
    primary = [event for event in pathway.events if event.episode_id != history.episode_id]
    assert len(pathway.episodes) >= 2
    assert all(
        link.target_event_id != history.id
        for event in primary for link in event.links
    )
    assert all(event.subject_id == pathway.cohort_id for event in pathway.events)
    assert all(event.participant_roles for event in pathway.events)


@pytest.mark.parametrize("archetype", ["chronic-care", "medication-intensive", "oncology", "rare-complex"])
@pytest.mark.parametrize("trait", ["metabolic-risk", "respiratory-history", "medication-exposure"])
def test_intensive_secondary_traits_are_observable_coherent_episodes(repository_root, archetype, trait):
    spec = load_spec(repository_root / "data/specs/native.json")
    catalog = json.loads((repository_root / spec["remediation"]["pathwayTemplateCatalog"]).read_text())
    pathway = build_patient_pathway(
        spec, 2, archetype, catalog["archetypes"][archetype][0]["id"], "1980-01-01",
        spec["remediation"]["pathwayTemplateCatalog"],
        spec["remediation"]["clinicalContentCatalog"], trait,
    )
    secondary = [event for event in pathway.events if event.clinical_domain.startswith("secondary-")]
    assert any(event.role == "condition" for event in secondary)
    assert len({event.episode_id for event in secondary}) == 1
    if trait == "medication-exposure":
        medication = next(event for event in secondary if event.role == "prescription")
        condition = next(event for event in secondary if event.role == "condition")
        assert medication.indication_fact_id == condition.fact_id
        assert medication.links == (EventLink("reason-for", condition.id),)
    captures, routes = project_capture(pathway, spec["remediation"]["capturePolicyCatalog"], spec["seed"])
    ledger = capture_ledger_rows(pathway, captures, routes)
    latent = [row for row in ledger if row["rowType"] == "latent-event"]
    assert all(row["subjectId"] == pathway.cohort_id for row in latent)
    assert all(row["participantRoles"] for row in latent)
    assert all("permittedDispositions" in row and "provenanceRequired" in row for row in latent)


def _adult_pathway(spec, index=1):
    archetype = primary_archetype(spec, index)
    legacy = care_pathway(spec, index, archetype)
    return build_patient_pathway(
        spec, index, archetype, legacy["pathwayVariant"], "1980-01-01",
        spec["remediation"]["pathwayTemplateCatalog"],
        spec["remediation"]["clinicalContentCatalog"],
    )


def test_capture_catalog_has_explicit_fixture_for_every_enabled_branch(repository_root):
    spec = load_spec(repository_root / "data/specs/native.json")
    evidence = capture_coverage_fixtures(
        spec["remediation"]["capturePolicyCatalog"], spec["seed"],
    )
    assert evidence["status"] == "pass"
    assert evidence["fixtureCount"] == 17
    assert len(evidence["rows"]) == 21


@pytest.mark.parametrize("target_source", ["hmo", "hospital-a", "hospital-b"])
def test_capture_policy_mutation_is_source_isolated(repository_root, tmp_path, target_source):
    spec = load_spec(repository_root / "data/specs/native.json")
    path = repository_root / spec["remediation"]["capturePolicyCatalog"]
    catalog = json.loads(path.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(catalog)
    mutated["sources"][target_source]["omissionPercent"] = 100
    mutated_path = tmp_path / f"capture-{target_source}-omits.json"
    mutated_path.write_text(json.dumps(mutated), encoding="utf-8")
    changed = False
    for index in range(1, 31):
        pathway = _adult_pathway(spec, index)
        original, original_routes = project_capture(pathway, str(path), spec["seed"])
        updated, updated_routes = project_capture(pathway, str(mutated_path), spec["seed"])
        assert original_routes == updated_routes
        for source in {"hmo", "hospital-a", "hospital-b"} - {target_source}:
            assert original[source] == updated[source]
        changed = changed or original[target_source] != updated[target_source]
        assert pathway == _adult_pathway(spec, index)
    assert changed


def test_routing_mutation_changes_ownership_not_clinical_truth(repository_root, tmp_path):
    spec = load_spec(repository_root / "data/specs/native.json")
    policy_path = repository_root / spec["remediation"]["capturePolicyCatalog"]
    catalog = json.loads(policy_path.read_text(encoding="utf-8"))
    catalog["sources"]["hospital-a"]["careSettings"].append("community")
    mutated_path = tmp_path / "capture-routing-change.json"
    mutated_path.write_text(json.dumps(catalog), encoding="utf-8")
    ownership_changed = False
    for index in range(1, 31):
        pathway = _adult_pathway(spec, index)
        facts_and_links = (
            tuple(asdict(fact) for fact in pathway.facts),
            tuple((event.fact_id, tuple(asdict(link) for link in event.links)) for event in pathway.events),
        )
        _original_capture, original_routes = project_capture(pathway, str(policy_path), spec["seed"])
        _changed_capture, changed_routes = project_capture(pathway, str(mutated_path), spec["seed"])
        ownership_changed = ownership_changed or original_routes != changed_routes
        assert facts_and_links == (
            tuple(asdict(fact) for fact in pathway.facts),
            tuple((event.fact_id, tuple(asdict(link) for link in event.links)) for event in pathway.events),
        )
    assert ownership_changed


def test_capture_policy_schema_rejects_every_removed_active_field(repository_root, tmp_path):
    spec = load_spec(repository_root / "data/specs/native.json")
    catalog = json.loads((repository_root / spec["remediation"]["capturePolicyCatalog"]).read_text(encoding="utf-8"))
    for field in tuple(catalog["sources"]["hmo"]):
        mutated = copy.deepcopy(catalog)
        del mutated["sources"]["hmo"][field]
        path = tmp_path / f"capture-missing-{field}.json"
        path.write_text(json.dumps(mutated), encoding="utf-8")
        with pytest.raises(ValueError):
            load_capabilities(str(path))


def test_generation_rejects_illegal_capture_unsatisfied_time_and_orphan_link(repository_root):
    spec = load_spec(repository_root / "data/specs/native.json")
    pathway = _adult_pathway(spec)
    illegal_event = replace(pathway.events[0], permitted_dispositions=("absent",))
    illegal = replace(pathway, events=(illegal_event, *pathway.events[1:]))
    with pytest.raises(ValueError, match="illegal capture disposition"):
        project_capture(illegal, spec["remediation"]["capturePolicyCatalog"], spec["seed"])
    constraint = next(iter(pathway.constraints))
    reversed_constraint = replace(
        constraint, predecessor_event_id=constraint.successor_event_id,
        successor_event_id=constraint.predecessor_event_id,
    )
    unsatisfied = replace(pathway, constraints=(reversed_constraint,))
    with pytest.raises(ValueError, match="unsatisfied temporal constraint"):
        validate_pathway(
            unsatisfied, datetime(1980, 1, 1, tzinfo=timezone.utc),
            datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00")),
        )
    orphan_event = replace(pathway.events[0], links=(EventLink("follows", "missing-event"),))
    orphan = replace(pathway, events=(orphan_event, *pathway.events[1:]))
    with pytest.raises(ValueError, match="orphan link"):
        validate_pathway(
            orphan, datetime(1980, 1, 1, tzinfo=timezone.utc),
            datetime.fromisoformat(spec["timelineAnchor"].replace("Z", "+00:00")),
        )


def test_temporal_day_precision_does_not_invent_an_order():
    first = parse_fhir_time("2026-01-01")
    same_day = parse_fhir_time("2026-01-01T23:00:00Z")
    next_day = parse_fhir_time("2026-01-02T00:00:00Z")
    assert not strictly_before(first, same_day)
    assert strictly_before(first, next_day)


def test_history_gate_ignores_demographics_and_requires_three_distinct_events():
    patient = {
        "resourceType": "Patient",
        "name": [{"family": "Example", "period": {"start": "2010-01-01", "end": "2025-01-01"}}],
        "address": [{"period": {"start": "2010-01-01", "end": "2025-01-01"}}],
    }
    one_event = {
        "resourceType": "MedicationStatement", "status": "active",
        "effectivePeriod": {"start": "2020-01-01", "end": "2025-01-01"},
    }
    assert not history_decision(clinical_occurrences([patient, one_event]), 365)["passed"]
    encounters = [
        {"resourceType": "Encounter", "status": "finished", "period": {"start": when, "end": when}}
        for when in ("2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z")
    ]
    baseline = history_decision(clinical_occurrences(encounters), 730)
    assert baseline["passed"]
    mutated_patient = copy.deepcopy(patient)
    mutated_patient["name"][0]["period"] = {"start": "1900-01-01", "end": "2099-01-01"}
    assert history_decision(clinical_occurrences([mutated_patient, *encounters]), 730) == baseline


def test_variability_normalizer_ignores_cosmetic_fields_but_detects_structure():
    resources = [{
        "resourceType": "Observation", "status": "final", "_source": "hmo",
        "_fullUrl": "urn:uuid:one", "id": "one",
        "meta": {"profile": ["http://hl7.org/fhir/StructureDefinition/Observation"]},
        "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "HbA1c"}]},
        "effectiveDateTime": "2026-01-01T00:00:00Z",
        "valueQuantity": {"value": 6.1, "code": "%"},
        "subject": {"reference": "urn:uuid:patient"},
    }]
    cosmetic = _cosmetic_mutation(resources)
    for view in ("role-topology", "semantic"):
        original = normalized_signature(resources, view)[0]
        assert normalized_signature(cosmetic, view)[0] == original
        material = copy.deepcopy(resources)
        material.append({"resourceType": "Condition", "status": "active"})
        assert normalized_signature(material, view)[0] != original


def test_all_declared_typed_predicates_have_executable_semantics():
    loinc = "http://loinc.org"
    snomed = "http://snomed.info/sct"
    rxnorm = "http://www.nlm.nih.gov/research/umls/rxnorm"
    resources = [
        {"resourceType": "Observation", "status": "final", "_evidenceId": "obs-1", "_observableSource": "hmo", "_fullUrl": "Observation/one", "code": {"coding": [{"system": loinc, "code": "A"}]}, "effectiveDateTime": "2025-01-01T00:00:00Z", "valueQuantity": {"value": 1}},
        {"resourceType": "Observation", "status": "final", "_evidenceId": "obs-2", "_observableSource": "hospital-a", "_fullUrl": "Observation/two", "code": {"coding": [{"system": loinc, "code": "A"}]}, "effectiveDateTime": "2025-02-01T00:00:00Z", "valueQuantity": {"value": 2}},
        {"resourceType": "Condition", "_evidenceId": "condition", "code": {"coding": [{"system": snomed, "code": "C"}]}, "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]}, "onsetDateTime": "2024-01-01T00:00:00Z"},
        {"resourceType": "ServiceRequest", "status": "completed", "_evidenceId": "order", "_fullUrl": "ServiceRequest/order", "code": {"coding": [{"system": loinc, "code": "O"}]}, "authoredOn": "2025-01-15T00:00:00Z"},
        {"resourceType": "Observation", "status": "final", "_evidenceId": "linked", "code": {"coding": [{"system": loinc, "code": "R"}]}, "effectiveDateTime": "2025-01-16T00:00:00Z", "basedOn": [{"reference": "ServiceRequest/order"}]},
        {"resourceType": "MedicationRequest", "status": "active", "_evidenceId": "med-1", "medicationCodeableConcept": {"coding": [{"system": rxnorm, "code": "M"}]}, "authoredOn": "2025-01-01T00:00:00Z", "dosageInstruction": [{"doseAndRate": [{"doseQuantity": {"value": 5}}]}]},
        {"resourceType": "MedicationRequest", "status": "active", "_evidenceId": "med-2", "medicationCodeableConcept": {"coding": [{"system": rxnorm, "code": "M"}]}, "authoredOn": "2025-03-01T00:00:00Z", "dosageInstruction": [{"doseAndRate": [{"doseQuantity": {"value": 10}}]}]},
    ]
    results = {
        "distinct-event-count": evaluate_supported_predicate("distinct-event-count", resources, system=loinc, code="A", minimum=2),
        "same-analyte-repeats": evaluate_supported_predicate("same-analyte-repeats", resources, system=loinc, code="A"),
        "trend-direction": evaluate_supported_predicate("trend-direction", resources, system=loinc, code="A", direction="increasing"),
        "active-resolved-status": evaluate_supported_predicate("active-resolved-status", resources, system=snomed, code="C", statuses=("active",)),
        "medication-exposure": evaluate_supported_predicate("medication-exposure", resources, system=rxnorm, code="M"),
        "dose-status-transition": evaluate_supported_predicate("dose-status-transition", resources, system=rxnorm, code="M"),
        "linked-events": evaluate_supported_predicate("linked-events", resources, system=loinc, code="R", second_system=loinc, second_code="O"),
        "age-sex-risk-eligibility": evaluate_supported_predicate("age-sex-risk-eligibility", resources, patient={"birthDate": "1980-01-01", "gender": "female"}, event_date=date(2025, 1, 1), minimum_age_days=18 * 365, permitted_sexes=("female",)),
        "temporal-relation": evaluate_supported_predicate("temporal-relation", resources, system=snomed, code="C", second_system=loinc, second_code="O", relation="strictly-before"),
        "source-participation": evaluate_supported_predicate("source-participation", resources, system=loinc, code="A", minimum=2),
        "forbidden-combination": evaluate_supported_predicate("forbidden-combination", resources, system=snomed, code="C", second_system=snomed, second_code="absent"),
    }
    assert set(results) == SUPPORTED_PREDICATES
    assert all(results.values()), results
    with pytest.raises(ValueError, match="unknown typed predicate"):
        evaluate_supported_predicate("decorative-count", resources)


def test_typed_grammar_schema_rejects_decorative_catalog_fields(repository_root, tmp_path):
    spec = load_spec(repository_root / "data/specs/native.json")
    source = repository_root / spec["remediation"]["pathwayTemplateCatalog"]
    catalog = json.loads(source.read_text(encoding="utf-8"))
    catalog["unusedNarrativePromise"] = True
    path = tmp_path / "grammar-with-decoration.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(ValueError, match="unused pathway/grammar catalog fields"):
        compile_grammar(str(path))


def _tag(resource):
    resource.setdefault("meta", {})["tag"] = [{
        "system": "urn:hdp-test-bundle:synthetic-scenario-role",
        "code": "positive-evidence",
    }]
    return resource


def test_culture_treatment_requires_order_collection_result_then_treatment():
    resources = [
        _tag({
            "resourceType": "AllergyIntolerance", "criticality": "high",
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical", "code": "active"}]},
            "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification", "code": "confirmed"}]},
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": "91936005"}]},
            "onsetDateTime": "2025-01-01", "reaction": [{"severity": "severe"}],
        }),
        _tag({
            "resourceType": "ServiceRequest", "status": "completed", "authoredOn": "2026-01-01T08:00:00Z",
            "code": {"coding": [{"system": "http://loinc.org", "code": "630-4"}]},
        }),
        _tag({
            "resourceType": "Specimen", "status": "available",
            "collection": {"collectedDateTime": "2026-01-01T09:00:00Z"},
        }),
        _tag({
            "resourceType": "Observation", "status": "final", "effectiveDateTime": "2026-01-02T09:00:00Z",
            "code": {"coding": [{"system": "http://loinc.org", "code": "630-4"}]},
        }),
        _tag({
            "resourceType": "MedicationRequest", "status": "active", "authoredOn": "2026-01-03T09:00:00Z",
            "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "723"}]},
            "dosageInstruction": [{
                "timing": {"repeat": {"frequency": 3, "period": 1, "periodUnit": "d"}},
                "route": {"coding": [{"system": "http://snomed.info/sct", "code": "26643006"}]},
                "doseAndRate": [{"doseQuantity": {"value": 500, "code": "mg"}}],
            }],
            "courseOfTherapyType": {"coding": [{"code": "acute"}]},
        }),
    ]
    assert evaluate_variant(
        "allergy-treatment-risk", "culture-informed-treatment", resources,
        require_three_source_context=True,
    )
    assert not evaluate_variant(
        "allergy-treatment-risk", "culture-informed-treatment",
        [row for row in resources if row["resourceType"] != "Specimen"],
        require_three_source_context=True,
    )


def test_every_canonical_scenario_allocation_requires_all_three_sources(repository_root):
    spec = load_spec(repository_root / "data/specs/native.json")
    expected = {
        "hmo": False, "hospital-a": False, "hospital-b": False,
        "hmo+hospital-a": False, "hmo+hospital-b": False,
        "hospital-a+hospital-b": False, "hmo+hospital-a+hospital-b": True,
    }
    exercised = []
    for index, problems in allocation_table(spec).items():
        if not problems or problems[0]["class"] != "cross-source-clinical":
            continue
        problem = problems[0]
        resources_by_source = {}
        for organization in spec["organizations"]:
            source = organization["id"]
            patient_url = full_url(spec["datasetVersion"], source, cohort_id(index), "Patient")
            clinical = _clinical_resources(
                spec, organization, index, patient_url, problems, "1980-01-01",
                full_capture=True, gender="female",
            )
            evidence = _scenario_variant_resources(
                spec, source, index, patient_url, problem, clinical,
            )
            assert all(
                resource.get('category') is None or isinstance(resource['category'], dict)
                for _, resource in clinical
                if resource['resourceType'] == 'MedicationAdministration'
            )
            _mark_scenario_evidence(clinical, evidence, "positive-evidence")
            from synthetic_population.blind import blind_resource
            resources_by_source[source] = [{**blind_resource(resource, spec), "_fullUrl": url, "_source": source} for url, resource in clinical]
        actual = subset_truth_table_variant(
            problem["scenario"], problem["variant"], resources_by_source,
            require_three_source_context=True,
        )
        assert actual == expected, (index, problem, actual)
        from synthetic_population.scenario_audit import _counterfactuals
        mutations = _counterfactuals(problem["scenario"], problem["variant"], [resource for rows in resources_by_source.values() for resource in rows])
        assert all(row["passed"] for row in mutations), (problem, mutations)
        exercised.append((problem["scenario"], problem["variant"]))
    assert len(exercised) == spec["problems"]["crossSourceClinicalPatients"]
    assert set(exercised) == {
        ("renal-medication-risk", "chronic-renal-trend"),
        ("renal-medication-risk", "acute-renal-precipitant"),
        ("renal-medication-risk", "medication-dose-escalation"),
        ("bleeding-risk", "overt-bleeding"),
        ("bleeding-risk", "falling-hemoglobin"),
        ("bleeding-risk", "interacting-exposure"),
        ("allergy-treatment-risk", "prescribing-after-allergy"),
        ("allergy-treatment-risk", "administration-after-allergy"),
        ("allergy-treatment-risk", "culture-informed-treatment"),
    }


def test_expected_profile_and_bucket_follow_clinical_role_not_declaration(repository_root):
    from synthetic_population.profiles import expected_profile, HDP
    from synthetic_population.buckets import observable_role, bucket_for_role
    lab = {'resourceType': 'Observation', 'category': [{'coding': [{'code': 'LAB'}]}],
           'code': {'coding': [{'system': 'http://loinc.org', 'code': '6298-4'}]}}
    vital = {'resourceType': 'Observation', 'code': {'coding': [{'system': 'http://loinc.org', 'code': '8302-2'}]}}
    assessment = {'resourceType': 'Observation', 'category': [{'coding': [{'code': 'exam'}]}]}
    for declared in ([], ['http://hl7.org/fhir/StructureDefinition/Observation'], [HDP + 'il-hdp-vital-signs']):
        lab['meta'] = {'profile': declared}
        assert expected_profile(lab) == HDP + 'il-hdp-observation-lab'
        assert bucket_for_role(observable_role(lab)) == 'laboratoryTests'
    assert bucket_for_role(observable_role(vital)) == 'measurements'
    assert bucket_for_role(observable_role(assessment)) == 'assessments'
    respiratory_assessment = {
        'resourceType': 'Observation', 'category': [{'coding': [{'code': 'exam'}]}],
        'code': {'coding': [{'system': 'http://loinc.org', 'code': '19868-9'}]},
    }
    assert expected_profile(respiratory_assessment) == HDP + 'il-hdp-observation'
    assert bucket_for_role(observable_role(respiratory_assessment)) == 'assessments'


def test_service_request_supporting_info_requires_allowed_target_type():
    from types import SimpleNamespace
    from synthetic_population.fhir_projection import _resource
    fact = SimpleNamespace(system='http://loinc.org', code='24331-1', display='Lipid panel')
    occurrence = SimpleNamespace(start=datetime(2025, 1, 1, tzinfo=timezone.utc), end=datetime(2025, 1, 1, tzinfo=timezone.utc))
    disposition = SimpleNamespace()
    def request(prior_type):
        return _resource('order', fact, occurrence, 'urn:uuid:patient', 'urn:uuid:practitioner',
                         'urn:uuid:organization', 'urn:uuid:prior', prior_type, disposition)
    assert 'supportingInfo' not in request('Encounter')
    assert request('Condition')['supportingInfo'] == [{'reference': 'urn:uuid:prior'}]


def test_qualitative_hiv_il_core_lab_has_required_laboratory_category(repository_root):
    from types import SimpleNamespace
    from synthetic_population.buckets import observable_role
    from synthetic_population.fhir_projection import _profile_fields
    from synthetic_population.metadata import meta
    from synthetic_population.profiles import expected_profile

    instant = datetime(2025, 1, 1, tzinfo=timezone.utc)
    fact = SimpleNamespace(id='hiv-fact', code='68961-2')
    event = SimpleNamespace(id='hiv-result', episode_id='screening', occurrence=SimpleNamespace(start=instant),
                            indication_fact_id='hiv-fact', role='result', links=())
    pathway = SimpleNamespace(cohort_id='cohort-000001', facts=(fact,), events=(event,))
    resource = {
        'resourceType': 'Observation',
        'code': {'coding': [{'system': 'http://loinc.org', 'code': '68961-2'}]},
        'category': [{'coding': [{'system': 'http://terminology.hl7.org/CodeSystem/v2-0074', 'code': 'LAB'}]}],
        'subject': {'reference': 'urn:uuid:patient'},
    }
    spec = load_spec(repository_root / 'data/specs/native.json')
    _profile_fields(resource, event, fact, spec, 'hmo', 'urn:uuid:practitioner',
                    'urn:uuid:organization', pathway, {}, {})
    assert any(coding == {'system': 'http://snomed.info/sct', 'code': '108252007',
                          'display': 'Laboratory procedure'}
               for category in resource['category'] for coding in category['coding'])
    assert 'basedOn' not in resource
    assert 'specimen' not in resource
    profile = expected_profile(resource, 'hmo')
    assert profile.endswith('/il-core-observation-lab')
    role = observable_role(resource, 'hmo')
    assert role == 'laboratory-result'
    security = meta(spec, 'hmo', profile, role, confidentiality='R', sensitivity=('HIV',))['security']
    assert {'system': spec['metadata']['bucketSystem'], 'code': 'laboratoryTests'} in security
    assert {'system': spec['metadata']['confidentialitySystem'], 'code': 'R'} in security
    assert {'system': spec['metadata']['clinicalSensitivitySystem'], 'code': 'HIV'} in security


def test_received_summary_and_metadata_capture_preserve_origin_and_content(repository_root, monkeypatch):
    from synthetic_population.capture import CaptureDisposition, observable_origin
    from synthetic_population.care_pathway import ClinicalFact, ClinicalEvent, PatientPathway
    from synthetic_population.fhir_projection import project_pathway
    from synthetic_population.temporal import ClinicalInterval
    spec = load_spec(repository_root / 'data/specs/native.json')
    instant = datetime(2025, 1, 2, tzinfo=timezone.utc)
    fact = ClinicalFact('potassium', 'potassium', 'http://loinc.org', '6298-4',
                        'Potassium [Moles/volume] in Blood', 'final', 4.5, 'mmol/L', 3.5, 5.1, 'N')
    event = ClinicalEvent('lab-event', 'lab-episode', fact.id, 'result', 'laboratory',
                          'tertiary-ambulatory', 'result', 'final', ClinicalInterval(instant, instant, 'instant'))
    pathway = PatientPathway('path', 'cohort-000001', 'chronic-care', 'test', 'none', 'none', (fact,), (event,), ())
    captured = CaptureDisposition(event.id, 'hmo', 'received', 30, True, 'received result', 'hospital-a')
    resources, _ = project_pathway(spec, {'id': 'hmo'}, 1, 'urn:uuid:patient', pathway, [captured])
    received = resources[0][1]
    assert observable_origin(received, 'hmo') == 'hospital-a'
    assert received['valueQuantity']['value'] == 4.5
    assert 'reference' not in received['performer'][0]
    assert received['meta']['source'].endswith(':hmo')
    # The shipped policy excludes metadata-only. Exercise its projection with
    # an isolated policy and matching CodeSystem, without changing that policy.
    from copy import deepcopy
    from synthetic_population import capture_class
    original_load = capture_class.load_catalog
    def fixture_catalog(path):
        catalog = deepcopy(original_load(path))
        if path == spec['remediation']['capturePolicyCatalog']:
            catalog['sources']['hmo']['enabledDispositions'].append('metadata-only')
        elif path == capture_class.CODE_SYSTEM_PATH:
            catalog['concept'].append({'code': 'hmo-metadata-only'})
        return catalog
    with monkeypatch.context() as scoped:
        scoped.setattr(capture_class, 'load_catalog', fixture_catalog)
        metadata, _ = project_pathway(spec, {'id': 'hmo'}, 1, 'urn:uuid:patient', pathway,
                                      [replace(captured, disposition='metadata-only')])
    withheld = metadata[0][1]
    assert withheld['resourceType'] == 'DocumentReference'
    assert not any(key.startswith('value') or key == 'component' for key in withheld)
    assert '6298-4' not in json.dumps(withheld)
    assert not any(row['attachment'].get('data') for row in withheld['content'])
    assert observable_origin(withheld, 'hmo') == 'hospital-a'
    from synthetic_population.capture import metadata_content_leaks
    assert not metadata_content_leaks(withheld)
    withheld['content'][0]['attachment']['data'] = 'NC41'
    assert metadata_content_leaks(withheld)
    encounter = replace(event, role='emergency-encounter', care_setting='tertiary-emergency')
    pathway = replace(pathway, events=(encounter,))
    summary, _ = project_pathway(spec, {'id': 'hmo'}, 1, 'urn:uuid:patient', pathway,
                                 [replace(captured, disposition='summarized')])
    assert observable_origin(summary[0][1], 'hmo') == 'hospital-a'
    assert 'participant' not in summary[0][1]
    assert summary[0][1]['meta']['profile'][0].endswith('/il-hdp-encounter-supplemental')


def test_capture_utilization_and_wrong_owner_regression(repository_root):
    from synthetic_population.config import scale_spec
    from synthetic_population.bundle import build_population
    from synthetic_population.metrics import build_quality_metrics
    from synthetic_population.latent_audit import build_latent_mapping_audit
    spec = scale_spec(load_spec(repository_root / 'data/specs/native.json'), 100, '4.0.3-capture-test')
    built = build_population(spec, repository_root)
    metrics = build_quality_metrics(built['bundles'], built['crosswalk'], spec)['sourceDifferentiation']
    assert metrics['hmoOnlyPatientCount'] >= 20
    assert metrics['exactlyOneHospitalPatientCount'] >= 10
    assert all(count >= 20 for count in metrics['patientsWithoutClinicalByHospital'].values())
    assert all(metrics['performedCarePatientCounts'][source] <= metrics['heldClinicalDataPatientCounts'][source]
               for source in metrics['performedCarePatientCounts'])
    assert any(row.get('handoffs') for row in built['latentTruthLedger'])
    audit = build_latent_mapping_audit(built['bundles'], built['crosswalk'], built['latentTruthLedger'], spec)
    assert audit['status'] == 'pass'
    ordinary = {row['cohortId'] for row in built['crosswalk']['patients'] if row['ordinaryPathwayEligible']}
    mapping = next(row for row in built['latentTruthLedger'] if row['rowType'] == 'fhir-mapping'
                   and row['cohortId'] in ordinary and row['source'] == 'hmo'
                   and row['originSource'] == 'hospital-a' and row['resourceType'] == 'Observation')
    resource = next(row['resource'] for row in built['bundles']['hmo']['entry'] if row['fullUrl'] == mapping['resourceFullUrl'])
    resource['performer'] = [{'reference': full_url(spec['datasetVersion'], 'hmo', 'master', 'Practitioner')}]
    broken = build_latent_mapping_audit(built['bundles'], built['crosswalk'], built['latentTruthLedger'], spec)
    assert broken['status'] == 'fail'
    assert any('origin' in failure for failure in broken['failures'])


@pytest.mark.parametrize('variant', ['reconciliation-stable', 'temporary-stop', 'dose-change', 'polypharmacy-review', 'dispense-follow-up'])
def test_medication_ingredients_and_exposure_lifecycle(repository_root, variant):
    from synthetic_population.fhir_projection import project_pathway
    spec = load_spec(repository_root / 'data/specs/native.json')
    pathway = build_patient_pathway(spec, 12, 'medication-intensive', variant, '1980-01-01',
        spec['remediation']['pathwayTemplateCatalog'], spec['remediation']['clinicalContentCatalog'])
    captures, _ = project_capture(pathway, spec['remediation']['capturePolicyCatalog'], spec['seed'])
    resources, _ = project_pathway(spec, {'id': 'hmo'}, 12, 'urn:uuid:patient', pathway, captures['hmo'])
    meds = [r for _, r in resources if r['resourceType'].startswith('Medication')]
    assert meds
    assert all(r['medicationCodeableConcept']['coding'][0]['system'].endswith('/rxnorm') for r in meds)
    facts = {f.id: f for f in pathway.facts}
    events = [e for e in pathway.events if e.role in {'prescription', 'medication-review'} and facts[e.fact_id].code == '29046']
    if variant == 'temporary-stop':
        first, stop, restart = events[:3]
        assert facts[first.fact_id].status == facts[stop.fact_id].status == 'stopped'
        assert facts[first.fact_id].medication_detail['exposureEnd'] < facts[restart.fact_id].medication_detail['exposureStart']
        assert restart.links[0].relationship == 'supersedes'
        assert restart.links[0].target_event_id == stop.id
        assert facts[stop.fact_id].medication_detail['reason']
    if variant == 'dose-change':
        assert facts[events[0].fact_id].medication_detail['doseMg'] != facts[events[1].fact_id].medication_detail['doseMg']
        assert facts[events[0].fact_id].status == 'completed'


def test_perioperative_medication_administration_capture_tag_and_security(repository_root):
    from synthetic_population.fhir_projection import project_pathway
    spec = load_spec(repository_root / 'data/specs/native.json')
    pathway = build_patient_pathway(
        spec, 906, 'perioperative-surgical', 'procedure-with-medication',
        '1942-10-11', spec['remediation']['pathwayTemplateCatalog'],
        spec['remediation']['clinicalContentCatalog'],
    )
    captures, _ = project_capture(
        pathway, spec['remediation']['capturePolicyCatalog'], spec['seed'],
    )
    resources, _ = project_pathway(
        spec, {'id': 'hospital-a'}, 906, 'urn:uuid:patient', pathway,
        captures['hospital-a'],
    )
    administrations = [resource for _, resource in resources
                       if resource['resourceType'] == 'MedicationAdministration']
    assert {r['medicationCodeableConcept']['coding'][0]['code'] for r in administrations} == {'161','2180'}
    for administration in administrations:
        assert 'category' not in administration
        assert {
            'system': 'urn:hdp-test-bundle:synthetic:capture-class',
            'code': 'hospital-a-performed',
        } in administration['meta']['tag']
        assert ('http://fhir.health.gov.il/cs/il-hdp-information-buckets', 'medications') in {
            (label['system'], label['code']) for label in administration['meta']['security']
        }


def test_occurrence_age_and_final_interpretation():
    from synthetic_population.clinical_data import quantitative_result
    kwargs = dict(spec={'seed': 42, 'timelineAnchor': '2026-09-01'}, index=1, organization='hmo',
                  archetype='generally-healthy', loinc='2160-0', reference_low=.6, reference_high=1.2,
                  ordinary_min=.6, ordinary_max=1.2, birth_date='1961-08-01', gender='male')
    before = quantitative_result(**kwargs, occurrence_date='2026-07-31')
    after = quantitative_result(**kwargs, occurrence_date='2026-08-01')
    assert 'older-adult-creatinine' not in before['effects']
    assert 'older-adult-creatinine' in after['effects']
    assert before['interpretation'] == after['interpretation'] == 'N'


def test_infant_bmi_uses_final_same_date_pair_and_mutation_fails(repository_root):
    from synthetic_population.bundle import build_population
    from synthetic_population.clinical_depth import build_clinical_depth_audit
    from synthetic_population.config import scale_spec
    from synthetic_population.observable import codings

    spec = scale_spec(
        load_spec(repository_root / 'data/specs/native.json'),
        100, '4.1.6-s12a-bmi-test',
    )
    built = build_population(spec, repository_root)
    target = {}
    for entry in built['bundles']['hmo']['entry']:
        resource = entry['resource']
        cohort = resource.get('subject', {}).get('identifier', {}).get('value')
        if resource.get('resourceType') != 'Observation' or cohort != 'cohort-000056':
            continue
        code = next((code for system, code in codings(resource.get('code')) if system == 'http://loinc.org'), None)
        if code in {'29463-7', '8302-2', '39156-5'}:
            target[code] = resource

    assert {code: row['valueQuantity']['value'] for code, row in target.items()} == {
        '29463-7': 3.62, '8302-2': 50.48, '39156-5': 14.21,
    }
    assert 'referenceRange' not in target['39156-5']
    assert 'interpretation' not in target['39156-5']
    depth = build_clinical_depth_audit(built['bundles'], built['crosswalk'], spec)
    assert depth['pediatric']['bmiRecomputationFailureCount'] == 0
    assert depth['pediatric']['bmiAgeApplicability']['pediatric-derived-only'] >= 1

    mutated = copy.deepcopy(built['bundles'])
    for entry in mutated['hmo']['entry']:
        resource = entry['resource']
        cohort = resource.get('subject', {}).get('identifier', {}).get('value')
        code = next((value for system, value in codings(resource.get('code')) if system == 'http://loinc.org'), None)
        if resource.get('resourceType') == 'Observation' and cohort == 'cohort-000056' and code == '39156-5':
            resource['valueQuantity']['value'] = 20.6
    broken = build_clinical_depth_audit(mutated, built['crosswalk'], spec)
    assert broken['pediatric']['bmiRecomputationFailureCount'] == 1
    assert 'pediatric BMI lacks a same-date recomputable height/weight pair' in broken['failures']

    missing_pair = copy.deepcopy(built['bundles'])
    missing_pair['hmo']['entry'] = [
        entry for entry in missing_pair['hmo']['entry']
        if not (
            entry['resource'].get('resourceType') == 'Observation'
            and entry['resource'].get('subject', {}).get('identifier', {}).get('value') == 'cohort-000056'
            and next((value for system, value in codings(entry['resource'].get('code')) if system == 'http://loinc.org'), None) == '8302-2'
            and entry['resource'].get('effectiveDateTime', '').startswith('2025-11-22')
        )
    ]
    missing = build_clinical_depth_audit(missing_pair, built['crosswalk'], spec)
    assert missing['pediatric']['bmiRecomputationFailureCount'] == 1

    non_pediatric_crosswalk = copy.deepcopy(built['crosswalk'])
    next(row for row in non_pediatric_crosswalk['patients'] if row['cohortId'] == 'cohort-000056')['primaryArchetype'] = 'generally-healthy'
    non_pediatric_broken = build_clinical_depth_audit(mutated, non_pediatric_crosswalk, spec)
    assert non_pediatric_broken['pediatric']['bmiRecomputationFailureCount'] == 1


def test_bmi_age_policy_boundary_and_fail_closed_metadata(repository_root):
    from datetime import date
    from synthetic_population.bundle import build_population
    from synthetic_population.clinical_catalog import age_applicability, completed_age_years
    from synthetic_population.config import scale_spec
    from synthetic_population.metrics import build_quality_metrics, metric_failures
    from synthetic_population.observable import codings
    from synthetic_population.utils import read_json

    spec = scale_spec(load_spec(repository_root / 'data/specs/native.json'), 100, '4.3.0-age-boundary-test')
    content = read_json(repository_root / spec['remediation']['clinicalContentCatalog'])
    birth = date(2006, 9, 22)
    assert completed_age_years(birth, date(2026, 9, 21)) == 19
    assert completed_age_years(birth, date(2026, 9, 22)) == 20
    assert age_applicability(content, '39156-5', 19)['id'] == 'pediatric-derived-only'
    assert age_applicability(content, '39156-5', 20)['id'] == 'adult-reference-range'

    built = build_population(spec, repository_root)
    mutated = copy.deepcopy(built['bundles'])
    adult_bmi = None
    for bundle in mutated.values():
        for entry in bundle['entry']:
            resource = entry['resource']
            code = next((value for system, value in codings(resource.get('code')) if system == 'http://loinc.org'), None)
            if resource.get('resourceType') == 'Observation' and code == '39156-5' and resource.get('referenceRange'):
                adult_bmi = resource
                break
        if adult_bmi:
            break
    assert adult_bmi is not None
    adult_bmi['valueQuantity']['value'] = 4.0
    assert 'clinical distribution 39156-5 exceeds reviewed ordinary bounds' in metric_failures(
        build_quality_metrics(mutated, built['crosswalk'], spec), spec,
    )

    missing_occurrence = copy.deepcopy(built['bundles'])
    removed = False
    for bundle in missing_occurrence.values():
        for entry in bundle['entry']:
            resource = entry['resource']
            code = next((value for system, value in codings(resource.get('code')) if system == 'http://loinc.org'), None)
            if resource.get('resourceType') == 'Observation' and code == '39156-5':
                resource.pop('effectiveDateTime', None)
                removed = True
                break
        if removed:
            break
    assert removed
    failures = metric_failures(build_quality_metrics(missing_occurrence, built['crosswalk'], spec), spec)
    assert 'clinical distribution 39156-5 lacks ordinary event timing' in failures
    assert 'clinical distribution 39156-5 has unresolved age applicability' in failures

    missing_birth = copy.deepcopy(built['bundles'])
    removed = False
    for entry in missing_birth['hmo']['entry']:
        resource = entry['resource']
        cohort = next((item.get('value') for item in resource.get('identifier', [])
                       if item.get('system') == spec['identity']['cohortIdentifierSystem']), None)
        if resource.get('resourceType') == 'Patient' and cohort == 'cohort-000056':
            resource.pop('birthDate', None)
            removed = True
            break
    assert removed
    missing_birth_metrics = build_quality_metrics(missing_birth, built['crosswalk'], spec)
    assert missing_birth_metrics['clinicalDistributions']['39156-5']['unknownAgeCount'] >= 1
    assert 'clinical distribution 39156-5 has unresolved age applicability' in metric_failures(
        missing_birth_metrics, spec,
    )


def test_bp_pair_and_quantity_mutation_regression(repository_root):
    from synthetic_population.config import scale_spec
    from synthetic_population.bundle import build_population
    from synthetic_population.metrics import build_quality_metrics, metric_failures
    from synthetic_population.latent_audit import build_latent_mapping_audit
    spec = scale_spec(load_spec(repository_root / 'data/specs/native.json'), 100, '4.0.4-quantity-test')
    built = build_population(spec, repository_root)
    assert metric_failures(build_quality_metrics(built['bundles'], built['crosswalk'], spec), spec) == []
    pairs = [e['resource']['component'] for b in built['bundles'].values() for e in b['entry']
             if e['resource']['resourceType'] == 'Observation' and e['resource'].get('code', {}).get('coding', [{}])[0].get('code') == '85354-9']
    assert pairs and all(pair[1]['valueQuantity']['value'] <= 98 for pair in pairs)
    assert len({round(pair[0]['valueQuantity']['value'] - pair[1]['valueQuantity']['value'], 2) for pair in pairs}) > 1
    # The transaction entry order is not clinical truth. Mutate the whole
    # measured set so the audit check is independent of Bundle entry order.
    for pair in pairs:
        pair[1]['valueQuantity']['value'] = 105
    assert build_latent_mapping_audit(built['bundles'], built['crosswalk'], built['latentTruthLedger'], spec)['status'] == 'fail'


@pytest.mark.parametrize('kind,variant', [
    (kind, variant) for kind, variants in __import__('synthetic_population.problems', fromlist=['PROBLEM_VARIANTS']).PROBLEM_VARIANTS.items()
    if kind in {'timestamp-order', 'unit-mismatch', 'conflicting-diagnosis', 'discordant-lab'} for variant in variants
])
def test_quality_variants_have_observable_positive_and_clean_baseline(repository_root, kind, variant):
    from synthetic_population.detectors import quality_variant_evidence
    from synthetic_population.blind import blind_resource
    spec = load_spec(repository_root / 'data/specs/native.json')
    problem = {'problemId': 'problem-000003', 'class': 'data-quality', 'kind': kind, 'variant': variant, 'scenario': None}
    index = next(i for i in range(1, 1001) if primary_archetype(spec, i) == 'chronic-care' and care_pathway(spec, i, 'chronic-care')['pathwayVariant'] == 'diabetes-stable') if variant == 'code-family-conflict' else 1
    kwargs = dict(spec=spec, organization={'id': 'hmo'}, index=index, patient_url='urn:uuid:patient', birth_date='1980-01-01', full_capture=True)
    positive = _clinical_resources(**kwargs, problems=[problem])
    clean = _clinical_resources(**kwargs, problems=[])
    def observable(entries):
        return [{**{k: v for k, v in blind_resource(r, spec).items() if k not in {'text', 'note'}}, '_fullUrl': url} for url, r in entries] + [{'resourceType': 'Patient', 'birthDate': '1980-01-01'}]
    assert variant in quality_variant_evidence(observable(positive), kind, anchor=spec['timelineAnchor'])
    assert variant not in quality_variant_evidence(observable(clean), kind, anchor=spec['timelineAnchor'])
    assert all('reported scale differs' not in json.dumps(r) for _, r in positive)
    near = observable(copy.deepcopy(positive))
    if kind == 'timestamp-order':
        for row in near:
            if row.get('resourceType') == 'Observation':
                row['effectiveDateTime'] = '2025-01-01T12:00:00Z'
                row['issued'] = '2025-01-01T13:00:00Z'
    elif kind == 'conflicting-diagnosis':
        conditions = [row for row in near if row.get('resourceType') == 'Condition']
        for row in conditions:
            row['code'] = copy.deepcopy(conditions[0]['code'])
            row['clinicalStatus'] = copy.deepcopy(conditions[0]['clinicalStatus'])
            row['onsetDateTime'] = conditions[0]['onsetDateTime']
    elif kind in {'unit-mismatch', 'discordant-lab'}:
        baseline = next(row for _, row in clean if row['resourceType'] == 'Observation')
        for row in near:
            if row.get('resourceType') == 'Observation':
                row['valueQuantity'] = copy.deepcopy(baseline['valueQuantity'])
    assert variant not in quality_variant_evidence(near, kind, anchor=spec['timelineAnchor'])


def test_blind_identity_contact_variation_and_withheld_source(repository_root):
    from synthetic_population.blind import detect_corpus
    from synthetic_population.problems import SCENARIO_VARIANTS
    spec = load_spec(repository_root / 'data/specs/native.json')
    index, problems = next((i, rows) for i, rows in allocation_table(spec).items() if rows and rows[0].get('scenario') == 'allergy-treatment-risk')
    problem = problems[0]
    bundles = {}
    for position, organization in enumerate(spec['organizations']):
        source = organization['id']
        url = full_url(spec['datasetVersion'], source, cohort_id(index), 'Patient')
        patient = {'resourceType': 'Patient', 'name': [{'family': 'Example', 'given': ['Alice']}], 'birthDate': '1980-01-01', 'gender': 'female'}
        if position != 1:
            patient['telecom'] = [{'system': 'phone', 'value': '050-000-000' + str(position)}]
        clinical = _clinical_resources(spec, organization, index, url, problems, '1980-01-01', full_capture=True)
        _scenario_variant_resources(spec, source, index, url, problem, clinical)
        bundles[source] = {'entry': [{'fullUrl': url, 'resource': patient}] + [{'fullUrl': u, 'resource': r} for u, r in clinical]}
    def strict(value):
        return [row for row in detect_corpus(value, spec) if row.get('conclusionLevel') == 'strict-variant' and row.get('scenario') == 'allergy-treatment-risk']
    assert any(row['variant'] == problem['variant'] for row in strict(bundles))
    renamed = copy.deepcopy(bundles)
    replacements = {entry['fullUrl']: 'urn:uuid:renamed-' + str(position) for position, entry in enumerate(entry for bundle in renamed.values() for entry in bundle['entry'])}
    def rewrite(value):
        if isinstance(value, dict):
            return {key: rewrite(child) for key, child in value.items()}
        if isinstance(value, list):
            return [rewrite(child) for child in value]
        return replacements.get(value, value) if isinstance(value, str) else value
    renamed = {f'source-{position}': rewrite(bundle) for position, bundle in enumerate(renamed.values())}
    for bundle in renamed.values():
        bundle['entry'].reverse()
    assert {row['variant'] for row in strict(renamed)} == {row['variant'] for row in strict(bundles)}
    withheld = {source: bundle for source, bundle in bundles.items() if source != 'hospital-a'}
    assert not strict(withheld)
    false_join = copy.deepcopy(bundles)
    false_join['hospital-b']['entry'][0]['resource']['name'][0]['given'] = ['Bob']
    false_join['hospital-b']['entry'][0]['resource']['telecom'] = copy.deepcopy(bundles['hmo']['entry'][0]['resource']['telecom'])
    assert not strict(false_join)


def test_safety_positive_and_control_conclusions_are_explicit(repository_root):
    from synthetic_population.bundle import _apply_matched_control
    from synthetic_population.problems import SCENARIO_VARIANTS, additional_fixture_conclusions
    spec = load_spec(repository_root / 'data/specs/native.json')
    seen = set()
    for index, problems in allocation_table(spec).items():
        if not problems or not problems[0].get('scenario'):
            continue
        problem = problems[0]
        name, variant = problem['scenario'], problem['variant']
        if (name, variant) in seen:
            continue
        seen.add((name, variant))
        for control in (False, True):
            resources = []
            for organization in spec['organizations']:
                source = organization['id']
                patient_url = full_url(spec['datasetVersion'], source, cohort_id(index), 'Patient')
                clinical = _clinical_resources(spec, organization, index, patient_url, [] if control else problems, '1980-01-01', full_capture=True)
                if control:
                    _apply_matched_control(spec, index, name, source, patient_url, clinical, SCENARIO_VARIANTS[name].index(variant))
                _scenario_variant_resources(spec, source, index, patient_url, problem, clinical, control=control)
                resources.extend({**r, '_fullUrl': u, '_source': source} for u, r in clinical)
            observed = {v for v in SCENARIO_VARIANTS[name] if evaluate_variant(name, v, resources, require_three_source_context=True)}
            expected = set(additional_fixture_conclusions(variant, control=control)) | (set() if control else {variant})
            assert observed == expected, (name, variant, control, observed, expected)


def test_clinical_graph_signature_ignores_capture_labels_and_tracks_directed_edges():
    from synthetic_population.variability_v4 import _metrics
    def coded(kind, key, code, **fields):
        return {'resourceType': kind, '_fullUrl': key, '_source': 'source', 'status': 'completed' if kind == 'ServiceRequest' else 'final',
                'code': {'coding': [{'system': 'http://loinc.org', 'code': code}]}, **fields}
    graph = [
        coded('ServiceRequest', 'ServiceRequest/a', '6298-4'),
        coded('ServiceRequest', 'ServiceRequest/b', '718-7'),
        coded('Specimen', 'Specimen/a', '6298-4', request=[{'reference': 'ServiceRequest/a'}]),
        coded('Specimen', 'Specimen/b', '718-7', request=[{'reference': 'ServiceRequest/b'}]),
        coded('Observation', 'Observation/result', '6298-4', valueQuantity={'value':4.2,'system':'http://unitsofmeasure.org','code':'mmol/L'}, basedOn=[{'reference': 'ServiceRequest/a'}], specimen={'reference': 'Specimen/a'}),
    ]
    relabeled = copy.deepcopy(graph)
    for position, resource in enumerate(relabeled):
        resource.setdefault('meta', {}).setdefault('tag', []).append(
            {'system': 'urn:hdp-test-bundle:synthetic:capture-class', 'code': f'source-{position}-received'})
    rewired = copy.deepcopy(graph)
    rewired[-1]['basedOn'][0]['reference'] = 'ServiceRequest/b'
    rewired[-1]['specimen']['reference'] = 'Specimen/b'
    renamed = _cosmetic_mutation(graph)
    renamed.reverse()
    for view in ('role-topology', 'semantic'):
        original = normalized_signature(graph, view)
        assert normalized_signature(relabeled, view) == original
        assert normalized_signature(renamed, view) == original
        assert normalized_signature(rewired, view)[0] != original[0]
        clone_rows = [{'signature': normalized_signature(rows, view)[0], 'tokens': normalized_signature(rows, view)[1]} for rows in [graph, relabeled, renamed] * 4]
        assert _metrics(clone_rows)['largestExactCloneShare'] == 1
        mutated_rows = clone_rows + [{'signature': normalized_signature(rewired, view)[0], 'tokens': normalized_signature(rewired, view)[1]}]
        for threshold in (0, .15, .7):
            pairs, largest = _exhaustive_near_clusters(mutated_rows, threshold)
            optimized = _metrics(mutated_rows, threshold)
            assert (optimized['nearClonePairCount'], optimized['largestNearCloneClusterCount']) == (pairs, largest)


@pytest.fixture()
def s02_mapping_population(repository_root):
    from synthetic_population.bundle import build_population
    from synthetic_population.config import scale_spec
    spec = scale_spec(load_spec(repository_root / 'data/specs/native.json'), 30, '4.0.0-s02')
    return spec, build_population(spec, repository_root)


@pytest.mark.parametrize('mutation', ['none', 'duplicate', 'missing', 'reference', 'missing-target'])
def test_mapping_index_matches_exhaustive_findings(s02_mapping_population, monkeypatch, mutation):
    from synthetic_population import latent_audit
    spec, built = s02_mapping_population
    ledger = built['latentTruthLedger']
    mappings = [row for row in ledger if row['rowType'] == 'fhir-mapping']
    ordinary = {row['cohortId'] for row in built['crosswalk']['patients'] if row['ordinaryPathwayEligible']}
    mapping = next(row for row in mappings if row['cohortId'] in ordinary and row['resourceType'] == 'ServiceRequest')
    if mutation == 'duplicate':
        ledger.append(copy.deepcopy(mapping))
    elif mutation == 'missing':
        ledger.remove(mapping)
    elif mutation == 'missing-target':
        built['bundles'][mapping['source']]['entry'] = [row for row in built['bundles'][mapping['source']]['entry'] if row['fullUrl'] != mapping['resourceFullUrl']]
    elif mutation == 'reference':
        # Redirect actual order references, preserving the declared event link.
        def redirect(value):
            if isinstance(value, dict):
                if value.get('reference') == mapping['resourceFullUrl']:
                    value['reference'] = 'urn:uuid:changed-order'
                for child in value.values():
                    redirect(child)
            elif isinstance(value, list):
                for child in value:
                    redirect(child)
        redirect(built['bundles'][mapping['source']])
    before = copy.deepcopy(built)
    indexed = latent_audit.build_latent_mapping_audit(built['bundles'], built['crosswalk'], ledger, spec)

    class Exhaustive:
        def __init__(self, rows):
            self.rows = rows

        def get(self, key, default=()):
            return [row for row in self.rows if (row['eventId'], row['source']) == key] or default

    monkeypatch.setattr(latent_audit, '_index_event_source', Exhaustive)
    exhaustive = latent_audit.build_latent_mapping_audit(built['bundles'], built['crosswalk'], ledger, spec)
    assert indexed == exhaustive
    assert built == before
    if mutation != 'none':
        expected = {'duplicate': 'duplicate mapping', 'missing': 'mapping count',
                    'reference': 'target is not referenced', 'missing-target': 'mapping target is missing'}[mutation]
        assert indexed['status'] == 'fail'
        assert any(expected in row for row in indexed['failures'])


def test_mapping_lookup_work_doubles_for_disjoint_population(s02_mapping_population):
    from synthetic_population.latent_audit import build_latent_mapping_audit
    spec, built = s02_mapping_population
    audit = build_latent_mapping_audit(built['bundles'], built['crosswalk'], built['latentTruthLedger'], spec)
    # Duplicate the complete graph and ledger with disjoint identities, preserving
    # codes, dates and source names. No extra clinical semantics are introduced.
    original = json.dumps(built)
    identities = {row['eventId'] for row in built['latentTruthLedger'] if row.get('eventId')}
    identities |= {row['cohortId'] for row in built['crosswalk']['patients']}
    identities |= {row['fullUrl'] for bundle in built['bundles'].values() for row in bundle['entry']}
    for identity in sorted(identities, key=len, reverse=True):
        original = original.replace(identity, identity + '-copy')
    other = json.loads(original)
    for source in built['bundles']:
        built['bundles'][source]['entry'].extend(other['bundles'][source]['entry'])
    built['crosswalk']['patients'].extend(other['crosswalk']['patients'])
    built['latentTruthLedger'].extend(other['latentTruthLedger'])
    doubled = build_latent_mapping_audit(built['bundles'], built['crosswalk'], built['latentTruthLedger'], spec)
    assert doubled['work'] == {key: 2 * value for key, value in audit['work'].items()}


@pytest.mark.parametrize('cached', [False, True])
def test_graph_consumers_isolate_nested_mutations(cached):
    from synthetic_population.observable import patient_graphs, with_graph_index
    spec = {'identity': {'cohortIdentifierSystem': 'urn:test:cohort'}}
    bundles = {'hmo': {'entry': [
        {'fullUrl': 'Patient/p', 'resource': {'resourceType': 'Patient', 'identifier': [{'system': 'urn:test:cohort', 'value': 'p'}], 'name': [{'given': ['Rina']}] }},
        {'fullUrl': 'Observation/o', 'resource': {'resourceType': 'Observation', 'subject': {'reference': 'Patient/p'}, 'code': {'coding': [{'system': 'urn:test', 'code': 'a'}]}}},
    ]}}
    original = copy.deepcopy(bundles)

    def consumers():
        expected = copy.deepcopy(patient_graphs(bundles, spec))
        graphs, patients = patient_graphs(bundles, spec)
        graphs['p'][1]['code']['coding'][0]['code'] = 'contaminated'
        patients['p']['name'][0]['given'].append('contaminated')
        graphs['p'].clear()
        assert patient_graphs(bundles, spec) == expected
        assert bundles == original

    (with_graph_index(consumers) if cached else consumers)()


@pytest.mark.parametrize('fail', [False, True])
def test_graph_cache_invocation_lifetime_and_nested_restore(monkeypatch, fail):
    from synthetic_population import observable
    spec = {'identity': {'cohortIdentifierSystem': 'urn:test'}}
    bundles = {'hmo': {'entry': []}}
    build = observable._build_patient_graphs
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs['include_demographics'])
        return build(*args, **kwargs)

    monkeypatch.setattr(observable, '_build_patient_graphs', counted)

    @observable.with_graph_index
    def nested():
        observable.patient_graphs(bundles, spec)
        if fail:
            raise RuntimeError('audit failed')

    @observable.with_graph_index
    def invocation():
        outer = observable._GRAPH_INDEX.get()
        observable.patient_graphs(bundles, spec)
        observable.patient_graphs(bundles, spec)
        observable.patient_graphs(bundles, spec, include_demographics=False)
        try:
            nested()
        finally:
            assert observable._GRAPH_INDEX.get() is outer
            observable.patient_graphs(bundles, spec)

    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError, match='audit failed'):
                invocation()
        else:
            invocation()
        assert observable._GRAPH_INDEX.get() is None
    assert calls == [True, False, True] * 2


def _exhaustive_near_clusters(rows, threshold):
    from itertools import combinations
    from synthetic_population.variability_v4 import _distance
    neighbors = {i: {i} for i in range(len(rows))}
    pairs = 0
    for left, right in combinations(range(len(rows)), 2):
        if _distance(rows[left]['tokens'], rows[right]['tokens']) <= threshold:
            neighbors[left].add(right)
            neighbors[right].add(left)
            pairs += 1
    clusters = []
    unseen = set(neighbors)
    while unseen:
        pending = [unseen.pop()]
        members = set(pending)
        while pending:
            for other in neighbors[pending.pop()] - members:
                members.add(other)
                unseen.discard(other)
                pending.append(other)
        clusters.append(members)
    return pairs, max(map(len, clusters), default=0)


@pytest.mark.parametrize('threshold', [0, .15, .3, .7, 1])
def test_near_clone_metrics_match_exhaustive_multiset_oracle(threshold):
    from itertools import combinations
    from synthetic_population.variability_v4 import _distance, _metrics
    tokens = [[], [('a',)] * 3, [('a',)] * 10,
              [('a',)] * 9 + [('b',)], [('a',)] * 8 + [('b',)] * 2,
              [('z',)] * 10, [('z',)] * 30]
    rows = [{'signature': str(i), 'tokens': values} for i, values in enumerate(tokens)]
    rows.extend([copy.deepcopy(rows[2]), copy.deepcopy(rows[2])])
    metrics = _metrics(rows, threshold)
    pairs, largest = _exhaustive_near_clusters(rows, threshold)
    assert (metrics['nearClonePairCount'], metrics['largestNearCloneClusterCount']) == (pairs, largest)
    assert metrics['largestNearCloneClusterShare'] == round(largest / len(rows), 6)
    expected_pruned = 0
    for left, right in combinations(tokens, 2):
        largest_size = max(len(left), len(right))
        bound = 1 - min(len(left), len(right)) / largest_size if largest_size else 0
        if bound > threshold:
            expected_pruned += 1
            assert _distance(left, right) > threshold
        else:
            # The optimized union cardinality must equal the exhaustive multiset union.
            from collections import Counter
            shared = sum((Counter(left) & Counter(right)).values())
            total = len(left) + len(right) - shared
            assert (1 - shared / total if total else 0) == _distance(left, right)
    assert metrics['comparisonWork'] == {'distinctSignatures': len(tokens),
        'lengthBoundPrunedPairs': expected_pruned,
        'exactDistanceComparisons': len(tokens) * (len(tokens) - 1) // 2 - expected_pruned}
    assert _metrics([], threshold)['largestNearCloneClusterCount'] == 0


def test_cached_native_audits_equal_uncached_results(s02_mapping_population, monkeypatch):
    from synthetic_population import observable
    from synthetic_population.history import build_history_audit
    from synthetic_population.latent_audit import build_latent_mapping_audit
    from synthetic_population.variability_v4 import build_variability_audit
    spec, built = s02_mapping_population
    original = copy.deepcopy(built)
    calls = []
    build = observable._build_patient_graphs

    def counted(*args, **kwargs):
        calls.append(kwargs['include_demographics'])
        return build(*args, **kwargs)

    monkeypatch.setattr(observable, '_build_patient_graphs', counted)

    def audits():
        return [audit(built['bundles'], built['crosswalk'], *args) for audit, args in (
            (build_history_audit, (spec, built['latentTruthLedger'])),
            (build_latent_mapping_audit, (built['latentTruthLedger'], spec)),
            (build_variability_audit, (spec, built['latentTruthLedger'])),
        )]

    expected = audits()
    calls.clear()

    @observable.with_graph_index
    def cached():
        assert audits() == expected
        assert audits() == expected

    cached()
    assert sorted(calls) == [False, True]
    assert built == original
    assert observable._GRAPH_INDEX.get() is None
