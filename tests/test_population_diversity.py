"""S12B pilot applicability and fail-closed diversity evidence."""

import copy
import json

from synthetic_population.bundle import build_population
from synthetic_population.config import load_spec, scale_spec
from synthetic_population import variability_v4
from synthetic_population.observable import patient_graphs
from synthetic_population.manifest import generate, refresh_checksums
from synthetic_population.verify import verify
from synthetic_population.variability_v4 import (
    _classify_variant_evidence,
    _canonical_qualification_eligible,
    build_variability_audit,
)


def test_sufficiently_sampled_clone_collapse_remains_a_diversity_failure():
    assert _classify_variant_evidence(
        assigned_patient_count=3,
        archetype_eligible_patient_count=10,
        variant_eligible_patient_count=3,
        observed_signature_count=1,
        minimum_signatures=3,
    ) == "fails-signature-minimum"
    assert _classify_variant_evidence(
        assigned_patient_count=3,
        archetype_eligible_patient_count=10,
        variant_eligible_patient_count=3,
        observed_signature_count=3,
        minimum_signatures=3,
    ) == "meets-signature-minimum"


def test_insufficient_sample_is_explicit_and_does_not_become_a_pass():
    assert _classify_variant_evidence(
        assigned_patient_count=2,
        archetype_eligible_patient_count=10,
        variant_eligible_patient_count=2,
        observed_signature_count=2,
        minimum_signatures=3,
    ) == "insufficient-variant-sample"
    assert _classify_variant_evidence(
        assigned_patient_count=2,
        archetype_eligible_patient_count=9,
        variant_eligible_patient_count=2,
        observed_signature_count=1,
        minimum_signatures=3,
    ) == "insufficient-archetype-sample"


def test_canonical_claim_is_closed_when_a_native_diversity_mutation_fails():
    assert _canonical_qualification_eligible(
        canonical=True, population_evidence_status="pass", native_execution_status="fail"
    ) is False
    assert _canonical_qualification_eligible(
        canonical=True, population_evidence_status="fail", native_execution_status="pass"
    ) is False
    assert _canonical_qualification_eligible(
        canonical=True, population_evidence_status="pass", native_execution_status="pass"
    ) is True


def test_rare_boundary_and_absent_variants_keep_their_scope():
    assert _classify_variant_evidence(
        assigned_patient_count=1,
        archetype_eligible_patient_count=10,
        variant_eligible_patient_count=1,
        observed_signature_count=1,
        minimum_signatures=2,
    ) == "insufficient-variant-sample"
    assert _classify_variant_evidence(
        assigned_patient_count=2,
        archetype_eligible_patient_count=10,
        variant_eligible_patient_count=2,
        observed_signature_count=2,
        minimum_signatures=2,
    ) == "meets-signature-minimum"
    assert _classify_variant_evidence(
        assigned_patient_count=0,
        archetype_eligible_patient_count=10,
        variant_eligible_patient_count=0,
        observed_signature_count=0,
        minimum_signatures=3,
    ) == "absent"
    assert _classify_variant_evidence(
        assigned_patient_count=1,
        archetype_eligible_patient_count=10,
        variant_eligible_patient_count=0,
        observed_signature_count=0,
        minimum_signatures=3,
    ) == "no-eligible-ordinary-history"


def test_pilot_insufficiency_is_separate_from_canonical_claim(repository_root):
    source = load_spec(repository_root / "data/specs/native.json")
    spec = scale_spec(source, 30, "4.1.12-s12b-test")
    built = build_population(spec, repository_root)
    audit = build_variability_audit(
        built["bundles"], built["crosswalk"], spec, built["latentTruthLedger"]
    )

    assert audit["status"] == "pass"
    assert audit["nativeExecutionStatus"] == "pass"
    assert audit["populationEvidenceStatus"] == "insufficient-sample"
    assert audit["canonicalQualificationEligible"] is False
    role_rows = [row for row in audit["variantEvidence"] if row["gateApplicable"]]
    assert len(role_rows) == 46
    assert any(row["disposition"] == "absent" for row in role_rows)
    assert any(row["disposition"] == "insufficient-archetype-sample" for row in role_rows)


def test_sufficiently_sampled_variant_collapse_fails_the_real_audit(repository_root, monkeypatch):
    source = load_spec(repository_root / "data/specs/native.json")
    spec = scale_spec(source, 100, "4.1.12-s12b-collapse-test")
    built = build_population(spec, repository_root)
    original_graphs, _ = patient_graphs(built["bundles"], spec, include_demographics=False)
    targets = [
        row for row in built["crosswalk"]["patients"]
        if row["primaryArchetype"] == "chronic-care"
        and row["pathwayVariant"] == "diabetes-stable"
        and row.get("ordinaryPathwayEligible", True)
    ]
    assert len(targets) >= 3
    collapsed_graphs = copy.deepcopy(original_graphs)
    template = copy.deepcopy(original_graphs[targets[0]["cohortId"]])
    for row in targets:
        collapsed_graphs[row["cohortId"]] = copy.deepcopy(template)

    def collapsed_patient_graphs(_bundles, _spec, *, include_demographics=True):
        return copy.deepcopy(collapsed_graphs), {}

    monkeypatch.setattr(variability_v4, "patient_graphs", collapsed_patient_graphs)
    audit = build_variability_audit(
        built["bundles"], built["crosswalk"], spec, built["latentTruthLedger"]
    )
    row = next(
        item for item in audit["variantEvidence"]
        if item["gateApplicable"] and item["variant"] == "diabetes-stable"
    )
    assert row["variantEligiblePatientCount"] >= row["minimumSignatures"]
    assert row["observedSignatureCount"] == 1
    assert row["disposition"] == "fails-signature-minimum"
    assert audit["populationEvidenceStatus"] == "fail"
    assert audit["nativeExecutionStatus"] == "fail"


def test_sparse_pilot_relabelled_as_canonical_stays_fail_closed(repository_root):
    source = load_spec(repository_root / "data/specs/native.json")
    pilot_spec = scale_spec(source, 30, "4.1.12-s12b-canonical-sparse-test")
    built = build_population(pilot_spec, repository_root)
    canonical_spec = copy.deepcopy(pilot_spec)
    canonical_spec["populationSize"] = 1000
    audit = build_variability_audit(
        built["bundles"], built["crosswalk"], canonical_spec, built["latentTruthLedger"]
    )
    assert audit["populationEvidence"]["scope"] == "canonical"
    assert audit["populationEvidenceStatus"] == "fail"
    assert audit["status"] == "fail"
    assert audit["canonicalQualificationEligible"] is False
    assert any("canonical evidence denominator is insufficient" in item for item in audit["failures"])


def test_native_mutation_failure_closes_canonical_eligibility(repository_root, monkeypatch):
    source = load_spec(repository_root / "data/specs/native.json")
    spec = scale_spec(source, 100, "4.1.12-s12b-mutation-test")
    built = build_population(spec, repository_root)
    monkeypatch.setattr(
        variability_v4,
        "_mutation_evidence",
        lambda _graphs, _declarations, **_kwargs: {"status": "fail", "failures": ["collapsed-fixture"]},
    )
    audit = build_variability_audit(
        built["bundles"], built["crosswalk"], spec, built["latentTruthLedger"]
    )
    assert audit["nativeExecutionStatus"] == "fail"
    assert "mutation/collapsed-fixture" in audit["failures"]
    assert audit["canonicalQualificationEligible"] is False


def test_manifest_claim_tampering_is_rejected_after_checksum_refresh(repository_root, tmp_path):
    source = load_spec(repository_root / "data/specs/native.json")
    spec = scale_spec(source, 30, "4.1.12-s12b-manifest-test")
    run = tmp_path / "generated" / "synthetic-population.v4.1.12-s12b-manifest-test"
    generate(spec, run, repository_root)
    assert verify(run)["ok"]
    manifest_path = run / "dataset-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["remediationEvidence"]["antiTemplatePopulationEvidenceStatus"] = "pass"
    manifest["remediationEvidence"]["antiTemplateCanonicalQualificationEligible"] = True
    manifest["remediationEvidence"]["antiTemplatePopulationEvidence"]["status"] = "pass"
    manifest["remediationEvidence"]["antiTemplatePopulationEvidence"]["canonicalQualificationEligible"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    refresh_checksums(run)
    result = verify(run)
    assert result["ok"] is False
    assert "manifest remediation evidence differs from recomputed audits" in result["errors"]
