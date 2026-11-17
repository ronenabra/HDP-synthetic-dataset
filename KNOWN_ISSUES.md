# Known issues

## Patient identifiers differ across sources for the same person

**Status: open.** The packaged release `4.2.2-2026-10-10`, including both
the 1k and 10k populations, does not provide a shared sandbox business
identifier for non-anchor patients across its three sources.

### Effect on the PCM consent flow

The consent flow requires the same patient identifier **system and value**
at every participating source. A consent identifying a patient at Hospital A
must identify that same person at Hospital B and the HMO without substituting
a source-specific identifier.

Most released Patient records use the business identifier system
`http://fhir.health.gov.il/identifier/il-hdp-sandbox-synthetic-id`, but its
UUID value differs between sources for the same population member. These
identifiers are accepted by the Doctor app in non-Strict connectathon mode;
the problem is their lack of shared identity across sources. Selecting one
source's identifier cannot resolve the corresponding patient at the other
sources through an exact identifier lookup. The two standard test anchors
(`il-hdp-test-id` values `1` and `2`) have shared identifiers and are exceptions.

For example, the packaged 1k population represents `cohort-000003` as:

| Source | Sandbox business identifier value |
| --- | --- |
| HMO | `38fea765-d7e2-4c8a-9887-2815463c31be` |
| Hospital A | `f38a0045-c171-470f-829f-1dfe604e790c` |
| Hospital B | `4d99e5f7-abe2-4c46-82ae-ad39cb2d9460` |

The cohort identifier correlates these records within the dataset, but it is
not an accepted patient business identifier for the PCM consent flow. Internal
source-local strong identifiers and FHIR resource IDs do not solve this either.

### Cause and required correction

[`business_guid`](src/synthetic_population/business_identifiers.py) derives the
business UUID from the source and the source-local strong identifier, as well
as the seed, dataset version and resource type. Consequently, corresponding
Patient records receive different business UUIDs at different sources. The
current bundle validator also rejects repeated business GUIDs across sources;
generation and validation rules therefore need to change together.

A future corrected release must assign the same sandbox patient business
identifier system/value pair to records representing the same person across
all three sources. Preserve source-local strong identifiers and FHIR references
separately. Keep unrelated people distinct and explicitly handle intentional
duplicate-record scenarios so they do not silently introduce ambiguous patient
resolution. Add regression coverage for shared identity and the cross-source
consent flow before claiming that compatibility.

Do not rewrite the accepted archives or their published hashes. The correction
requires a new release identity and review. Until then, use anchors `1` and `2`
for shared-identity tests; other patient GUIDs support source-specific testing
only.

### Scope and verification

Offline inspection on 17 November 2026 verified all six transaction-bundle
hashes against the release index. In the 1k population, all 998 non-anchor cohort
members shared by HMO and either hospital had different sandbox business UUIDs.
The corresponding 10k comparison found different UUIDs for all 9,998 shared
non-anchor cohort members. Both comparisons used the cohort identifier to match
people across sources, rather than resource order or names.

This is a documented dataset integration limitation. No live HAPI import or
end-to-end consent test was performed for this investigation, and no generator
or released payload was changed.

## Generic graph audit: medication-exposure declarations

**Status: open.** The test remains enabled as a normal failing test.

Reproduce the case with:

```bash
python -m pytest -q 'tests/test_graph_closure.py::test_each_named_typed_successor_rejects_missing_decisive_evidence[oncology-monitoring-oncology-solid-tumor-surgical-follow-up-390906007-zero]'
```

The case builds a 1,000-person generic population without intentional problems.
Before testing its oncology mutation, it requires the whole baseline graph audit
to have zero findings. Five records instead declare `medication-exposure` while
the audit infers `none`:

| Cohort | Primary pathway |
| --- | --- |
| `cohort-000359` | Chronic care: medication adjustment, switch class |
| `cohort-000365` | Chronic care: medication adjustment, switch class |
| `cohort-000377` | Chronic care: medication adjustment, switch class |
| `cohort-000437` | Chronic care: medication adjustment, switch class |
| `cohort-000784` | Medication intensive: dispense follow-up, on-time refill |

The finding is `declared-observable-mismatch:medication-exposure:none`.

The audit recognizes active hypertension with a complete active lisinopril
prescription, or a completed amoxicillin statement. The generator's medication
lifecycle can end earlier therapy when switching ingredients. The declarations
and the audit's current-treatment criteria need to be reconciled; the exact
cause has not been established for every affected record.

### Scope and verification

This finding affects the generic generation/audit path. Its test population is
not the accepted-release configuration. The accepted 1k reproduction matched all
three released transaction files byte for byte; that comparison does not establish
clinical correctness. The standalone 10k reproduction has not yet been replayed.

The latest focused verification covered 113 cases: **112 passed and this case
failed**. A fully passing full-suite result is not claimed.
