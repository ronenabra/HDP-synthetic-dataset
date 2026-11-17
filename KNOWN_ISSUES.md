# Known issues

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
