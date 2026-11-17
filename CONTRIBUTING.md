# Contributing

Thank you for helping improve the generator and its documentation. Start with
the [README](README.md) and [known issues](KNOWN_ISSUES.md).

## Development setup

Create a virtual environment, install the project with its test dependency, and
restore the pinned inputs as described in the README:

```bash
python -m pip install -e '.[dev]'
hdp-synthetic-rehydrate --check
python -m pytest -q
```

The suite currently has a documented failing graph-audit case. Report whether
your change adds failures or changes that finding; do not claim a clean baseline.
Use focused tests while developing and record the commands and results you ran.

## Propose a change

Open an issue or pull request through the repository host. Describe the problem, the resulting behavior, relevant tests, and any
limits to your verification. Small changes are easier to review.

- Keep patient records and examples synthetic. Do not submit real patient data,
  credentials, private endpoints, or confidential execution logs.
- Preserve source relationships, synthetic identifier rules, and confidentiality
  labels. Include a small relevant regression case for behavior changes.
- Keep native construction, accepted-release repairs, and generic experiments
  distinct. A generic improvement must not silently change an accepted release.
- Do not edit accepted archives or replace expected hashes just to accept new
  output. Changed datasets need a separate release identity and review.
- Retain failed-run evidence when investigating a failure. Do not add generated
  populations, caches, virtual environments, or expanded release bundles to Git.
- Explain new inputs and their canonical source, version, license, and attribution.
  Prefer restoring public reference packages over checking in full copies.

Use UTF-8, two-space indentation for JSON, and the surrounding Python style.
Keep documentation in clear English. Describe observed results separately from
estimates and assumptions.

## Contribution terms

Unless you explicitly state otherwise, original contributions intentionally
submitted for inclusion are offered under the project's Apache License 2.0,
consistent with section 5 of [LICENSE](LICENSE). You must have the right to
submit the material. Third-party material keeps its own license and notices;
identify it in your contribution.

Follow the [code of conduct](CODE_OF_CONDUCT.md). For sensitive reports, see
[SECURITY.md](SECURITY.md).
