# Synthetic dataset release: 4.2.2-2026-10-10

This release contains two compressed populations with three complete FHIR R4
transaction bundles each: HMO, Hospital A, and Hospital B. Expanded bundles are
not checked in. The [release index](release-index.json) lists every archive and
payload's size, SHA-256 checksum, and resource count.

| Archive | Distinct people | FHIR resources | Archive bytes | Expanded files, bytes |
| --- | ---: | ---: | ---: | ---: |
| `hdp-synthetic-1k.tar.zst` | 1,000 | 19,179 | 1,667,795 | 88,081,518 |
| `hdp-synthetic-10k.tar.zst` | 10,000 | 190,525 | 17,554,269 | 872,678,732 |

Each archive contains a population directory (`1k/` or `10k/`), its original
manifest and payload checksums, and the three transaction files under `bundles/`.
Source-specific records and intentional duplicates mean Patient-resource counts
exceed distinct-person counts. Archive bytes are preserved unchanged; the release
index is the portable inventory for this repository.

## Verify and extract

From the repository root, using GNU tar, zstd, and sha256sum:

```bash
release_dir=releases/synthetic-population/4.2.2-2026-10-10
(cd "$release_dir" && sha256sum -c archive-checksums.sha256)
mkdir -p generated/bundled-release
tar --zstd -xf "$release_dir/hdp-synthetic-1k.tar.zst" -C generated/bundled-release
(cd generated/bundled-release/1k && sha256sum -c checksums.sha256)
```

For 10k, substitute `hdp-synthetic-10k.tar.zst` and check the `10k` directory.
Extraction is optional. Keep expanded transaction files out of version control.

The receiving FHIR server must support the size of these transactions. Server
loading and retry behavior have not been verified by this project.

## Validation scope and limitations

Original 1k validation covered every resource and 34 validation envelopes, with
reviewed exceptions. Original 10k official validation sampled twelve graphs;
it was not exhaustive validation of every resource. The forty canonical culture
errors remain under a reviewed exception.

Offline terminology warnings, intentional data-quality problems, unknown pediatric
doses or administration amounts, allergy-associated prescribing risks, pediatric
blood-pressure interpretation limits, and dated dose-reference weights remain.
Automated clinical review is not external human clinical clearance. These are
synthetic software-testing fixtures, not clinical guidance or representative
patient statistics.

The standalone 1k reproduction matched all three source transaction files byte
for byte. A standalone 10k replay has not yet been executed. The generic test
suite has an open medication-exposure graph-audit finding; see
[known issues](../../../KNOWN_ISSUES.md).

For generation instructions and the distinction between native construction,
release repairs, and generic generation, see the [project README](../../../README.md).

## License and notices

Project-owned content is offered under [Apache License 2.0](../../../LICENSE).
Embedded terminology and reference content retain their own terms. Provide
[NOTICE](../../../NOTICE), [THIRD_PARTY_NOTICES.md](../../../THIRD_PARTY_NOTICES.md),
and the [LOINC attribution](../../../LOINC_short_license.txt) alongside shared
archives. The project uses selected terminology references rather than providing
full terminology databases.
