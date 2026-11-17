# HDP Synthetic Dataset

A rule-based generator and ready-to-use synthetic datasets for testing systems
that exchange Israeli healthcare data using the HL7® FHIR® R4 standard and HDP
profiles. The project does not use Synthea or read real patient records.

Each population is represented across three fictional sources: an HMO, Hospital A,
and Hospital B. The data includes clinical histories, source-specific records,
synthetic identifiers, confidentiality labels, and deliberately introduced data
quality problems. It is intended for software development, interoperability
testing, demonstrations, and load testing.

The supported reproduction target is release **4.2.2-2026-10-10**. It is
synthetic test data, not a claim of government approval, complete conformance,
or clinical safety.

## Use the bundled release

The [release folder](releases/synthetic-population/4.2.2-2026-10-10/) contains
compressed archives, checksums, and a release index with payload identities and
validation scope. Expanded transaction bundles are not checked in. The archives are repository
assets; they are not included in the Python wheel.

| Archive | People | FHIR resources | Compressed | Expanded files |
| --- | ---: | ---: | ---: | ---: |
| `hdp-synthetic-1k.tar.zst` | 1,000 | 19,179 | 1.67 MB | about 88.1 MB |
| `hdp-synthetic-10k.tar.zst` | 10,000 | 190,525 | 17.55 MB | about 872.7 MB |

Sizes use decimal MB. People are distinct population members, not the number of
Patient resources: a person can have records at several sources and intentional
duplicate records. Each archive contains three complete transaction bundles.

To check the downloads and extract the 1k dataset into an ignored output folder:

```bash
release_dir=releases/synthetic-population/4.2.2-2026-10-10
(cd "$release_dir" && sha256sum -c archive-checksums.sha256)
mkdir -p generated/bundled-release
tar --zstd -xf "$release_dir/hdp-synthetic-1k.tar.zst" -C generated/bundled-release
(cd generated/bundled-release/1k && sha256sum -c checksums.sha256)
```

For 10k, extract `hdp-synthetic-10k.tar.zst` and check the `10k` directory instead.
These examples use Bash, GNU tar, `zstd`, and `sha256sum`. On systems without
`sha256sum`, use `shasum -a 256 -c` for the checksum commands.

Extracted payloads are in `1k/bundles/` or `10k/bundles/`. The project does not
upload them to a FHIR server. Server transaction limits, loading, and retry
behavior must be handled by the receiving system.

## Install the generator

Use Python **3.10 or later**. The generator has no third-party Python runtime
dependencies.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Generation inputs include pinned public FHIR packages. The restore command reads
the standard FHIR package cache at `~/.fhir/packages`; it does not download packages.
Populate missing cache entries from the public registry, or use an existing cache:

```bash
(
  set -euo pipefail
  for package_id in 'il.core.fhir.r4#0.21.0' 'il.core.fhir.r4#0.22.0' 'il.hdp.fhir.r4#0.6.0'; do
    package_name="${package_id%#*}"
    package_version="${package_id##*#}"
    cache_dir="$HOME/.fhir/packages/$package_id"
    if [ ! -f "$cache_dir/package/package.json" ]; then
      mkdir -p "$cache_dir"
      curl --fail --location "https://packages.fhir.org/$package_name/$package_version" \
        | tar -xz -C "$cache_dir"
    fi
  done
)
hdp-synthetic-rehydrate
hdp-synthetic-rehydrate --check
```

Only one city CodeSystem is restored from IL Core 0.21.0 for the pinned native
generator. IL Core 0.22.0 and HDP 0.6.0 supply the reference context used by the
broader generic generator and its tests. The small synthetic coverage seed is
included with the code; its current identifier form is restored locally.

Restored files go into the ignored `src/synthetic_population/reference/` folder.
Their content is checked against recorded hashes. Use
`hdp-synthetic-rehydrate --cache /path/to/packages` for another cache location;
use `--replace` only when intentionally replacing stale restored files.

Package and terminology terms still apply; see [third-party notices](THIRD_PARTY_NOTICES.md).

## Reproduce the accepted dataset

One command runs two explicit stages:

1. **`native_generator`** constructs a population from the frozen rules, inputs,
   seed, and release configuration.
2. **`release_repairs`** applies the identifier, laboratory, narrative, and clinical
   corrections used for the accepted release. It checks intermediate identities
   and requires exact published hashes for all three final transaction bundles.

The accepted archives are not used as generation inputs. The repair definitions
target the accepted 1k and 10k configurations; they are not general corrections
for arbitrary population sizes or seeds.

```bash
hdp-synthetic-population reproduce-release \
  --population-size 1000 \
  --out generated/synthetic-population.accepted-1k

hdp-synthetic-population verify --out generated/synthetic-population.accepted-1k
```

For 10k, use `--population-size 10000` and a different output directory, such as
`generated/synthetic-population.accepted-10k`.

Output directory names must start with `synthetic-population` and be directly
inside a directory named `generated`. Existing output is preserved; choose a new
name for another run. Failed runs retain a sibling `.partial-<pid>` directory.
Allow at least **10 GiB of free disk space plus the output size**. Construction
holds the population in memory, so the larger dataset also needs more RAM.

A successful run creates:

```text
generated/synthetic-population.accepted-1k/
  fhir-server-load/
    hmo.source.transaction.json
    hospital-a.source.transaction.json
    hospital-b.source.transaction.json
  dataset-manifest.json
  dataset-spec.json
  run-state.json
  checksums.sha256
```

The three transaction files must match the accepted release byte for byte.
New run metadata, timings, and compressed archive bytes are not reproduced. `verify` checks integrity against the pinned release; it does
not run an official FHIR validator or certify clinical correctness.

Reproduction status: the 1k command has been checked and all three transaction
files matched the bundled release exactly. The standalone 10k command has pinned
targets but **has not yet been replayed**. A successful 10k reproduction is not
claimed.

## Generic generation and tests

The top-level package also retains configurable generation, audits, and projection
commands. Use `hdp-synthetic-population --help` to list them. The `generate`
command uses a different, configurable path and does not apply the complete
accepted-release repair recipe. Its results should not be described as reproducing
the bundled release.

To install the test dependency and run the retained suite:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

**The test suite is not fully passing.** One known generic graph-audit case reports
five medication-exposure declarations without the required observable evidence.
The test remains enabled and failing. See [known issues](KNOWN_ISSUES.md) for the
exact test, affected records, and the limits of the recorded test results.

## Scope and limitations

- Patient sandbox identifiers differ across sources for the same person. For consent
  tests requiring shared identity, only test anchors `1` and `2` supply a shared
  identifier; see the [known issue](KNOWN_ISSUES.md#patient-identifiers-differ-across-sources-for-the-same-person).
- The records are synthetic fixtures, not a representative sample of Israeli
  patients. They are unsuitable for estimating prevalence or making clinical decisions.
- Intentional data-quality problems and inherited clinical limitations are part
  of the accepted release. An exact match also preserves those limitations.
- Original 1k qualification covered all resources and selected envelopes with
  reviewed exceptions. Original 10k official validation sampled twelve graphs;
  it was not exhaustive validation of every scale resource.
- Recorded limitations include culture-profile errors, offline terminology
  warnings, unknown pediatric doses or administration amounts, allergy-related
  prescribing risks, pediatric blood-pressure interpretation, and dated dose-reference
  weights. Automated clinical review is not external human clinical clearance.
- Confidentiality labels are test data, not an access-control implementation.

See the [release notes](releases/synthetic-population/4.2.2-2026-10-10/README.md)
and [release index](releases/synthetic-population/4.2.2-2026-10-10/release-index.json)
for payload identities, validation scope, and limitations.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/synthetic_population/native_generator/` | Pinned native construction code and its required inputs |
| `src/synthetic_population/release_repairs.py` | Release generation, repairs, and hash verification |
| `src/synthetic_population/data/` | Current catalogs, release configurations, and repair definitions |
| `src/synthetic_population/reference/` | Restored public inputs; ignored by Git |
| `tests/` | Generator, repair, and supporting checks |
| `scripts/rehydrate_references.py` | Restore inputs directly from a source checkout |
| `releases/` | Compressed accepted datasets and their metadata |
| `generated/` | Local outputs; ignored by Git |

## License and contributing

Project-owned code, documentation, and original synthetic content are offered
under [Apache License 2.0](LICENSE). Embedded terminology and reference material
retain their own terms; Apache 2.0 does not replace them. Read
[NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for
attribution and the applicable terminology and reference-package terms.

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md),
[SECURITY.md](SECURITY.md), and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
HL7 and FHIR are registered trademarks of Health Level Seven International;
their use here does not imply endorsement.
