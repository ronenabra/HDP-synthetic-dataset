# Third-party notices

The [Apache license](LICENSE) applies to project-owned work. Third-party material
in code, catalogs, fixtures, and archives retains its own terms. The publisher
sources below describe those terms.

## Python software

The runtime source imports only the Python standard library. Python and the
following build/test tools are obtained separately; their implementations are
not vendored in this repository or the project's wheel.

| Component | Use | License | Publisher source |
| --- | --- | --- | --- |
| Python | Interpreter and standard library | Python Software Foundation License and bundled notices | [Python license](https://docs.python.org/3/license.html) |
| setuptools | Build backend, declared `>=68` | MIT | [setuptools license](https://github.com/pypa/setuptools/blob/main/LICENSE) |
| wheel | Wheel building with older build-backend environments | MIT | [wheel license](https://github.com/pypa/wheel/blob/main/LICENSE.txt) |
| pytest | Optional test dependency, declared `>=8.0.0` | MIT | [pytest license](https://github.com/pytest-dev/pytest/blob/main/LICENSE) |
| pluggy | pytest dependency | MIT | [pluggy license](https://github.com/pytest-dev/pluggy/blob/main/LICENSE) |
| iniconfig | pytest dependency | MIT | [iniconfig license](https://github.com/pytest-dev/iniconfig/blob/main/LICENSE) |
| packaging | pytest dependency | Apache-2.0 OR BSD-2-Clause | [packaging licenses](https://github.com/pypa/packaging/blob/main/LICENSE) |
| Pygments | pytest dependency | BSD-2-Clause | [Pygments license](https://github.com/pygments/pygments/blob/master/LICENSE) |
| colorama | pytest dependency on Windows | BSD-3-Clause | [colorama license](https://github.com/tartley/colorama/blob/master/LICENSE.txt) |
| exceptiongroup | pytest dependency on Python before 3.11 | MIT, plus PSF-2.0 notice for copied standard-library code | [exceptiongroup license](https://github.com/agronholm/exceptiongroup/blob/main/LICENSE) |
| tomli | pytest dependency on Python before 3.11 | MIT | [tomli license](https://github.com/hukkin/tomli/blob/master/LICENSE) |

Installed versions vary with platform, Python version, and dependency resolution.
Consult the installed distributions' license files when redistributing their code
or a complete Python environment.

The documentation also uses external command-line tools such as curl, tar,
zstd, and sha256sum. Their binaries are not distributed here. Installing or
redistributing those tools is separate from using this Python source package.

## FHIR specifications, references, and examples

### HL7 FHIR R4

FHIR R4 specification material is published by Health Level Seven International
under [CC0 1.0 and the FHIR legal terms](https://hl7.org/fhir/R4/license.html).
The small synthetic coverage fixture includes adapted FHIR example material.
Third-party terminology referenced by FHIR keeps its own terms. HL7 and FHIR are
registered trademarks of Health Level Seven International; no endorsement is implied.

### Israel Core and HDP

The restore configuration pins:

| Package | Version | What is used |
| --- | --- | --- |
| `il.core.fhir.r4` | `0.21.0` | One city CodeSystem for native release construction |
| `il.core.fhir.r4` | `0.22.0` | Current reference context for generic generation/checks |
| `il.hdp.fhir.r4` | `0.6.0` | Current HDP reference context |

Publisher: Israel Core Team / Israeli Ministry of Health FHIR work.
Canonical sources are the [public package registry](https://packages.fhir.org/),
[Israel Core project](https://simplifier.net/ilcore), and the canonical URLs
inside the resources. Restored package resources are ignored by Git and are
not included in the Python distributions.

The IL Core ImplementationGuide resources declare `CC0-1.0`. Embedded terminology
keeps its own terms. HDP 0.6.0 has no license declaration in its package metadata
or ImplementationGuide; consult the publisher for the applicable reference-resource
terms rather than assuming it inherits FHIR or IL Core's license.

## Terminology included in catalogs, fixtures, and generated records

### LOINC

The required attribution is reproduced in [NOTICE](NOTICE) and
[LOINC_short_license.txt](LOINC_short_license.txt).
LOINC content is used under the publisher's
[Copyright Notice and License](https://loinc.org/license), not Apache 2.0.
Keep the notice with redistributed project material and make it accessible
alongside downloads. LOINC-specific naming and third-party-content conditions
continue to apply.

### SNOMED CT

The records and small terminology tables use selected SNOMED CT identifiers,
descriptions, and mappings. The full SNOMED CT database is not distributed.
That content retains [SNOMED International's applicable terms](https://www.snomed.org/licensing).

Clause 2.2.5 of the published [Affiliate License Agreement](https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/license_agreement_snomed.html)
addresses licensed users transmitting patient messages containing identifiers
and descriptions. Terminology-table and mapping uses must be considered under
the applicable terms as well. This project does not claim that a separate
redistribution agreement is necessarily required, or establish which existing
permissions apply to a particular user. Apache 2.0 does not replace those terms.

### RxNorm

RxNorm normalized names and identifiers created by the U.S. National Library of
Medicine are public-domain material according to the
[RxNorm Terms of Service](https://www.nlm.nih.gov/research/umls/rxnorm/docs/termsofservice.html).
NLM's requested acknowledgement appears in [NOTICE](NOTICE). These fixed inputs
may not reflect the most current or accurate NLM data. The full RxNorm release
also contains proprietary sources with separate restrictions; no blanket rights
to those sources are claimed and no full RxNorm database is included.

### UCUM

The records use unit codes from the Unified Code for Units of Measure, copyright
©1999-2024 Regenstrief Institute, Inc. All rights reserved. See the
[UCUM License, Version 1.1](https://ucum.org/license), including its restrictions
on changing the standard and its warranty disclaimer. The full UCUM specification
is not distributed here; this project does not define a replacement unit standard.

### DICOM and other coding systems

The fixture includes DICOM coding references. DICOM is a registered trademark of
NEMA; see the [official standard](https://dicom.nema.org/) and
[DICOM Standards Committee procedures](https://dicom.nema.org/dicom/geninfo/procedures.pdf)
for the publisher's copyright and excerpt terms. HL7 terminology, Israeli Ministry
of Health codes, and other external identifiers retain their publishers' terms.
An identifier or canonical URL is not a grant to redistribute a full terminology.

## Locality data

The locality catalogs record the Israel Central Bureau of Statistics (CBS) and
the [Israel government data portal](https://data.gov.il/datasets/lamas/2022) as
the sources of locality names/codes and rounded weights. Their recorded extraction
date is **2026-08-12**. Street, building, unit, and postal components are fictional.
Weights were normalized for deterministic fixture stratification and are not
an official or current population table.

The original provenance records “Israel Government Open Data terms”; see the
[Data.gov.il terms](https://data.gov.il/terms-of-use) and
[CBS end-user license](https://www.cbs.gov.il/en/Pages/Enduser-license.aspx).
Source attribution and normalization details are recorded in the locality
catalogs. No CBS or government endorsement is implied.

## Redistribution boundary

Ship [LICENSE](LICENSE), [NOTICE](NOTICE), and these notices with project-owned
material and preserve applicable third-party notices. The accepted archives are
copied without modification; their hashes must not be changed merely to insert
new notices. When sharing archives separately, provide the notices alongside
the download rather than describing all embedded terminology as Apache-licensed.

Applicable third-party terms remain separate from the Apache license for this
project's original work.
