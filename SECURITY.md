# Security and sensitive reports

This project is a local synthetic-data generator, not a production healthcare
service. It must not be supplied with real patient records or secrets.
Security labels in generated resources describe test scenarios; they do not
enforce access control on a receiving system.

There is no guaranteed security-support period or response-time commitment.
Fixes will normally target the current source tree. Accepted dataset archives
are immutable records; a correction needs a separately identified successor.

## Reporting

For a suspected security problem, use a private reporting facility on the
repository host if one is available, or a private contact published by a
maintainer.

Do not put credentials, real health information, or identifying patient details
in a public issue, pull request, or discussion. A public report may contain a
non-sensitive description and a minimal synthetic reproducer. If you cannot
find a private channel, request one without posting the sensitive details.

Include the affected version, command or configuration, observed impact, and a
small synthetic example when possible. Problems involving unintended real-data
content, identifier leakage, or lost confidentiality labels should be treated
as sensitive. Ordinary reproducibility and synthetic-content issues can be
reported publicly when the evidence contains no sensitive material.
