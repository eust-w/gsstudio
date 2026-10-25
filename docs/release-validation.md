# Source validation

Validation date: **2026-10-26**. This report covers the original GSStudio source
prepared for publication. It does not certify a hosted service or GPU
inference. Historical reports remain available under `docs/experiments/` with
their original dates and acceptance scope.

Publication status on **2026-10-26**: the canonical repository has not been made
public. This document records local source verification while credential and
repository-history handling remain pending; it does not certify publication.

## Application checks

The retained application and original deployment-contract tests are checked in
the existing Python 3.11 environment and Node.js 24 frontend environment.
Build the frontend before backend tests because the SPA deep-link test needs
`web/dist/index.html`:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
npm --prefix web ci
npm --prefix web test
npm --prefix web run build
GSSTUDIO_MANAGE_VLLM=false .venv/bin/python -m pytest -q
```

| Check | Result |
| --- | --- |
| Original tracked paths | All 98 retained; 88 byte-identical to the source revision |
| Application source and tests | Server, frontend, API documentation, original deployment tests, and dependency baseline retained unchanged |
| Backend | 159 tests passed; two upstream test-client deprecation warnings |
| Frontend | 56 tests passed across eight files |
| TypeScript and production frontend build | Passed; existing viewer bundle-size warning remains |
| Shell syntax and whitespace | Passed |
| Patch behavior | All four patch algorithms have identical AST after excluding only the license-notice insertion mechanism |
| Exact credential scan | 115 candidate source files checked against available known credentials and encoded variants; zero matches |
| Secret-pattern scan | Gitleaks found no secrets in the candidate source snapshot |

The ten original files with byte differences are `.gitignore`, `README.md`,
the two Dockerfiles, the four upstream patch helpers, and two historical WASD
plan documents. Their differences are protective ignore rules, credential/license
documentation, license files copied into images, generated modification notices,
and removal of extra blank lines at the end of those two documents. No application
features, original deployment templates, tests, or historical documents were
removed. Snapshot scanning does not clear historical GitHub commit or pull
request objects; publication history must be reviewed separately.
No cluster deployment, GPU image build, or real inference is included in these
source checks. The original `Dockerfile.lite` recipe is also not rebuilt here;
the CPU container verification recorded for the earlier packaging recipe does
not transfer to it.

The pinned-action [Application checks workflow](../.github/workflows/ci.yml)
runs frontend tests/build and backend tests on Linux, Python 3.11, and Node.js
24. Live remote status is available in
[GitHub Actions](https://github.com/eust-w/gsstudio/actions/workflows/ci.yml).

## Dependency audit results

The dependency baseline was audited again on **2026-10-26** with `npm audit`
and `pip-audit -r requirements-dev.txt`. The results below remain unresolved;
this is not a clean security audit. No dependency versions were upgraded.
Package counts differ from advisory counts, and findings can change as
advisory databases change.

`npm audit` reported **six affected packages: four high, two moderate**:

| Package | Severity |
| --- | --- |
| `nanoid` | High |
| `source-map-js` | High |
| `undici` | High |
| `vite` | High |
| `@vitest/mocker` | Moderate |
| `vitest` | Moderate |

`pip-audit -r requirements-dev.txt`, resolved with Python 3.11, reported
**44 advisories across four packages**:

| Package | Resolved version | Advisory count | Fix version reported by the audit |
| --- | --- | --- | --- |
| `filelock` | 3.32.2 | 3 | 4.2.1 |
| `urllib3` | 2.7.0 | 3 | 2.8.0 |
| `setuptools` | 84.0.0 | 1 | 84.1.0 |
| `Pillow` | 12.3.0 | 37 | 13.0.0 |

These are dependency-tool findings, not an exploitability assessment of every
GSStudio route. Fix versions are audit output, not compatibility-validated
upgrade recommendations. This publication keeps the existing dependency
baseline; major-version upgrades and GPU-stack verification remain future work.
See [security guidance](../SECURITY.md) for exposure boundaries and reporting.

## Source and license scope

The source retains all 98 paths from the original application revision,
including its deployment templates, tests, development scripts, and historical
plans and experiments. Original application code uses Apache-2.0; external
runtimes, model weights, and upstream-derived patch excerpts retain their own
terms. See [third-party notices](../THIRD_PARTY_NOTICES.md).

Publication changes are limited to removing credential text, retaining license
and attribution notices, adding protective ignore rules and CI, and updating
publication/security documentation. Secret resource names are configuration
references; actual values must be provided externally. Model weights, local
databases, caches, and generated output are not included.

## Limits

- Original GPU base-image and node/storage assumptions remain. The repository
  does not provide a complete GPU base-image build recipe or registry access.
- Real GPU inference, model quality, and real-scene browser rendering were not
  validated during these source checks.
- Mock reconstruction demonstrates application state transitions and artifact
  delivery only. Placeholder PLY/SPZ files cannot be rendered as real scenes.
- World-generation creation retains SAM3 access/cache checks even in mock mode.
- The frontend viewer bundle-size warning and unresolved dependency findings
  remain. Functional test success does not establish security.
