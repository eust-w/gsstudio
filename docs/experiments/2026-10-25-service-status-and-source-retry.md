# Service investigation and source-retry repair — 2026-10-25

## Scope and outcome

The source-retry repair and documentation update were validated locally.
Production reachability is impaired, and the current cluster, GPU, and deployed
application state cannot be verified through the recorded management endpoint.
The last real GPU acceptance remains the
[2026-08-12 v50 record](2026-08-12-guaranteed-3dgs-delivery.md).

## Live investigation

| Surface | Observation on 2026-10-25 | What this establishes |
|---------|---------------------------|-----------------------|
| `http://120.48.33.45/api/health` | TCP connection timed out after 5 seconds | The recorded public HTTP endpoint was unreachable from the checking host |
| Historical Cloudflare TLS URL in README | HTTP/2 502, `server: cloudflare`, Bad gateway | The recorded tunnel endpoint did not serve the application |
| Kubernetes `/readyz` and `/api/v1/namespaces/gsstudio/pods` | EOF after approximately 5 seconds | No HTTP or authentication response was received |
| Control-plane `43.106.107.38:6443` | TCP connected; TLS handshake did not finish | TCP reachability does not establish a working Kubernetes API |
| Pods, node, GPU processes and memory | Unavailable through the management connection | Actual A800 status and active jobs remain unknown |

The default context pointed to an unrelated local cluster, so checks explicitly
selected the POC context with the recorded control-plane address; the default
was not changed.
Kubernetes checks used direct API paths with bounded overall timeouts to avoid
repeated discovery retries.

Local DNS returned a Fake-IP address for the tunnel. Google DNS-over-HTTPS
resolved two public Cloudflare addresses; requests using `curl --resolve` and
normal TLS certificate verification returned 502 from both. The tunnel result
therefore cannot be explained solely by that local DNS response. Shell,
system, and kubeconfig proxy settings did not provide an alternative route;
ordinary GitHub/Google HTTPS remained reachable. No SSH alias explicitly tied
to this project or GPU node was present.

The remote cause is still undetermined. A working cluster/VPN/jump-host route
or cloud-console access is required before checking Deployment/Pod state,
node readiness, `nvidia-smi`, vLLM, and the application queue. No production
restart, rollout, or inference task was performed. The manifest still points
to `a800-v50`; this repair has not been deployed.

## Source-retry root cause and repair

The task page offered source retry for completed, recovery-exhausted
`anchored_3d` results, but the API's completed-task allowlist omitted that tier.
Those requests returned 409. Zero-trajectory legacy `panorama_only` journals
also became `anchored_3d` when read and hit the same rejection.

- The API now accepts `anchored_3d` alongside `degraded_3d` and the legacy tier
  when recovery is exhausted; existing task-type and high/auto checks remain.
- The UI checks task type, high/auto configuration, report availability, and
  explicit eligible tiers. Strict, unknown-tier, active, and incompatible
  tasks do not display the source-retry entry.
- Existing failed/cancelled UI behavior still requires a quality-blocked
  recovery state. Missing historical configuration values use high/auto
  defaults, matching the API.
- Retry preserves earlier attempts and cumulative video usage, archives
  downstream output/checkpoint markers/promotion metadata, and queues the
  replacement source without lowering any quality gate.

The initial regression run reproduced four API failures (409 instead of 200
for anchored and migrated legacy results across both generation task types)
and five incorrect UI-entry cases. The same tests passed after the repair.

## Local verification

| Check | Result |
|-------|--------|
| Existing environment: `GSSTUDIO_MANAGE_VLLM=false .venv/bin/python -m pytest -q` | 159 passed |
| Isolated Python 3.11 environment resolved from `requirements-dev.txt` | 159 passed; two upstream TestClient deprecation warnings |
| `npm --prefix web test` | 56 passed across 8 files |
| `npm --prefix web run build` | TypeScript and Vite succeeded |
| Browser against the built frontend and real local API | Both source-retry flows returned 200 and became pending; strict/unknown tiers had no retry entry |
| Independent review and `git diff --check` | No correctness/regression finding; whitespace check passed |

The isolated Python command was:

```sh
GSSTUDIO_MANAGE_VLLM=false uv run --isolated --python python3.11 \
  --with-requirements requirements-dev.txt python -m pytest -q
```

Backend tests exercise the real route, SQLite records, recovery journals,
replacement images, and filesystem archives in temporary directories. They
cover both generation types, legacy migration, positive retry behavior,
rejected requests without state mutation, and preservation of prior work.
Frontend tests cover button eligibility, upload requirements, clicks, and
failed/cancelled historical configuration compatibility. GPU inference is
mocked in the automated suite.

Browser verification used isolated fixture tasks in a temporary SQLite/data
directory, with the actual application routes and the built frontend. The
worker was deliberately not started, so no GPU job ran. Clicking next-seed
retry changed an anchored task to `pending` with seed 1025 and attempt 1.
The panorama retry button was disabled until an image was uploaded, then
returned 200 and queued its replacement. Filesystem checks confirmed the
replacement PNG and prior archived output, with the original 20-path/1,200-second
usage retained. Strict and unknown tiers displayed no source-retry entry.
The only observed console error was the existing missing `favicon.ico`.
Local screenshots are retained under `output/playwright/` (Git-ignored):
`2026-10-25-anchored-retry-pending.png` and
`2026-10-25-panorama-retry-pending.png`.

## Existing dependency and build findings

The dependency manifests and lockfile were not changed in this repair.
The August audit result is historical; the current audit is not clean:

- `npm audit --audit-level=high`: six affected packages, four high and two
  moderate findings (`vite`, `vitest`, `@vitest/mocker`, `nanoid`,
  `source-map-js`, and `undici`).
- Python 3.11 `pip-audit -r requirements-dev.txt`: 44 advisories across four
  resolved packages: `filelock 3.32.2` (3), `urllib3 2.7.0` (3),
  `setuptools 84.0.0` (1), and `Pillow 12.3.0` (37). Reported patched versions
  are respectively 4.2.1, 2.8.0, 84.1.0, and 13.0.0. Some require major-version
  compatibility work, including checking the inherited model runtime.
- The older local Python 3.9 environment separately reported 91 advisories in
  13 packages; it differs from the declared requirements and is not evidence
  about the production image. Its `pip check` passed.
- Vite retains the existing large-chunk warning: the SplatViewer bundle is
  approximately 4.99 MB (1.77 MB gzip).

Functional local verification passed. Dependency remediation and a reachable
production environment are still required before claiming a clean release
audit or live deployment acceptance.
