# Security

GSStudio retains its original POC deployment behavior. Functional tests do not
establish production security, and current dependencies have unresolved
advisories; see [release validation](docs/release-validation.md).

## Deployment boundaries

- `scripts/dev_local.sh` binds both API and Vite to `0.0.0.0`. Use the loopback
  commands in [deployment guidance](docs/deployment.md) for isolated evaluation,
  or restrict access with your host/network controls.
- Basic Auth is optional and is disabled unless both
  `GSSTUDIO_BASIC_AUTH_USER` and `GSSTUDIO_BASIC_AUTH_PASSWORD` are set. The
  recorded GPU manifest reads them from a Secret; the generic deployment
  manifest does not inject them. Review the exact manifest before exposure.
- Use an independent, strong password; never reuse the username as a password.
  Supply credentials from a secret manager or Kubernetes Secrets. Do not place
  passwords, Hugging Face tokens, registry credentials, TLS keys, or `.env`
  files in Git, images, logs, screenshots, or issue reports.
- Use TLS, authentication, and a restricted reverse proxy for remote access.
  Historical HTTP endpoint records do not endorse sending credentials over
  plaintext HTTP. Temporary tunnel URLs and old health results are not proof
  of current availability or access control.
- The API has permissive CORS defaults. It does not provide per-user task
  isolation, quotas, or authorization for mutually untrusted tenants. Uploaded
  data, logs, and artifacts are accessible to configured application users.
- `service.yaml` exposes vLLM within the cluster on port 8100 as well as the
  API. Restrict access to that service. Review load-balancer and ingress
  manifests before applying them; `deploy.sh` applies the ingress file when
  it exists.
- Restrict upload sizes at the reverse proxy, protect task/model volumes, and
  define retention rules. Keep one API worker and one replica for the SQLite
  queue. Review upstream GPU images, extensions, and models separately.

Mock execution uses placeholder artifacts and does not validate GPU inference.
The repository includes deployment source and historical records, not an
accessible service, model weights, or a reproducible GPU base image.

## Dependency audit

Run audits in a fresh environment:

```sh
npm --prefix web audit
.venv/bin/python -m pip install pip-audit
.venv/bin/python -m pip_audit -r requirements-dev.txt
```

Audit tools may exit nonzero for known findings. `Application checks` verifies
tests and builds; success does not imply a clean dependency audit. The file
`deploy/requirements-security.txt` preserves a compatibility baseline, not a
claim that the listed versions are free of current advisories. Upgrades need
API and GPU-stack compatibility testing.

## Reporting a vulnerability

Use [private vulnerability reporting](https://github.com/eust-w/gsstudio/security/advisories/new)
for sensitive findings. Include the affected revision, a synthetic minimal
reproduction, and impact. Do not disclose live credentials, personal data, or
model weights. If private reporting is unavailable, do not post sensitive
findings in a public issue.

There is no long-term support or response-time commitment. Security changes
will be documented in the repository and release notes.
