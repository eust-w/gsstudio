# Auto Quality Recovery Experiment Record

## Baseline and hypothesis

The preserved failure `80313f5ea237` accepted only 2 of 31 WorldStereo
trajectories. Its two accepted paths were short, approximately 35-degree,
horizontal rotations. The hypothesis is therefore that quality improves by
reusing collision-validated prefixes rather than spending GPU time on longer
exploration paths.

## Reproducible offline replay

The deterministic state-machine replay covers the observed low-acceptance
case without accessing or modifying the original task:

| Check | Result |
| --- | --- |
| First inference batch | 8 regular-first trajectories |
| 0--3 accepted | Directly enters rescue round 1; it does not consume the remaining regular backlog |
| Rescue round 1 | Six source prefixes capped at 25 degrees and resampled to 21 poses |
| Rescue round 2 | Six source prefixes capped at 12 degrees and resampled to 21 poses |
| Accepted path handling | Only accepted ids are written to `trajectory_whitelist.json` for `gen_gs_data` |
| Budget | 40 WorldStereo trajectories or 120 minutes across all source attempts |
| Generated source fallback | One isolated `base_seed + 1` attempt; original scene remains archived until promotion |
| Uploaded panorama fallback | Original upload remains unchanged; terminal state is `source_quality_blocked` until a replacement is supplied |

The control path represented by successful task `fac2f9d99c2f` is also
covered: 4--5 accepted routes schedule exactly one four-path normal refill,
and six accepted routes stop recovery immediately.

## Automated evidence

- Backend: `105 passed` after adding durable-state, prefix-geometry,
  whitelist, atomic-promotion, source-retry, API, and worker-resume tests.
- Frontend: `32 passed`; task details show phase, accepted/required count,
  40-path/120-minute budget, terminal diagnosis, and source replacement.
- Build: TypeScript/Vite production build completes. `npm audit
  --audit-level=high` reports no vulnerabilities. A clean Python 3.11
  environment created from `requirements-dev.txt` reports no known
  vulnerabilities with `pip-audit`, passes `pip check`, and passes the same
  105 backend tests. The A800 image installs the audited generic dependency
  pins from `deploy/requirements-security.txt` without upgrading HY-World's
  tested CUDA/model stack.

## Isolated GPU acceptance run

The live acceptance run must use a new task created from the source image,
never mutate `80313f5ea237`. On the initial immutable `a800-v34` rollout
(superseded by `a800-v36` for the manual-retry archival edge case and audited
generic Python dependencies), collect the following before
declaring the experiment complete:

1. `auto_recovery.json`, `quality_report.json`, and `alignment_quality.json`.
2. Accepted/rejected counts and rejection reason histogram for each batch.
3. GPU runtime from the task stage records and the bounded WorldStereo count.
4. The final 3DGS artifact, one fixed-camera screenshot, and the generated
   trajectory video.
5. Browser evidence that the authenticated 3D viewer has no new console
   errors and the public unauthenticated API remains `401`.

The deployment record will be appended with the actual task id and metrics;
this document deliberately distinguishes replay proof from a completed GPU
generation run.

### Active v34 run

- Task: `435d4b33616e` (`frame-000001-auto-recovery-experiment-v34`)
- Source: original `frame_000001.jpg`, uploaded as a new isolated task
- Configuration: high quality, `recovery_policy=auto`, deterministic seed
  `1024`
- Status at creation: running `panogen`; no existing task, panorama, or
  artifact was modified.
