# GSStudio 高质量任务自动恢复

## Goal

High-quality WorldStereo tasks automatically recover from low trajectory
acceptance without relaxing the existing alignment thresholds or allowing
rejected trajectories into 3DGS.

## Policy

- High quality defaults to `recovery_policy=auto`; fast mode disables recovery.
- Start with eight regular-first candidates. At four or five accepted, add four
  normal candidates; at three or fewer accepted, create conservative rescue
  trajectories immediately.
- Rescue trajectories are collision-safe prefixes of regular left/right
  rotations: six capped at 25 degrees, then six capped at 12 degrees.
- The task may generate at most forty WorldStereo videos or spend 120 minutes
  in video generation. A generated-source task may create one additional
  panorama with `base_seed + 1`; uploaded panoramas are never changed.
- Recovery state is durable in `auto_recovery.json`, is merged into
  `quality_report.json`, and is available through the task quality endpoint.

## Validation

Unit and integration tests cover transition selection, durable resume,
trajectory construction, source-specific failure handling, API contracts, and
the frontend recovery status. The preserved failed task `80313f5ea237` is used
only as an isolated replay/experiment baseline; a successful high-quality task
is the non-recovery control.

## Recovery state and source isolation

- The journal is atomically persisted before every expensive batch, records
  pending paths, accepted ids, rejection outcome, cumulative GPU budget, seed,
  source attempt, and terminal diagnosis, then mirrors into
  `quality_report.json`.
- A generated-source fallback runs under
  `recovery_attempts/attempt-N/scene` with `base_seed + N`. The original scene
  is moved only after all source stages pass; `source_promotion.json` makes
  the two directory renames restart-safe.
- An uploaded panorama is never overwritten. Two failed rescue rounds produce
  `source_quality_blocked`; the authenticated `POST /api/tasks/{id}/retry-source`
  endpoint requires a replacement panorama and places it in a new isolated
  attempt directory.
- A manual retry can request an additional deterministic generated source.
  It never resets the task-wide 40-video / 120-minute limit and archives
  downstream checkpoints and partial output before training is rerun.

## Product surface

- `GET /api/tasks/{id}/quality` exposes a compact pollable state.
- New high-quality tasks persist `recovery_policy=auto`; fast DMD tasks
  persist `recovery_policy=off`.
- The create page explains automatic recovery. The detail page displays the
  current recovery action, accepted/required count, pending work, budget,
  terminal diagnosis, and a seed/replacement-panorama retry action.

## Release

Build immutable image `a800-v34`, validate offline model preflight, deploy only
the GSStudio workload, and rollback to `a800-v33` on health, GPU, model-load,
or regression failure. No unrelated Kubernetes workload is deleted or scaled.

Implementation note: `a800-v34` was built as the initial immutable rollout.
The final source-retry archival edge-case fix and reproducible Python-audit
remediation are released as immutable `a800-v36`; it keeps the same
`a800-v33` rollback target and never retags v34.
