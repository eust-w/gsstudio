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

## Effective-input delivery upgrade (a800-v37)

The v37 policy keeps every per-trajectory gate unchanged while making content
quality a delivery decision instead of a task failure:

- `strict_3d`: six or more strictly accepted trajectories.
- `degraded_3d`: four or five strictly accepted trajectories; GS preparation
  receives only the accepted whitelist and the UI reports reduced coverage.
- `panorama_only`: zero to three strictly accepted trajectories; panorama,
  generated videos, and diagnostics are delivered, while GS preparation and
  training are explicitly marked skipped.

The recovery journal migrates in place to version 2. It retains every completed
batch and expands the task-wide ceiling to 56 WorldStereo trajectories, 180
minutes, and two deterministic regenerated sources. Each source attempt is
isolated and scored by accepted count, quality margin, then earlier attempt;
only the best scene is promoted.

Rescue construction now uses every `kind=regular` trajectory named by the
planning report, including `traj2` and nonstandard trajectory names. Camera
poses must be finite and orthonormal, the source start frame must exist, and a
rejected/invalid planning entry is ignored. Existing partial rescue directories
are recovered by their `recovery_source_trajectory` metadata and keep their
indices. A batch requests `min(6, max(2, missing + 1))` candidates, uses fewer
when fewer are safe, and journals `rescue_skipped` rather than failing when none
exist.

Network, NCCL, connection, and timeout failures receive two infrastructure
retries after 30 and 120 seconds. Configuration errors, missing models, invalid
arguments, and exhausted transient retries remain real system failures. The
deployment target was immutable `a800-v37`; because v36 was not deployed, the
production rollback target remained the verified `a800-v34`.

### Recovery renderer follow-up (a800-v38)

The first live v37 resume exposed a parent-image composition defect: v34
already contained the renderer CLI and caption markers, so the v37 patch
mistook the patch as complete and did not define the recovery candidate set.
The renderer then raised `NameError: candidate_trajectory_ids` before any new
WorldStereo inference. v38 makes every patch section independently idempotent
and adds a regression fixture for partially patched parent images. v37 remains
immutable and v34 stays the verified rollback baseline.

The v38 build was intentionally not deployed after the real upstream fixture
showed that its legacy assignment existed but ran after captioning. The final
v39 patch supersedes both builds by anchoring whitelist loading directly after
argument parsing and list filtering directly before the upstream
trajectory-count log, so execution
order is guaranteed even when the parent contains the misplaced legacy block.

### Source-attempt isolation follow-up (a800-v40)

The live seed-1025 transition showed that rescue render checkpoints also need
source-attempt scope. v40 clears rendered candidate ids when a generated or
uploaded source attempt begins, preventing same-named rescue paths from an
older scene being treated as ready. It also carries the original seed through
nested attempts so automatic attempts are exactly `base_seed + 1` and
`base_seed + 2`, never compounded from the prior attempt seed. v39 was
preflighted but not deployed; v40 is the rollout target.
