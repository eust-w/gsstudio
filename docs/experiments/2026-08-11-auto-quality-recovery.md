# Auto Quality Recovery Experiment Record

## Goal and preserved baseline

This experiment verifies that valid high-quality inputs no longer fail merely
because fewer than six trajectories pass the strict alignment gate. The gate
itself remains unchanged: minimum guidance mask `0.80`, maximum relative depth
deviation `0.50`, and maximum keyframe depth-error ratio `0.10`.

The preserved low-acceptance baseline `80313f5ea237` passed only 2 of 31
WorldStereo trajectories. Task `435d4b33616e` was used for the resumable live
acceptance run without deleting its original panorama, videos, logs, or rescue
paths.

## Deterministic replay and automated coverage

The recovery suite covers arbitrary regular paths (including `traj2`), fewer
than six sources, zero sources, partially rendered rescue directories, stable
recovery indexes, crash resume, source-attempt isolation, best-attempt
promotion, and rejected-trajectory isolation. It also covers all delivery
tiers:

| Strictly accepted | Delivery | Top-level task result |
| --- | --- | --- |
| 6 or more | `strict_3d` | `completed`, whitelist-only 3D training |
| 4 or 5 | `degraded_3d` | `completed`, whitelist-only 3D training with warning |
| 0 through 3 | `panorama_only` | `completed`, 3D data/training stages skipped |

The global journaled budget is 56 WorldStereo trajectories, 10,800 seconds,
and at most two deterministic source regenerations. Transient infrastructure
failures retry twice; content-quality exhaustion selects the best preserved
attempt instead of failing the task.

## Live GPU acceptance: `435d4b33616e`

- Public task: `http://120.48.33.45/tasks/435d4b33616e`
- Source: original `frame_000001.jpg`
- Configuration: high quality, automatic recovery, base seed `1024`, full
  `worldstereo-memory`
- Final state: `completed / strict_3d`
- Final source: attempt 1, deterministic seed `1025`
- Recovery total: 30 of 56 WorldStereo trajectories and 9,472.48 of 10,800
  budgeted seconds; no transient infrastructure retry

### Attempt history

| Attempt | Seed | Initial batch | Refill | Rescue | Final accepted | Quality margin |
| --- | ---: | --- | --- | --- | ---: | ---: |
| 0 | 1024 | 5/8 | 0/4 | 0/6 | 5 | 0.5400674 |
| 1 | 1025 | 4/8 | 2/4 | not needed | 6 | 0.6657242 |

Attempt 0 remained archived while attempt 1 ran. Attempt 1 became active only
after it reached six accepted trajectories; attempt 2 was never scheduled.
No already completed trajectory was recomputed after a retry.

### Planning and alignment evidence

- NavMesh present, `reconstruct_pairs.json` present, no planning log errors.
- 25 valid planned trajectories: 8 regular, 9 surround, 4 exploration, and 4
  reconstruction; two collision-invalid paths were discarded.
- Attempt 1 evaluated 12 trajectories and accepted exactly six.
- Rejection histogram: guidance mask 6, relative depth deviation 3, keyframe
  depth-error ratio 3. A trajectory can contribute to multiple reasons.
- `trajectory_whitelist.json` exactly equals the six accepted IDs from
  `alignment_quality.json`:
  `target_bookshelf_1/traj0`, `target_bookshelf_1/traj1`, `view0/traj1`,
  `view0/traj2`, `view1/traj2`, and `view2/traj2`.
- The six rejected IDs are absent from GS-data generation and 3DGS training.

### Downstream result

| Stage | Result | Duration |
| --- | --- | ---: |
| World expansion and recovery | completed | 7,303.82 s |
| GS data preparation | completed | 43.40 s |
| 3DGS training | completed, 5 GPUs, 2,000/2,000 steps | 191.62 s |

Produced artifacts include:

- `output/ply/point_cloud_1999.spz` — 15,624,698 bytes
- `output/ply/point_cloud_1999.ply` — 50,318,378 bytes
- `output/ply/fuse_simplified.ply` — 538,153 bytes for collision navigation
- `output/ply/position_meta_info.json` — official scene pose and scale
- `output/renders/val_step1999_0000.png` through `_0003.png` — fixed-camera
  validation images
- accepted WorldStereo result videos under `scene/render_results/...`
- `quality_report.json`, `alignment_quality.json`,
  `trajectory_whitelist.json`, `auto_recovery.json`, and
  `source_promotion.json`

## Release and acceptance gates

- Backend: 121 passed.
- Frontend: 37 passed after adding source-panorama selection regressions.
- TypeScript/Vite production build passed; `npm audit --audit-level=high`
  reports zero vulnerabilities.
- The clean Python 3.11 dependency audit reports no known vulnerabilities and
  passes `pip check`; the final code-only rerun also passes `git diff --check`
  and Python compilation.
- Immutable image: `a800-v41`, digest
  `sha256:ddb65575de2b234edf6192faf909d6bf4a26932c3632ca2d08c6c3aefe1e5297`.
- The verified rollback baseline remains deployed-history image `a800-v34`;
  no unverified intermediate tag is used as a rollback target.
- Image preflight passed rtree, SAM3 offline configuration, full WorldStereo,
  VGG16 checksum, recovery ordering/isolation, and the source-panorama web
  asset check.
- Public unauthenticated task API returns `401`; authenticated health, task,
  quality, source panorama, and artifact endpoints return `200`.
- Real Chrome acceptance passed 3D default view, official pose, SPZ streaming,
  focused WASD/Shift/R controls, reset, Panorama/Video/3D fullscreen, and zero
  application-origin console warnings or errors.
- Browser acceptance exposed and fixed one final issue: the Panorama tab had
  selected a generated depth map. `a800-v41` now prioritizes the exact source
  `scene/panorama.png`; the live panorama and fullscreen views were rechecked.

The automated negative-tier fixtures prove that a best attempt with three or
fewer accepted trajectories completes as `panorama_only` without training a
fabricated 3D result. The live run exercised the stronger strict-delivery path;
no destructive GPU duplicate of the same source was needed for that negative
case.
