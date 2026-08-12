# Guaranteed 3DGS delivery acceptance — 2026-08-12

## Scope

This acceptance repaired only historical task `3dc27ec897d9` in place. Its
prompt, source image, panoramas, WorldStereo videos, recovery attempts, logs,
and quality reports were retained. No other historical panorama-only task was
requeued.

## Training inputs and isolation

- Delivery tier: `anchored_3d` (`panorama_anchored`).
- Strictly accepted trajectories: 0.
- Dense panorama and polar RGB-D anchors: 188.
- Rejected trajectories: 20; all are recorded in
  `training_input_manifest.json` and `rejected_trajectories_consumed` is false.
- Initialization uses the global, aligned, and sky point clouds. The anchored
  safe run deterministically reduced 2,802,692 initialization points to
  1,000,000, froze means, disabled densification, and still ran 2,000 steps.

Both the standard and anchored-safe trainers exported their 2,000-step
artifacts before the upstream evaluator divided by an empty validation loader.
The release therefore treats the hard artifact validator—not a post-export
auxiliary evaluation—as the completion boundary. A validated prior export is
reused without launching a third training process.

## Delivered artifacts

- SPZ: `output/ply/point_cloud_1999.spz`, 6,950,372 bytes.
- Gaussian PLY: `output/ply/point_cloud_1999.ply`, 23,504,850 bytes.
- Five same-step checkpoints: `output/ckpts/ckpt_1999_rank0.pt` through
  `ckpt_1999_rank4.pt`, 12,804,089 bytes each.
- Viewer metadata: `output/ply/position_meta_info.json`.
- SPZ decode result: 419,723 finite Gaussians, finite decoded opacities,
  non-degenerate bounds, and 23,653 valid quantized opacity endpoints.

The completion validator now only accepts real trainer checkpoints under
`output/ckpts/ckpt_*.pt` and requires their step number to match the SPZ. Depth
cache `.pt` files cannot satisfy this gate.

## Release gates

- Backend: 145 tests passed.
- Frontend: 39 tests passed.
- TypeScript/Vite production build passed.
- `npm audit --audit-level=high`: zero vulnerabilities.
- Clean Python 3.11 `pip-audit`: no known vulnerabilities; `pip check` passed.
- `git diff --check` passed.
- Immutable image: `a800-v50`, digest
  `sha256:31781e16e3b5f73f2badef91db24267f1d56a2b9d92ecfa37e58f718440308ef`.
- Image preflight decoded the production SPZ, verified the same-step rank-4
  checkpoint, imported rtree/SPZ, and proved resume does not invoke training.
- Startup preflight passed VGG16, rtree, HunyuanImage-3 prompt source,
  WorldStereo offline loading, and all five A800 GPUs.

## Public acceptance

- Unauthenticated task API returned 401.
- Authenticated health, task, and quality APIs returned 200; SPZ byte-range
  download returned 206.
- The task completed at its original URL with `guaranteed_3d: true`, 188 anchor
  views, and the `Panorama-anchored 3DGS` label.
- Playwright loaded 419,723 splats and produced a non-empty room rendering at
  `output/playwright/3dc27ec897d9-3d.png`.
- 3D Model was the default preview. Fullscreen, WASD movement, reset, and the
  official scene pose were visible and functional.
- Real Chrome acceptance confirmed Pointer Lock behavior: after locking in
  fullscreen, the first physical Escape released the mouse while fullscreen
  remained active; the second Escape exited fullscreen.
- The only browser console error was the unrelated missing `favicon.ico`; no
  application or WebGL error was introduced.

Public task: <http://120.48.33.45/tasks/3dc27ec897d9>
