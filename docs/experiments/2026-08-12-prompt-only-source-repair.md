# Prompt-only source generation release evidence

Date: 2026-08-12

## Release

- Production baseline: `a800-v41`
- Released image: `a800-v45`
- Registry digest: recorded after publication below
- Scope: only the `gsstudio/gsstudio` Deployment was updated.
- Rollback target: `a800-v44`

## Automated gates

- Backend: 132 tests passed.
- Frontend: 39 tests passed.
- TypeScript and Vite production build passed.
- `npm audit --audit-level=high`: zero vulnerabilities.
- Python 3.11 `pip-audit`: no known vulnerabilities.
- Python compile check and `git diff --check` passed.

The backend fixtures cover prompt-only atomic creation, upload decoding and RGB
normalization, corrupt replacement isolation, FIFO pending order, exact and
idempotent historical migration, retry-source input strategies, and the
single-load adapter checkpoint contract.

## Image and runtime preflight

The published image was started in an isolated preflight Pod on the production
GPU node without changing the Deployment. It successfully imported `rtree` and
the shipped GSStudio adapter, and verified that the cached
`HunyuanImage3ForCausalMM.generate_image` method accepts `image=None`.

After rollout, startup also verified the offline VGG16, rtree, HunyuanImage-3,
and complete/fast WorldStereo cache contracts. The Pod became Ready with no
restart.

## Historical migration and live GPU experiment

The versioned migration found the historical `worldgen` tasks that had a valid
prompt, no `input/input.png`, an explicit missing-input panogen error, and no
completed panorama checkpoint. Their old panogen logs were archived, the tasks
were requeued in FIFO order, and their URLs and task IDs were retained.

The first migrated task entered the real GPU adapter with:

- `mode=prompt`
- `state=generating_source`
- `source_attempt=0`
- `seed=1024`

This confirms that execution passed the former official CLI image-existence
failure and reached HunyuanImage-3 text-to-image inference. Final source,
panorama, downstream delivery tier, public artifact, and browser evidence are
recorded below after the run completes.

The source image was produced at 1216x832 and expanded to a panorama in the
same process in 825.452 seconds. Its source SHA-256 was
`9c0f178c06626b196434dd1fdaffcfe8d4320ce0e20152fdda5fae6b285100dd`.

The subsequent planner exposed a pre-existing validation mismatch: upstream
only writes `reconstruct_pairs.json` when segmented reconstruction targets are
available, while GSStudio treated the file as unconditional. This scene had no
`camera_trajectory/target_camera.json`, but it did have 15 collision-planned
regular/exploration trajectories. The `a800-v43` follow-up keeps the file
mandatory when segmented targets exist and records a warning otherwise; all
trajectory-count and alignment thresholds remain unchanged. `a800-v42` is the
rollback baseline for this narrow follow-up.

## Public acceptance

The historical task `3dc27ec897d9` resumed in place at its original URL. Its
seed-1024 prompt source was generated at 1024x1024 with SHA-256
`e2d910d2ec3644b46df863b8b8c30119c39b6aec0c0d1f0029f0203787f33a7f`,
and its panorama completed in the same model process in 820.549 seconds.

The strict quality recovery then exercised its full production contract:

- initial source attempt: 8 regular-first candidates plus two six-trajectory
  conservative rescue batches;
- isolated source attempt 1: seed 1025, a distinct 1024x1024 first view with
  SHA-256 `515e85e30449722885ecbec9638bf8d477ec7cb4dc326f992d7c408c9c283dfe`,
  followed by a panorama in the same process in 739.437 seconds;
- attempt 1 also ran 8 regular-first candidates and both conservative rescue
  batches before the atomic batch budget boundary was reached;
- all 40 generated candidates remained below at least one unchanged strict
  alignment threshold, so the accepted whitelist stayed empty and neither GS
  data preparation nor 3DGS training consumed a rejected trajectory.

The task finished as `completed / panorama_only`, with `gen_gs_data` and
`gs_training` explicitly marked `skipped` and an actionable quality warning.
This is the intended valid-input delivery fallback: it returns the generated
first view, panorama, videos, reports, and diagnostics without fabricating a
3D result or converting content quality into a system failure.

The production task API denies unauthenticated access with HTTP 401. With the
configured account, the original task URL and quality endpoint return the
completed delivery tier and authenticated artifacts. The browser task page
shows the source-generation checkpoint and opens the panorama by default
because no SPZ was produced.

Browser acceptance also verified that navigation and reconstruction mesh PLYs
are not advertised as Gaussian previews. A panorama-only task therefore has no
3D Model tab unless a real SPZ or trained Gaussian PLY artifact exists, avoiding
the former unsupported-ASCII-PLY error when a user clicked a NavMesh artifact.
