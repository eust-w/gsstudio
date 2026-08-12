# Prompt-only source generation release evidence

Date: 2026-08-12

## Release

- Production baseline: `a800-v41`
- Released image: `a800-v42`
- Registry digest: `sha256:e80c0a1bf952689cc3bdaf5ad825ffdd6983b7b069bc0c632c123d78ee935c20`
- Scope: only the `gsstudio/gsstudio` Deployment was updated.
- Rollback target: `a800-v41`

## Automated gates

- Backend: 131 tests passed.
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

Pending live generation completion.
