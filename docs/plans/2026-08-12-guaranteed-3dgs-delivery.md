# GSStudio Guaranteed 3DGS Delivery

## Product contract

For every valid input, a task may become `completed` only after producing a
viewer-ready SPZ. Strict trajectory gates classify quality; they no longer
decide whether 3DGS training runs.

Delivery tiers are:

- `strict_3d`: at least six accepted trajectories.
- `degraded_3d`: one to five accepted trajectories.
- `anchored_3d`: no accepted trajectories; train only from panorama/polar
  RGB-D anchors and sanitized scene point clouds.

Rejected trajectories never enter GS data preparation. Automatic strict
recovery is bounded to 90 minutes, 32 generated trajectories, and one
additional deterministic source seed.

## Implementation

1. Change recovery finalization to always select one of the three 3D tiers and
   persist the training mode and accepted trajectory count.
2. Allow an empty trajectory whitelist in the upstream GS-data patch while
   retaining panorama/polar anchors and point-cloud initialization. Write an
   atomic `training_input_manifest.json` describing all consumed and excluded
   inputs.
3. Validate the trained PLY/checkpoint/SPZ before completion. Content-caused
   training failure uses deterministic downsampling and an anchored-safe retry;
   persistent infrastructure failure remains a real failure.
4. Extend the quality API and task UI with the actual training mode, anchor
   count, validation status, and three user-visible quality labels. Completed
   tasks default to the 3D preview.
5. Build immutable image `a800-v50`, update only the `gsstudio` Deployment, and
   resume task `3dc27ec897d9` from GS data preparation without regenerating its
   source, panorama, trajectories, or videos.

`a800-v46` was rejected before deployment when its real-SPZ preflight exposed
that the SPZ decoder represents exact opacity endpoints as infinite logits.
The corrected validator checks decoded 0-1 opacity, retains NaN rejection, and
uses the next immutable release tag rather than mutating the rejected image.
`a800-v47` then passed image/SPZ preflight but was also held back before
deployment when the one-off resume rehearsal found that an old completed
video checkpoint still enforced the retired 6-trajectory blocking rule. The
v48 worker regenerates that evidence without blocking tiered training. The
live anchored run then exposed an empty-validation-set exception after 2,000
steps and successful artifact export. v49 makes the hard PLY/checkpoint/SPZ
validator authoritative after any trainer exit, preventing redundant retrains
without weakening the completed-delivery contract. v49 was held before
deployment so that v50 could additionally require a real same-step trainer
checkpoint (rather than accepting an unrelated depth-cache `.pt`) and reuse a
fully validated prior export without starting another training process.

## Acceptance

- No newly completed valid-input task lacks a validated SPZ.
- Rejected trajectories are absent from `training_input_manifest.json` and GS
  camera data.
- `3dc27ec897d9` completes as `anchored_3d` with a 2,000-step checkpoint, PLY,
  and SPZ that load in the authenticated public browser viewer.
- Existing strict task `435d4b33616e` retains its strict whitelist behavior.
- Backend/frontend tests, production build, dependency audits, public 401
  checks, and browser console checks pass before rollout completion.
