# Preview Fullscreen and 3D Keyboard Navigation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Default completed task previews to 3D, add fullscreen to every preview type, and add continuous WASD movement to the live Spark 3D viewer.

**Architecture:** A reusable `PreviewFrame` owns native Fullscreen API state and a CSS overlay fallback around the active preview. `TaskDetail` owns preview priority, while `SplatViewer` owns focus, key state, and frame-rate-independent camera translation without replacing OrbitControls.

**Tech Stack:** React 18, TypeScript, Three.js, Spark, OrbitControls, Vitest, Testing Library, browser Fullscreen API, Kubernetes/Kaniko.

---

### Task 1: Default preview selection and shared fullscreen frame

**Files:**
- Create: `web/src/components/PreviewFrame.tsx`
- Create: `web/src/components/PreviewFrame.test.tsx`
- Modify: `web/src/pages/TaskDetail.tsx`
- Modify: `web/src/pages/TaskDetail.test.tsx`

**Step 1: Write failing tests**

Add tests that assert:

- A task with panorama, video, and 3D artifacts opens on `3D Model`.
- Each preview tab renders an `Enter <type> fullscreen` button.
- Native `requestFullscreen()` and `document.exitFullscreen()` are called.
- Missing/rejected native fullscreen activates an in-page fallback.
- Escape exits fallback fullscreen.

**Step 2: Run tests to verify RED**

Run: `cd web && npm test -- --run src/components/PreviewFrame.test.tsx src/pages/TaskDetail.test.tsx`

Expected: FAIL because `PreviewFrame` does not exist, fullscreen controls are absent, and panorama is selected first.

**Step 3: Implement the minimal frame and priority change**

Create a focus-preserving frame with this public shape:

```tsx
interface PreviewFrameProps {
  label: string;
  children: ReactNode;
}

export default function PreviewFrame({ label, children }: PreviewFrameProps) {
  // Own native fullscreenchange and CSS fallback state.
  // Focus [data-keyboard-viewer] after expansion.
}
```

In `TaskDetail`, order available previews as `3dgs`, `video`, `panorama`, and wrap the active preview with one `PreviewFrame` labelled from the active tab.

**Step 4: Run tests to verify GREEN**

Run: `cd web && npm test -- --run src/components/PreviewFrame.test.tsx src/pages/TaskDetail.test.tsx`

Expected: PASS.

### Task 2: Frame-rate-independent WASD camera movement

**Files:**
- Create: `web/src/components/splatNavigation.ts`
- Create: `web/src/components/splatNavigation.test.ts`
- Modify: `web/src/components/SplatViewer.tsx`

**Step 1: Write failing navigation tests**

Test a real Three.js camera and target:

```ts
const keys = new Set(["KeyW", "KeyD"]);
moveCamera(camera, target, keys, 1, 2);
expect(camera.position.distanceTo(start)).toBeCloseTo(2);
expect(target.clone().sub(startTarget)).toEqual(camera.position.clone().sub(start));
```

Also cover reverse directions, diagonal normalization, and Shift acceleration.

**Step 2: Run tests to verify RED**

Run: `cd web && npm test -- --run src/components/splatNavigation.test.ts`

Expected: FAIL because the navigation module is absent.

**Step 3: Implement navigation and viewer integration**

Implement `moveCamera(camera, target, keys, deltaSeconds, speed)` using the camera view direction and camera-local X axis. Translate both camera and OrbitControls target.

In `SplatViewer`:

- Make `.splat-viewer` focusable with `tabIndex={0}`, `data-keyboard-viewer`, and an accessible label.
- Focus it on pointer down.
- Track `KeyW`, `KeyA`, `KeyS`, `KeyD`, `ShiftLeft`, and `ShiftRight` only while focused/fullscreen.
- Ignore editable form targets.
- Clear keys on blur, visibility change, and cleanup.
- Apply movement in `renderer.setAnimationLoop()` from elapsed milliseconds.
- Base speed on the loaded scene diagonal and multiply it while Shift is held.
- Update the visible hint with mouse, WASD, and Shift controls.

**Step 4: Run navigation and viewer tests**

Run: `cd web && npm test -- --run src/components/splatNavigation.test.ts src/pages/TaskDetail.test.tsx`

Expected: PASS.

### Task 3: Fullscreen and focus styling

**Files:**
- Modify: `web/src/App.css`

**Step 1: Add responsive styles**

Style:

- `.preview-frame` and `.preview-fullscreen-button`
- `.preview-frame:fullscreen`
- `.preview-frame.preview-frame-fallback`
- Full-height `.preview-content`, `.splat-viewer`, `.splat-canvas`, panorama, and video children while expanded
- Visible keyboard focus without covering the rendered model
- Compact control and hint layout at mobile widths

**Step 2: Run the production frontend checks**

Run: `cd web && npm test -- --run && npm run build && npm audit --audit-level=high`

Expected: all tests pass, TypeScript/Vite build succeeds, and no high vulnerabilities are reported.

### Task 4: Full regression and deployment

**Files:**
- Modify: `deploy/k8s/deployment-gpu.yaml`
- Modify: `deploy/k8s/kaniko-build.yaml`
- Modify: `deploy/scripts/prefetch_models.sh`
- Modify: `docs/design.md`

**Step 1: Run repository verification**

Run:

```bash
.venv/bin/python -m pytest -q
cd web && npm test -- --run && npm run build && npm audit --audit-level=high
git diff --check
```

Expected: backend and frontend suites pass with a clean production build.

**Step 2: Build and deploy `a800-v18`**

Update all deployment image references to `a800-v18`, upload the clean build context, run the Kaniko build, apply the GPU deployment, and wait for a Ready rollout with zero restarts.

### Task 5: Public browser acceptance and publication

**Files:**
- No source files unless acceptance exposes a defect.

**Step 1: Exercise the public target flow**

The flow under test is: `http://120.48.33.45` -> open completed task `kldfjas` -> verify 3D is selected -> enter/exit fullscreen -> focus the viewer -> hold W/A/S/D and observe camera movement -> verify Video and Panorama expose and toggle fullscreen.

Use the Browser plugin first. Verify page identity, meaningful DOM, no framework overlay, console health, desktop/mobile screenshots, and target interaction state changes. Do not modify or delete existing tasks.

**Step 2: Review, commit, and push**

Run the full regression again if acceptance required changes, inspect `git diff`, commit with Conventional Commits, and push `main` to both `origin` and `gitea`.
