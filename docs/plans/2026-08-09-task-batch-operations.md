# Task Batch Operations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add safe batch cancel, delete, and checkpoint-aware retry actions to the task list.

**Architecture:** Keep cancel and delete as existing per-task API primitives, add one retry endpoint, and orchestrate selected tasks in the React list with `Promise.allSettled`. Eligibility is derived from task status in one shared frontend helper so buttons, counts, and executed IDs cannot diverge.

**Tech Stack:** FastAPI, SQLite, React 18, TypeScript, Vitest, Testing Library, Playwright, Kubernetes/Kaniko.

---

### Task 1: Retry API

**Files:**
- Modify: `tests/test_backend.py`
- Modify: `server/routes/tasks.py`

**Step 1: Write the failing tests**

Add API tests proving failed/cancelled tasks become pending with `progress=0`, `current_stage=None`, and `error=None`, while pending/running/completed tasks return HTTP 409 and missing tasks return 404.

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_backend.py -k retry`
Expected: FAIL because `POST /api/tasks/{id}/retry` does not exist.

**Step 3: Implement the minimal endpoint**

Add `POST /tasks/{task_id}/retry`. Validate status against `{failed, cancelled}` and call:

```python
update_task(
    cfg.DB_PATH,
    task_id,
    status=TaskStatus.PENDING.value,
    progress=0.0,
    current_stage=None,
    error=None,
)
```

Do not remove the task directory or checkpoint marker files.

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest -q tests/test_backend.py -k retry`
Expected: all retry tests PASS.

### Task 2: Batch Selection UI Tests

**Files:**
- Create: `web/src/pages/TaskList.test.tsx`
- Modify: `web/src/pages/TaskList.tsx`
- Modify: `web/src/api.ts`

**Step 1: Write the failing tests**

Mock a mixed task list containing pending, running, completed, failed, and cancelled tasks. Test that:

- `Select visible` selects every displayed card.
- Cancel operates only on pending/running IDs.
- Delete confirms and operates only on completed/failed/cancelled IDs.
- Retry operates only on failed/cancelled IDs.
- Clicking a checkbox does not navigate to task detail.

**Step 2: Run tests to verify they fail**

Run: `cd web && npm test -- --run src/pages/TaskList.test.tsx`
Expected: FAIL because selection controls and retry API client are absent.

**Step 3: Add the retry API client**

Add:

```ts
export async function retryTask(id: string): Promise<void> {
  const resp = await fetch(`${API_BASE}/tasks/${id}/retry`, { method: "POST" });
  if (!resp.ok) throw new Error(`Retry failed: ${resp.statusText}`);
}
```

**Step 4: Implement list selection and action orchestration**

Keep selected IDs in `Set<string>`. Derive eligible arrays from status. Use `Promise.allSettled` for each action, refresh after completion, keep only failed IDs selected, and display a concise partial-failure error. Confirm before deletion.

Refactor each card to an `article` containing a separate checkbox and detail link so checkbox clicks never trigger navigation.

**Step 5: Run tests to verify they pass**

Run: `cd web && npm test -- --run src/pages/TaskList.test.tsx`
Expected: all batch UI tests PASS.

### Task 3: Batch Toolbar Styling

**Files:**
- Modify: `web/src/App.css`

**Step 1: Add responsive styles**

Add selected-card highlighting, a top-left checkbox hit area, a batch toolbar, selection count, action grouping, and mobile wrapping. Preserve the existing three-column desktop grid.

**Step 2: Verify production build**

Run: `cd web && npm run build`
Expected: TypeScript and Vite build PASS.

### Task 4: Full Verification and Deployment

**Files:**
- Modify: `deploy/k8s/deployment-gpu.yaml`
- Modify: `deploy/k8s/kaniko-build.yaml`
- Modify: `deploy/scripts/prefetch_models.sh`
- Modify: `docs/design.md`

**Step 1: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all backend tests PASS.

Run: `cd web && npm test -- --run && npm run build && npm audit --audit-level=high`
Expected: all frontend tests PASS, build PASS, zero high-severity vulnerabilities.

**Step 2: Build and deploy the next A800 image**

Increment the image tag consistently, populate the build context, run Kaniko, apply the Deployment, and wait for a successful rollout.

**Step 3: Public browser acceptance**

Create temporary tasks in terminal states, select a mixed set through the public IP, execute retry/cancel/delete, verify eligible-only behavior and confirmations, then remove all temporary records and directories.

**Step 4: Commit and push**

Review `git diff`, commit implementation and deployment manifests, then push `main` to both `origin` and `gitea`.
