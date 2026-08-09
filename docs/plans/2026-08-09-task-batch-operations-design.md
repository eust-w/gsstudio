# Task Batch Operations Design

## Scope

Add batch selection and three actions to the task list:

- Cancel selected `pending` or `running` tasks.
- Delete selected `completed`, `failed`, or `cancelled` tasks.
- Retry selected `failed` or `cancelled` tasks.

Mixed selections are supported. Each action operates only on eligible selected tasks and displays its eligible count. Ineligible tasks are left untouched.

## Interaction

Each task card gets a checkbox that does not navigate to the task detail page. The filter row gains a `Select visible` control. When at least one task is selected, a batch toolbar shows the selected count, eligible counts for Cancel/Delete/Retry, and a Clear action.

Deletion requires confirmation because it permanently removes task records and generated files. Cancellation and retry do not require confirmation. Selection is reconciled after every refresh so hidden or deleted task IDs cannot remain selected accidentally.

## API and State Rules

Existing per-task cancel and delete endpoints remain the primitives used by the batch UI. A new `POST /api/tasks/{task_id}/retry` endpoint changes only `failed` or `cancelled` tasks back to `pending`, clears their transient error/current-stage state, and resets displayed progress. Existing checkpoint files remain in place, so the worker can skip completed stages and resume useful work.

The frontend executes eligible operations with `Promise.allSettled`, refreshes the list afterward, removes successful IDs from selection, and reports partial failures without hiding successful work.

## Safety

- Active tasks are never included in batch deletion.
- Completed tasks are never included in retry.
- Delete keeps the existing confirmation and filesystem cleanup semantics.
- Retry preserves inputs, outputs, logs, and checkpoint markers.
- Checkbox interaction stops card navigation.

## Verification

- Backend tests cover retry eligibility and state reset.
- Frontend tests cover selecting visible tasks, mixed-status eligibility, batch cancel, batch delete confirmation, batch retry, and checkbox navigation isolation.
- Production verification uses the public IP with temporary tasks, then removes those tasks.
