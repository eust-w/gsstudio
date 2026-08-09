import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { cancelTask, deleteTask, listTasks, retryTask, Task } from "../api";
import StatusBadge from "../components/StatusBadge";

const FILTERS = ["all", "pending", "running", "completed", "failed", "cancelled"] as const;
const ACTIVE_STATUSES = new Set<Task["status"]>(["pending", "running"]);
const TERMINAL_STATUSES = new Set<Task["status"]>(["completed", "failed", "cancelled"]);
const RETRYABLE_STATUSES = new Set<Task["status"]>(["failed", "cancelled"]);
const ACTION_FAILURE_LABELS = {
  cancel: "cancelled",
  delete: "deleted",
  retry: "retried",
} as const;

const TYPE_LABELS: Record<string, string> = {
  worldgen: "Text to World",
  worldgen_from_pano: "Panorama to World",
  worldrecon: "Reconstruction",
};

export default function TaskList() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [total, setTotal] = useState(0);
  const [filter, setFilter] = useState<string>("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [batching, setBatching] = useState<"cancel" | "delete" | "retry" | null>(null);

  const fetchTasks = useCallback(async () => {
    try {
      const params: Record<string, string> = {};
      if (filter !== "all") params.status = filter;
      const data = await listTasks(params);
      setTasks(data.tasks);
      setTotal(data.total);
      const visibleIds = new Set(data.tasks.map((task) => task.id));
      setSelected((current) => new Set([...current].filter((id) => visibleIds.has(id))));
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load tasks");
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    fetchTasks();
  }, [fetchTasks]);

  // Auto-refresh when there are running tasks
  useEffect(() => {
    const hasRunning = tasks.some((t) => t.status === "running" || t.status === "pending");
    if (!hasRunning) return;
    const interval = setInterval(fetchTasks, 5000);
    return () => clearInterval(interval);
  }, [tasks, fetchTasks]);

  const selectedTasks = tasks.filter((task) => selected.has(task.id));
  const activeSelected = selectedTasks.filter((task) => ACTIVE_STATUSES.has(task.status));
  const terminalSelected = selectedTasks.filter((task) => TERMINAL_STATUSES.has(task.status));
  const retryableSelected = selectedTasks.filter((task) => RETRYABLE_STATUSES.has(task.status));
  const allVisibleSelected = tasks.length > 0 && tasks.every((task) => selected.has(task.id));

  const toggleTask = (taskId: string) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(taskId)) next.delete(taskId);
      else next.add(taskId);
      return next;
    });
  };

  const toggleVisible = () => {
    setSelected((current) => {
      const next = new Set(current);
      if (allVisibleSelected) tasks.forEach((task) => next.delete(task.id));
      else tasks.forEach((task) => next.add(task.id));
      return next;
    });
  };

  const runBatch = async (
    action: "cancel" | "delete" | "retry",
    eligibleTasks: Task[],
    operation: (taskId: string) => Promise<void>,
  ) => {
    if (batching || eligibleTasks.length === 0) return;
    setBatching(action);
    const results = await Promise.allSettled(eligibleTasks.map((task) => operation(task.id)));
    const succeeded = eligibleTasks
      .filter((_, index) => results[index].status === "fulfilled")
      .map((task) => task.id);
    const failed = results.length - succeeded.length;
    setSelected((current) => {
      const next = new Set(current);
      succeeded.forEach((id) => next.delete(id));
      return next;
    });
    await fetchTasks();
    setError(
      failed > 0
        ? `${failed} task${failed === 1 ? "" : "s"} could not be ${ACTION_FAILURE_LABELS[action]}.`
        : "",
    );
    setBatching(null);
  };

  const deleteSelected = async () => {
    if (terminalSelected.length === 0) return;
    if (!window.confirm(`Delete ${terminalSelected.length} finished task(s) and all generated files?`)) return;
    await runBatch("delete", terminalSelected, deleteTask);
  };

  return (
    <div className="page">
      <div className="page-header">
        <h2>Tasks</h2>
        <Link to="/create" className="btn btn-primary">
          + New Task
        </Link>
      </div>

      <div className="filter-bar">
        <label className="select-visible-control">
          <input
            type="checkbox"
            aria-label="Select visible tasks"
            checked={allVisibleSelected}
            onChange={toggleVisible}
          />
          <span>Select visible</span>
        </label>
        {FILTERS.map((f) => (
          <button
            key={f}
            className={`filter-btn ${filter === f ? "active" : ""}`}
            onClick={() => setFilter(f)}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
        <span className="task-count">{total} tasks</span>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {selectedTasks.length > 0 && (
        <div className="batch-toolbar" role="toolbar" aria-label="Batch task actions">
          <span className="batch-selection-count">{selectedTasks.length} selected</span>
          <div className="batch-actions">
            <button
              className="btn btn-secondary btn-sm"
              disabled={batching !== null || activeSelected.length === 0}
              onClick={() => runBatch("cancel", activeSelected, cancelTask)}
            >
              {batching === "cancel" ? "Cancelling..." : `Cancel active (${activeSelected.length})`}
            </button>
            <button
              className="btn btn-secondary btn-sm"
              disabled={batching !== null || retryableSelected.length === 0}
              onClick={() => runBatch("retry", retryableSelected, retryTask)}
            >
              {batching === "retry" ? "Retrying..." : `Retry failed (${retryableSelected.length})`}
            </button>
            <button
              className="btn btn-danger btn-sm"
              disabled={batching !== null || terminalSelected.length === 0}
              onClick={deleteSelected}
            >
              {batching === "delete" ? "Deleting..." : `Delete finished (${terminalSelected.length})`}
            </button>
            <button className="btn btn-sm" disabled={batching !== null} onClick={() => setSelected(new Set())}>
              Clear
            </button>
          </div>
        </div>
      )}

      {loading ? (
        <div className="loading">Loading...</div>
      ) : tasks.length === 0 ? (
        <div className="empty-state">
          <p>No tasks yet.</p>
          <Link to="/create" className="btn btn-primary">
            Create your first task
          </Link>
        </div>
      ) : (
        <div className="task-grid">
          {tasks.map((task) => (
            <article key={task.id} className={`task-card-shell ${selected.has(task.id) ? "selected" : ""}`}>
              <input
                className="task-select-checkbox"
                type="checkbox"
                aria-label={`Select task ${task.name}`}
                checked={selected.has(task.id)}
                onChange={() => toggleTask(task.id)}
              />
              <Link to={`/tasks/${task.id}`} className="task-card-link">
                <div className="task-card">
                  <div className="task-card-header">
                    <h3>{task.name}</h3>
                    <StatusBadge status={task.status} />
                  </div>
                  <div className="task-card-meta">
                    <span className="task-type">{TYPE_LABELS[task.type] || task.type}</span>
                    <span className="task-date">
                      {new Date(task.created_at).toLocaleString()}
                    </span>
                  </div>
                  {task.current_stage && (
                    <div className="task-current-stage">
                      Stage: {task.current_stage}
                    </div>
                  )}
                  <div className="progress-bar">
                    <div
                      className={`progress-fill ${task.status === "running" ? "pulse" : ""}`}
                      style={{ width: `${task.progress * 100}%` }}
                    />
                  </div>
                  <div className="progress-text">{Math.round(task.progress * 100)}%</div>
                </div>
              </Link>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
