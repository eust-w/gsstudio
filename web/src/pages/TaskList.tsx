import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { cancelTask, deleteTask, listTasks, retryTask, Task } from "../api";
import Icon from "../components/Icon";
import {RuntimeNotice} from "../components/SystemStatus";
import StatusBadge from "../components/StatusBadge";

const FILTERS = ["all", "running", "completed", "failed", "pending", "cancelled"] as const;
const PAGE_SIZE = 12;
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
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState("");
  const [taskType, setTaskType] = useState("");
  const [page, setPage] = useState(0);
  const requestRef = useRef<AbortController | null>(null);
  useEffect(() => {
    const timer = setTimeout(() => { setQuery(search.trim()); setPage(0); }, 250);
    return () => clearTimeout(timer);
  }, [search]);
  const [filter, setFilter] = useState<string>("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [batching, setBatching] = useState<"cancel" | "delete" | "retry" | null>(null);

  const scopeKey = JSON.stringify([filter, query, taskType, page]);
  const scopeRef = useRef(scopeKey);
  scopeRef.current = scopeKey;
  const aliveRef = useRef(true);

  const fetchTasks = useCallback(async () => {
    if (!aliveRef.current || scopeRef.current !== scopeKey) return;
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    try {
      const data = await listTasks({status: filter === "all" ? undefined : filter,
        type: taskType || undefined, q: query || undefined, limit: PAGE_SIZE, offset: page * PAGE_SIZE,
        signal: controller.signal});
      if (controller.signal.aborted) return;
      if (page > 0 && data.tasks.length === 0 && data.total <= page * PAGE_SIZE) {
        setPage(Math.max(0, Math.ceil(data.total / PAGE_SIZE) - 1));
        return;
      }
      setTasks(data.tasks);
      setTotal(data.total);
      const visibleIds = new Set(data.tasks.map((task) => task.id));
      setSelected((current) => new Set([...current].filter((id) => visibleIds.has(id))));
      setError("");
    } catch (e) {
      if (!controller.signal.aborted) setError(e instanceof Error ? e.message : "Failed to load tasks");
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, [filter, query, taskType, page, scopeKey]);

  useEffect(() => {
    aliveRef.current = true;
    setLoading(true);
    setSelected(new Set());
    void fetchTasks();
    // Always refresh: a task can be created or finish in another browser.
    const interval = setInterval(() => { if (!document.hidden) void fetchTasks(); }, 10000);
    return () => { aliveRef.current = false; clearInterval(interval); requestRef.current?.abort(); };
  }, [fetchTasks]);

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
    if (!aliveRef.current || scopeRef.current !== scopeKey) { setBatching(null); return; }
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
    if (!aliveRef.current || scopeRef.current !== scopeKey) { setBatching(null); return; }
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
        <div><h2>Your scenes</h2><p className="page-description">Create, reconstruct, and explore in 3D.</p></div>
        <Link to="/create" className="btn btn-primary"><Icon name="plus" size={20} /> Create scene</Link>
      </div>
      <RuntimeNotice />
      <div className="scene-toolbar">
        <div className="filter-bar" aria-label="Filter by status">
          {FILTERS.map((f) => <button key={f} aria-pressed={filter === f}
            className={`filter-btn ${filter === f ? "active" : ""}`}
            onClick={() => { setFilter(f); setPage(0); }}>
            {f === "all" ? "All tasks" : f === "failed" ? "Needs attention" : f.charAt(0).toUpperCase() + f.slice(1)}
          </button>)}
        </div>
        <label className="scene-search"><Icon name="search" size={18} /><input aria-label="Search scenes" placeholder="Search scenes..." value={search} onChange={event => setSearch(event.target.value)} /></label>
      </div>
      {(tasks.length > 0 || taskType) && <div className="list-controls">
        <label className="select-visible-control"><input type="checkbox" aria-label="Select visible tasks" checked={allVisibleSelected} onChange={toggleVisible} /><span>Select visible</span></label>
        <select className="type-filter" aria-label="Filter by pipeline" value={taskType} onChange={event => {setTaskType(event.target.value); setPage(0);}}><option value="">All pipelines</option>{Object.entries(TYPE_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select>
        <span className="task-count">{total} tasks</span>
        <button className="btn btn-sm refresh-button" title="Refresh tasks" aria-label="Refresh tasks" onClick={() => void fetchTasks()}><Icon name="refresh" size={16} /></button>
      </div>}
      {error && <div className="error-banner" role="alert">{error} <button className="btn btn-sm" onClick={() => void fetchTasks()}>Try again</button></div>}

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
        <div className="scene-empty"><span className="loading-spinner" /><p>Loading scenes…</p></div>
      ) : tasks.length === 0 ? (
        <div className="scene-empty">
          <div className="empty-cube"><Icon name={error ? "alert" : "cube"} size={86} /></div>
          <h3>{error ? "Your scenes are temporarily unavailable" : query || filter !== "all" || taskType ? "No matching scenes" : "A new perspective starts here"}</h3>
          <p>{error ? "Reconnect to the API to load your existing workspace." : query || filter !== "all" || taskType ? "Try another search or clear the filters to see your scenes." : "Turn photos, video, or an idea into an explorable 3D scene."}</p>
          <div className="empty-actions">{query || filter !== "all" || taskType ? <button className="btn btn-primary" onClick={() => {setSearch(""); setQuery(""); setFilter("all"); setTaskType(""); setPage(0);}}>Clear filters</button> : !error && <Link to="/create" className="btn btn-primary">Create your first scene</Link>}<Link to="/system" className="btn">View system status</Link></div>
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
                  <div className="task-visual"><Icon name={task.type === "worldrecon" ? "camera" : "cube"} size={42} /><span>{TYPE_LABELS[task.type]}</span>{task.execution_mode === "demo" && <span className="demo-badge">DEMO</span>}</div>
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
                      style={{ width: `${Math.max(0, Math.min(1, task.progress || 0)) * 100}%` }}
                    />
                  </div>
                  <div className="progress-text">{Math.round(task.progress * 100)}%</div>
                </div>
              </Link>
            </article>
          ))}
        </div>
      )}
      {!loading && !error && total === 0 && !query && filter === "all" && !taskType && <div className="workflow-guide">
        <div><span>01</span><section><h4>Capture</h4><p>Upload overlapping photos or a video with camera movement.</p></section></div>
        <div><span>02</span><section><h4>Reconstruct</h4><p>Run reconstruction with input and geometry checks.</p></section></div>
        <div><span>03</span><section><h4>Explore</h4><p>Inspect your scene and download verified artifacts.</p></section></div>
      </div>}
      <footer className="scene-footer"><span>{error ? "Workspace unavailable" : total === 0 ? (query || filter !== "all" || taskType ? "0 matching scenes" : "No scenes yet") : `${page * PAGE_SIZE + 1}–${Math.min((page + 1) * PAGE_SIZE, total)} of ${total} scenes`}</span>
        {total > PAGE_SIZE ? <div className="pagination"><button className="btn btn-sm" disabled={page === 0} onClick={() => setPage(value => value - 1)}>Previous</button><span>Page {page + 1} of {Math.ceil(total / PAGE_SIZE)}</span><button className="btn btn-sm" disabled={(page + 1) * PAGE_SIZE >= total} onClick={() => setPage(value => value + 1)}>Next</button></div> : <span><Icon name="monitor" size={16} /> Local workspace</span>}
      </footer>
    </div>
  );
}
