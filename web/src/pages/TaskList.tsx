import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { listTasks, Task } from "../api";
import StatusBadge from "../components/StatusBadge";

const FILTERS = ["all", "pending", "running", "completed", "failed"] as const;

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

  const fetchTasks = useCallback(async () => {
    try {
      const params: Record<string, string> = {};
      if (filter !== "all") params.status = filter;
      const data = await listTasks(params);
      setTasks(data.tasks);
      setTotal(data.total);
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

  return (
    <div className="page">
      <div className="page-header">
        <h2>Tasks</h2>
        <Link to="/create" className="btn btn-primary">
          + New Task
        </Link>
      </div>

      <div className="filter-bar">
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
            <Link to={`/tasks/${task.id}`} key={task.id} className="task-card-link">
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
          ))}
        </div>
      )}
    </div>
  );
}
