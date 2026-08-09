import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  cancelTask,
  getArtifactUrl,
  getTask,
  Artifact,
  TaskDetail as TaskDetailType,
} from "../api";
import LogStream from "../components/LogStream";
import PanoViewer from "../components/PanoViewer";
import StageProgress from "../components/StageProgress";
import StatusBadge from "../components/StatusBadge";

const SplatViewer = lazy(() => import("../components/SplatViewer"));

export function preferredSplatArtifact(artifacts: Artifact[]): Artifact | undefined {
  return artifacts.find((artifact) => artifact.name.toLowerCase().endsWith(".spz")) ?? artifacts[0];
}

export default function TaskDetail() {
  const { taskId } = useParams<{ taskId: string }>();
  const [task, setTask] = useState<TaskDetailType | null>(null);
  const [error, setError] = useState("");
  const [activePreview, setActivePreview] = useState<string>("panorama");
  const [cancelling, setCancelling] = useState(false);

  const fetchTask = useCallback(async () => {
    if (!taskId) return;
    try {
      const data = await getTask(taskId);
      setTask(data);
      setActivePreview((current) => {
        const available = [
          data.artifacts.some((artifact) => artifact.type === "image" && artifact.name.includes("pano")) && "panorama",
          data.artifacts.some((artifact) => artifact.type === "video") && "video",
          data.artifacts.some((artifact) => artifact.type === "3dgs") && "3dgs",
        ].filter(Boolean) as string[];
        return available.includes(current) ? current : (available[0] ?? current);
      });
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load task");
    }
  }, [taskId]);

  useEffect(() => {
    fetchTask();
  }, [fetchTask]);

  // Auto-refresh while running
  useEffect(() => {
    if (!task || (task.status !== "running" && task.status !== "pending")) return;
    const interval = setInterval(fetchTask, 3000);
    return () => clearInterval(interval);
  }, [task?.status, fetchTask]);

  if (!task) {
    return (
      <div className="page">
        {error ? <div className="error-banner">{error}</div> : <div className="loading">Loading...</div>}
      </div>
    );
  }

  const handleCancel = async () => {
    if (!taskId || cancelling) return;
    setCancelling(true);
    try {
      await cancelTask(taskId);
      await fetchTask();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Cancel failed");
    } finally {
      setCancelling(false);
    }
  };

  // Categorize artifacts
  const panoArtifacts = task.artifacts.filter((a) => a.type === "image" && a.name.includes("pano"));
  const videoArtifacts = task.artifacts.filter((a) => a.type === "video");
  const splatArtifacts = task.artifacts.filter((a) => a.type === "3dgs");
  const previewSplat = preferredSplatArtifact(splatArtifacts);
  const imageArtifacts = task.artifacts.filter((a) => a.type === "image" && !a.name.includes("pano"));
  const otherArtifacts = task.artifacts.filter(
    (a) => !["image", "video", "3dgs"].includes(a.type)
  );

  const canCancel = task.status === "pending" || task.status === "running";

  return (
    <div className="page task-detail-page">
      <div className="page-header">
        <div>
          <Link to="/" className="back-link">&larr; Back</Link>
          <h2>{task.name}</h2>
          <StatusBadge status={task.status} />
        </div>
        {canCancel && (
          <button
            className="btn btn-danger"
            onClick={handleCancel}
            disabled={cancelling}
          >
            {cancelling ? "Cancelling..." : "Cancel Task"}
          </button>
        )}
      </div>

      {error && <div className="error-banner">{error}</div>}
      {task.error && <div className="error-banner">Error: {task.error}</div>}

      <div className="task-detail-layout">
        <div className="task-left-panel">
          <StageProgress stages={task.stages} currentStage={task.current_stage} />

          <div className="log-section">
            <h3>Logs</h3>
            <LogStream taskId={task.id} />
          </div>
        </div>

        <div className="task-right-panel">
          <div className="preview-tabs">
            {panoArtifacts.length > 0 && (
              <button
                className={`preview-tab ${activePreview === "panorama" ? "active" : ""}`}
                onClick={() => setActivePreview("panorama")}
              >
                Panorama
              </button>
            )}
            {videoArtifacts.length > 0 && (
              <button
                className={`preview-tab ${activePreview === "video" ? "active" : ""}`}
                onClick={() => setActivePreview("video")}
              >
                Video
              </button>
            )}
            {splatArtifacts.length > 0 && (
              <button
                className={`preview-tab ${activePreview === "3dgs" ? "active" : ""}`}
                onClick={() => setActivePreview("3dgs")}
              >
                3D Model
              </button>
            )}
            {task.artifacts.length === 0 && (
              <span className="no-artifacts">No artifacts yet</span>
            )}
          </div>

          <div className="preview-content">
            {activePreview === "panorama" && panoArtifacts.length > 0 && (
              <PanoViewer url={getArtifactUrl(task.id, panoArtifacts[0].path)} />
            )}
            {activePreview === "video" && videoArtifacts.length > 0 && (
              <video
                controls
                className="video-preview"
                src={getArtifactUrl(task.id, videoArtifacts[0].path)}
              >
                Your browser does not support video playback.
              </video>
            )}
            {activePreview === "3dgs" && previewSplat && (
              <Suspense fallback={<div className="viewer-loading">Preparing streaming viewer…</div>}>
                <SplatViewer url={getArtifactUrl(task.id, previewSplat.path)} />
              </Suspense>
            )}
          </div>

          {task.artifacts.length > 0 && (
            <div className="artifacts-list">
              <h3>Artifacts</h3>
              <table className="artifacts-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Size</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {task.artifacts.map((a: Artifact) => (
                    <tr key={a.path}>
                      <td>{a.name}</td>
                      <td>{a.type}</td>
                      <td>{(a.size / 1024 / 1024).toFixed(1)} MB</td>
                      <td>
                        <a
                          href={getArtifactUrl(task.id, a.path)}
                          download
                          className="btn btn-sm"
                        >
                          Download
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
