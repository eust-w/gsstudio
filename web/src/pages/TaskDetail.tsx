import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  cancelTask,
  deleteTask,
  getArtifactUrl,
  getTask,
  getTaskQuality,
  retryTaskSource,
  updateTask,
  Artifact,
  TaskDetail as TaskDetailType,
  TaskQualityRecovery,
} from "../api";
import LogStream from "../components/LogStream";
import PanoViewer from "../components/PanoViewer";
import PreviewFrame from "../components/PreviewFrame";
import StageProgress from "../components/StageProgress";
import StatusBadge from "../components/StatusBadge";

const SplatViewer = lazy(() => import("../components/SplatViewer"));

export function preferredSplatArtifact(artifacts: Artifact[]): Artifact | undefined {
  return artifacts.find((artifact) => artifact.name.toLowerCase().endsWith(".spz")) ?? artifacts[0];
}

function findArtifactByName(artifacts: Artifact[], name: string): Artifact | undefined {
  return artifacts.find((artifact) => artifact.name === name);
}

function recoveryLabel(state?: string): string {
  switch (state) {
    case "generating_normal":
      return "Regular candidate batch";
    case "generating_rescue":
    case "needs_rescue":
      return "Conservative safe trajectories";
    case "regenerating_panorama":
      return "Regenerating panorama with the next deterministic seed";
    case "replacing_panorama_source":
      return "Processing replacement panorama";
    case "accepted":
      return "Quality gate passed";
    case "source_quality_blocked":
    case "quality_blocked":
      return "Source needs replacement";
    case "budget_exhausted":
      return "Recovery budget exhausted";
    default:
      return "Preparing automatic recovery";
  }
}

export default function TaskDetail() {
  const { taskId } = useParams<{ taskId: string }>();
  const navigate = useNavigate();
  const [task, setTask] = useState<TaskDetailType | null>(null);
  const [recovery, setRecovery] = useState<TaskQualityRecovery | null>(null);
  const [error, setError] = useState("");
  const [activePreview, setActivePreview] = useState<string>("3dgs");
  const [cancelling, setCancelling] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draftName, setDraftName] = useState("");
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [replacementPanorama, setReplacementPanorama] = useState<File | null>(null);
  const [retryingSource, setRetryingSource] = useState(false);

  const fetchTask = useCallback(async () => {
    if (!taskId) return;
    try {
      const data = await getTask(taskId);
      setTask(data);
      try {
        setRecovery(await getTaskQuality(taskId));
      } catch {
        // Recovery status is an enhancement for new high-quality tasks; a
        // task must remain usable if an older deployment lacks this endpoint.
        setRecovery(null);
      }
      setActivePreview((current) => {
        const available = [
          data.artifacts.some((artifact) => artifact.type === "3dgs") && "3dgs",
          data.artifacts.some((artifact) => artifact.type === "video") && "video",
          data.artifacts.some((artifact) => artifact.type === "image" && artifact.name.includes("pano")) && "panorama",
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

  const startEditing = () => {
    setDraftName(task.name);
    setEditing(true);
  };

  const handleSave = async () => {
    if (!taskId || saving || !draftName.trim()) return;
    setSaving(true);
    try {
      const updated = await updateTask(taskId, { name: draftName.trim() });
      setTask(updated);
      setEditing(false);
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Update failed");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!taskId || deleting) return;
    if (!window.confirm(`Delete task "${task.name}" and all of its generated files?`)) return;
    setDeleting(true);
    try {
      await deleteTask(taskId);
      navigate("/");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
      setDeleting(false);
    }
  };

  const handleRetrySource = async () => {
    if (!taskId || !task || retryingSource) return;
    if (task.type === "worldgen_from_pano" && !replacementPanorama) {
      setError("Choose a replacement panorama before retrying this source.");
      return;
    }
    setRetryingSource(true);
    try {
      await retryTaskSource(taskId, replacementPanorama ?? undefined);
      setReplacementPanorama(null);
      await fetchTask();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Source retry failed");
    } finally {
      setRetryingSource(false);
    }
  };

  // Categorize artifacts
  const panoArtifacts = task.artifacts.filter((a) => a.type === "image" && a.name.includes("pano"));
  const videoArtifacts = task.artifacts.filter((a) => a.type === "video");
  const splatArtifacts = task.artifacts.filter((a) => a.type === "3dgs");
  const previewSplat = preferredSplatArtifact(splatArtifacts);
  const positionMetadata = findArtifactByName(task.artifacts, "position_meta_info.json");
  const collisionMesh = findArtifactByName(task.artifacts, "fuse_simplified.ply");
  const imageArtifacts = task.artifacts.filter((a) => a.type === "image" && !a.name.includes("pano"));
  const otherArtifacts = task.artifacts.filter(
    (a) => !["image", "video", "3dgs"].includes(a.type)
  );

  const canCancel = task.status === "pending" || task.status === "running";
  const canRetrySource =
    (task.status === "failed" || task.status === "cancelled") &&
    (recovery?.state === "source_quality_blocked" || recovery?.state === "quality_blocked");
  const previewLabels: Record<string, string> = {
    "3dgs": "3D Model",
    video: "Video",
    panorama: "Panorama",
  };

  return (
    <div className="page task-detail-page">
      <div className="page-header">
        <div>
          <Link to="/" className="back-link">&larr; Back</Link>
          {editing ? (
            <div className="task-name-editor">
              <label htmlFor="task-name">Task name</label>
              <input
                id="task-name"
                value={draftName}
                onChange={(event) => setDraftName(event.target.value)}
                autoFocus
              />
              <button className="btn btn-primary" onClick={handleSave} disabled={saving || !draftName.trim()}>
                {saving ? "Saving..." : "Save"}
              </button>
              <button className="btn btn-secondary" onClick={() => setEditing(false)} disabled={saving}>
                Cancel
              </button>
            </div>
          ) : (
            <h2>{task.name}</h2>
          )}
          <StatusBadge status={task.status} />
        </div>
        <div className="task-actions">
          {!editing && <button className="btn" onClick={startEditing}>Edit task</button>}
          {canCancel ? (
            <button
              className="btn btn-danger"
              onClick={handleCancel}
              disabled={cancelling}
            >
              {cancelling ? "Cancelling..." : "Cancel Task"}
            </button>
          ) : (
            <button className="btn btn-danger" onClick={handleDelete} disabled={deleting}>
              {deleting ? "Deleting..." : "Delete task"}
            </button>
          )}
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {task.error && <div className="error-banner">Error: {task.error}</div>}

      <div className="task-detail-layout">
        <div className="task-left-panel">
          <StageProgress stages={task.stages} currentStage={task.current_stage} />

          {Boolean(task.config.quality_preset) && (
            <section className="quality-summary" aria-label="Generation configuration">
              <h3>Generation configuration</h3>
              <p><span className="quality-label">Preset</span> {String(task.config.quality_preset)}</p>
              <p><span className="quality-label">Actual model</span> {String(task.config.worldstereo_model_type ?? "not recorded")}</p>
              {task.config.quality_preset === "high" ? (
                <>
                  <p><span className="quality-label">Trajectory gate</span> minimum 8 valid trajectories; at least 6 alignment-approved trajectories</p>
                  <p><span className="quality-label">Recovery</span> starts with 8 regular-first candidates, then uses bounded safe-prefix rescue paths; rejected trajectories never enter 3DGS</p>
                  <p><span className="quality-label">Training</span> 5 GPUs · 2,000 steps · refine 200 / 1000 / 133</p>
                </>
              ) : (
                <p><span className="quality-label">Trajectory gate</span> fast DMD mode (quality report still recorded)</p>
              )}
            </section>
          )}

          {recovery?.available && (
            <section className="quality-summary recovery-summary" aria-label="Automatic recovery status">
              <h3>Automatic recovery</h3>
              <p><span className="quality-label">Current action</span> {recoveryLabel(recovery.state)}</p>
              <p><span className="quality-label">Quality gate</span> {recovery.accepted_trajectories ?? 0} of {recovery.required_trajectories ?? 6} accepted</p>
              <p><span className="quality-label">WorldStereo budget</span> {recovery.generated_video_trajectories ?? 0} of {recovery.limits?.maximum_video_trajectories ?? 40} WorldStereo trajectories</p>
              <p><span className="quality-label">Time budget</span> {Math.round((recovery.video_seconds ?? 0) / 60)} of {Math.round((recovery.limits?.maximum_video_seconds ?? 7200) / 60)} minutes</p>
              {typeof recovery.pending_trajectories === "number" && recovery.pending_trajectories > 0 && (
                <p><span className="quality-label">Pending</span> {recovery.pending_trajectories} trajectories</p>
              )}
              {recovery.terminal_reason && <p className="recovery-diagnosis">{recovery.terminal_reason}</p>}
              {canRetrySource && (
                <div className="source-retry-controls">
                  {task.type === "worldgen_from_pano" && (
                    <label className="form-label" htmlFor="replacement-panorama">
                      Replacement panorama
                      <input
                        id="replacement-panorama"
                        type="file"
                        accept="image/*"
                        onChange={(event) => setReplacementPanorama(event.target.files?.[0] ?? null)}
                      />
                    </label>
                  )}
                  <button
                    className="btn btn-primary"
                    onClick={handleRetrySource}
                    disabled={retryingSource || (task.type === "worldgen_from_pano" && !replacementPanorama)}
                  >
                    {retryingSource
                      ? "Queuing source retry..."
                      : task.type === "worldgen_from_pano"
                        ? "Retry with replacement panorama"
                        : "Retry with next seed"}
                  </button>
                </div>
              )}
            </section>
          )}

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

          {previewLabels[activePreview] && (
            <PreviewFrame label={previewLabels[activePreview]}>
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
                    <SplatViewer
                      url={getArtifactUrl(task.id, previewSplat.path)}
                      metadataUrl={positionMetadata ? getArtifactUrl(task.id, positionMetadata.path) : undefined}
                      collisionMeshUrl={collisionMesh ? getArtifactUrl(task.id, collisionMesh.path) : undefined}
                    />
                  </Suspense>
                )}
              </div>
            </PreviewFrame>
          )}

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
