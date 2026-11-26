import {lazy, Suspense, useEffect, useState} from "react";
import {Link, useParams} from "react-router-dom";
import {getArtifactUrl, getTask, getTaskQuality, TaskDetail, TaskQualityRecovery} from "../api";

const SplatViewer = lazy(() => import("../components/SplatViewer"));

export function viewerAccess(task: TaskDetail, quality: TaskQualityRecovery, path: string) {
  const artifact = task.artifacts.find(item => item.path === path && item.type === "3dgs" && (/\.spz$/i.test(item.name) || /gaussian.*\.ply$/i.test(item.name)));
  if (!artifact) return {allowed: false, navigation: false, reason: "This file is not a supported Gaussian scene artifact."};
  if (task.execution_mode === "demo" || task.config.execution_mode === "demo") return {allowed: false, navigation: false, reason: "Demo tasks do not contain a real 3D model."};
  const legacy = task.execution_mode !== "real" && task.config.execution_mode !== "real" && task.type === "worldrecon" && task.status === "completed" && (!quality.contract_version || quality.contract_version < 4);
  const verified = task.status === "completed" && quality.free_view_eligible === true && (task.type === "worldrecon" || quality.delivery_tier === "strict_3d");
  return {allowed: legacy || verified, navigation: verified && !legacy, reason: legacy ? "Historical model: inspection only. Free-view navigation has not been validated." : verified ? "" : "This scene has not passed the strict 3D delivery checks. Review its quality report first."};
}

export default function ViewerPage() {
  const {taskId, artifactPath} = useParams<{taskId: string; artifactPath: string}>();
  const [data, setData] = useState<{task: TaskDetail; quality: TaskQualityRecovery} | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let alive = true;
    setData(null); setError("");
    if (taskId) Promise.all([getTask(taskId), getTaskQuality(taskId)]).then(([task, quality]) => {
      if (alive) setData({task, quality});
    }).catch(cause => { if (alive) setError(cause instanceof Error ? cause.message : "Unable to verify scene access"); });
    return () => { alive = false; };
  }, [taskId, artifactPath, retry]);
  const access = data && artifactPath ? viewerAccess(data.task, data.quality, artifactPath) : null;
  const metadata = data?.task.artifacts.find(item => item.name === "position_meta_info.json");
  const collision = data?.task.artifacts.find(item => item.name === "fuse_simplified.ply");
  return <div className="viewer-page">
    <div className="viewer-header"><Link to={`/tasks/${taskId}`} className="back-link">← Back to task</Link></div>
    {error ? <div className="error-banner" role="alert">{error}<button className="btn btn-sm" onClick={() => setRetry(value => value + 1)}>Try again</button></div> : !data ? <div className="loading">Checking scene and quality report…</div> : !access?.allowed ? <div className="preview-empty"><h3>3D preview unavailable</h3><p>{access?.reason}</p></div> : <>
      {access.reason && <div className="runtime-notice">{access.reason}</div>}
      <div className="viewer-container"><Suspense fallback={<div className="viewer-loading">Preparing streaming viewer…</div>}><SplatViewer
        url={getArtifactUrl(taskId!, artifactPath!)} metadataUrl={metadata ? getArtifactUrl(taskId!, metadata.path) : undefined}
        collisionMeshUrl={collision ? getArtifactUrl(taskId!, collision.path) : undefined} navigationEnabled={access.navigation}
      /></Suspense></div>
    </>}
  </div>;
}
