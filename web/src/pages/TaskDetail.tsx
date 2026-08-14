import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  cancelTask,
  deleteTask,
  getArtifactUrl,
  getTask,
  getTaskQuality,
  repairTaskModel,
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
  return artifacts.find((artifact) => artifact.name.toLowerCase().endsWith(".spz"))
    ?? artifacts.find((artifact) => {
      const name = artifact.name.toLowerCase();
      return name.endsWith(".ply") && name.includes("gaussian");
    });
}

export function preferredPanoramaArtifact(artifacts: Artifact[]): Artifact | undefined {
  return artifacts.find((artifact) => artifact.name.toLowerCase() === "panorama.png")
    ?? artifacts.find((artifact) => artifact.path.includes("/gs_data/images/"))
    ?? artifacts[0];
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
    case "needs_regeneration":
      return "Replacing a source with collapsed geometry guidance";
    case "replacing_panorama_source":
      return "Processing replacement panorama";
    case "accepted":
      return "Quality gate passed";
    case "completed":
      return "Best valid result delivered";
    case "source_quality_blocked":
    case "quality_blocked":
      return "Source needs replacement";
    case "budget_exhausted":
      return "Recovery budget exhausted";
    default:
      return "Preparing automatic recovery";
  }
}

function deliveryTierLabel(tier?: TaskQualityRecovery["delivery_tier"]): string | null {
  return tier === "strict_3d" ? "Strict free-view 3DGS" : null;
}

function sourceGenerationLabel(state?: string): string {
  switch (state) {
    case "generating_source": return "Generating source view from prompt";
    case "source_ready": return "Source view ready";
    case "expanding_panorama": return "Expanding source view into panorama";
    case "completed": return "Source and panorama ready";
    case "failed": return "Source generation needs attention";
    default: return "Preparing source generation";
  }
}

export default function TaskDetail() {
  const { taskId } = useParams<{ taskId: string }>();
  const navigate = useNavigate();
  const [task, setTask] = useState<TaskDetailType | null>(null);
  const [recovery, setRecovery] = useState<TaskQualityRecovery | null>(null);
  const [error, setError] = useState("");
  const [activePreview, setActivePreview] = useState<string>("");
  const [cancelling, setCancelling] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draftName, setDraftName] = useState("");
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [replacementPanorama, setReplacementPanorama] = useState<File | null>(null);
  const [replacementImages, setReplacementImages] = useState<File[]>([]);
  const [replacementVideo, setReplacementVideo] = useState<File | null>(null);
  const [retryingSource, setRetryingSource] = useState(false);
  const [repairingModel, setRepairingModel] = useState(false);

  const fetchTask = useCallback(async () => {
    if (!taskId) return;
    try {
      const [taskResult, qualityResult] = await Promise.allSettled([
        getTask(taskId),
        getTaskQuality(taskId),
      ]);
      if (taskResult.status === "rejected") throw taskResult.reason;
      const data = taskResult.value;
      const quality = qualityResult.status === "fulfilled" ? qualityResult.value : null;
      setTask(data);
      setRecovery(quality);
      setActivePreview((current) => {
        const hasSpz = data.artifacts.some((artifact) => artifact.name.toLowerCase().endsWith(".spz"));
        const has3d = Boolean(preferredSplatArtifact(
          data.artifacts.filter((artifact) => artifact.type === "3dgs"),
        ));
        const legacyWorldrecon = data.type === "worldrecon"
          && data.status === "completed"
          && (!quality?.contract_version || quality.contract_version < 4);
        const mayShow3d = legacyWorldrecon || (
          data.status === "completed"
          && quality?.free_view_eligible === true
          && (data.type === "worldrecon" || quality.delivery_tier === "strict_3d")
        );
        const hasVideo = data.artifacts.some((artifact) => artifact.type === "video");
        const hasPanorama = data.artifacts.some(
          (artifact) => artifact.type === "image" && artifact.name.includes("pano"),
        );
        const available = [
          mayShow3d && hasSpz && "3dgs",
          hasPanorama && "panorama",
          hasVideo && "video",
          mayShow3d && has3d && !hasSpz && "3dgs",
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
    if (task.type === "worldrecon" && !replacementVideo && replacementImages.length < 12) {
      setError("Choose a replacement video or at least 12 multi-view images.");
      return;
    }
    setRetryingSource(true);
    try {
      await retryTaskSource(
        taskId,
        replacementPanorama ?? undefined,
        task.type,
        replacementImages,
        replacementVideo ?? undefined,
      );
      setReplacementPanorama(null);
      setReplacementImages([]);
      setReplacementVideo(null);
      await fetchTask();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Source retry failed");
    } finally {
      setRetryingSource(false);
    }
  };

  const handleRepairModel = async () => {
    if (!taskId || repairingModel) return;
    setRepairingModel(true);
    try {
      await repairTaskModel(taskId);
      await fetchTask();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Model repair could not be queued");
    } finally {
      setRepairingModel(false);
    }
  };

  // Categorize artifacts
  const panoArtifacts = task.artifacts.filter((a) => a.type === "image" && a.name.includes("pano"));
  const videoArtifacts = task.artifacts.filter((a) => a.type === "video");
  const splatArtifacts = task.artifacts.filter((a) => a.type === "3dgs");
  const legacyWorldrecon = task.type === "worldrecon"
    && task.status === "completed"
    && (!recovery?.contract_version || recovery.contract_version < 4);
  const mayShow3d = legacyWorldrecon || (
    task.status === "completed"
    && recovery?.free_view_eligible === true
    && (task.type === "worldrecon" || recovery.delivery_tier === "strict_3d")
  );
  const previewSplat = mayShow3d ? preferredSplatArtifact(splatArtifacts) : undefined;
  const previewPanorama = preferredPanoramaArtifact(panoArtifacts);
  const positionMetadata = findArtifactByName(task.artifacts, "position_meta_info.json");
  const collisionMesh = findArtifactByName(task.artifacts, "fuse_simplified.ply");
  const imageArtifacts = task.artifacts.filter((a) => a.type === "image" && !a.name.includes("pano"));
  const otherArtifacts = task.artifacts.filter(
    (a) => !["image", "video", "3dgs"].includes(a.type)
  );

  const canCancel = task.status === "pending" || task.status === "running";
  const canRetrySource =
    (task.status === "failed" || task.status === "cancelled")
    && recovery?.next_action !== "repair_model"
    && (
      recovery?.state === "source_quality_blocked"
      || recovery?.state === "quality_blocked"
      || task.error_code === "insufficient_free_view_coverage"
      || task.error_code === "source_replacement_required"
      || task.error_code === "input_preflight_failed"
      || task.error_code === "pilot_content_failed"
      || task.error_code === "navigation_stress_failed"
    );
  const canRepairModel =
    (task.status === "failed" || task.status === "cancelled")
    && recovery?.next_action === "repair_model"
    && Boolean(recovery.repair_eligibility?.eligible || recovery.repair_eligibility?.requires_reanalysis);
  const tierLabel = deliveryTierLabel(recovery?.delivery_tier);
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
      {task.error && (
        <div className="error-banner task-error-summary">
          <p>{recovery?.user_message || task.error_summary || "The task could not be completed. Review the recommended action below."}</p>
          <details>
            <summary>Technical details</summary>
            <pre>{task.error}</pre>
          </details>
        </div>
      )}

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
                  <p><span className="quality-label">Trajectory gate</span> first batch 8 regular trajectories; at least 6 alignment-approved, target 10</p>
                  <p><span className="quality-label">Recovery</span> target 10 · minimum 6 · global cap 24 trajectories / 4 hours / 3 ranked sources</p>
                  <p><span className="quality-label">Training</span> independent 500-step pilot · fresh 2,000-step final training · rejected trajectories never enter 3DGS</p>
                </>
              ) : (
                <p><span className="quality-label">Preset status</span> legacy preset; new World Generation tasks require strict high quality</p>
              )}
            </section>
          )}

          {legacyWorldrecon && (
            <section className="quality-summary legacy-warning" aria-label="Legacy reconstruction status">
              <h3>Legacy unvalidated</h3>
              <p>This historical reconstruction can be rotated for inspection, but it has no v4 navigation report. WASD is disabled.</p>
            </section>
          )}

          {recovery?.source_generation && (
            <section className="quality-summary source-generation-summary" aria-label="Source generation status">
              <h3>Source generation</h3>
              <p><span className="quality-label">Current action</span> {sourceGenerationLabel(recovery.source_generation.state)}</p>
              <p><span className="quality-label">Mode</span> {recovery.source_generation.mode === "prompt" ? "Prompt to source view" : "Uploaded image"}</p>
              {typeof recovery.source_generation.seed === "number" && (
                <p><span className="quality-label">Seed</span> {recovery.source_generation.seed}</p>
              )}
              {typeof recovery.source_generation.source_attempt === "number" && (
                <p><span className="quality-label">Source attempt</span> {recovery.source_generation.source_attempt}</p>
              )}
              {recovery.source_generation.mode === "prompt" &&
                recovery.source_generation.effective_prompt &&
                recovery.source_generation.effective_prompt !== recovery.source_generation.original_prompt && (
                  <p><span className="quality-label">Prompt refinement</span> Automatic geometry guidance applied</p>
                )}
              {typeof recovery.source_generation.requested_source_width === "number" &&
                typeof recovery.source_generation.requested_source_height === "number" && (
                  <p><span className="quality-label">Source frame</span> {recovery.source_generation.requested_source_width}×{recovery.source_generation.requested_source_height} fixed landscape</p>
                )}
              {recovery.source_generation.error_summary && <p className="quality-warning">{recovery.source_generation.error_summary}</p>}
            </section>
          )}

          {recovery?.available && (
            <section className="quality-summary recovery-summary" aria-label="Automatic recovery status">
              <h3>Automatic recovery</h3>
              {recovery.product_state && (
                <div className={`product-state product-state-${recovery.product_state}`}>
                  <strong>{recovery.product_state === "ready"
                    ? "Ready for free view"
                    : recovery.product_state === "repair_available"
                      ? "Safe repair available"
                      : recovery.product_state === "repairing"
                        ? "Repair and validation in progress"
                        : recovery.product_state === "source_replacement_required"
                          ? "Source replacement required"
                          : "Quality validation in progress"}</strong>
                  {recovery.user_message && <span>{recovery.user_message}</span>}
                </div>
              )}
              {tierLabel && (
                <div className={`delivery-tier delivery-tier-${recovery.delivery_tier}`}>
                  <span>Delivery</span>
                  <strong>{tierLabel}</strong>
                </div>
              )}
              <p><span className="quality-label">Current action</span> {recoveryLabel(recovery.state)}</p>
              {recovery.contract_version && <p><span className="quality-label">Contract</span> v{recovery.contract_version} · {recovery.input_lane}</p>}
              <p><span className="quality-label">Quality gate</span> {recovery.accepted_trajectories ?? 0} accepted · minimum {recovery.required_trajectories ?? 6} · target {recovery.target_trajectories ?? 10}</p>
              <p><span className="quality-label">WorldStereo budget</span> {recovery.global_budget?.worldstereo_trajectories_used ?? recovery.generated_video_trajectories ?? 0} of {recovery.limits?.maximum_video_trajectories ?? 24} WorldStereo trajectories</p>
              <p><span className="quality-label">Task time budget</span> {Math.round((recovery.global_budget?.elapsed_seconds ?? recovery.video_seconds ?? 0) / 60)} of {Math.round((recovery.limits?.maximum_video_seconds ?? 14400) / 60)} minutes</p>
              {recovery.source_pool && (
                <>
                  <p><span className="quality-label">Source pool</span> {recovery.source_pool.accepted_candidate_count ?? recovery.source_pool.selected_ids?.length ?? 0} accepted · selected {recovery.source_pool.selected_candidate_id ?? recovery.source_pool.selected_ids?.[0] ?? "none"} · ranking {(recovery.source_pool.ranking ?? recovery.source_pool.selected_ids ?? []).join(" → ") || "pending"}</p>
                  {recovery.source_pool.candidates && recovery.source_pool.candidates.length > 0 && (
                    <ol className="source-ranking">
                      {recovery.source_pool.candidates.map((candidate) => (
                        <li key={candidate.id ?? String(candidate.seed)}>
                          {candidate.id ?? "candidate"} · seed {candidate.seed ?? "n/a"} · score {candidate.composite_score?.toFixed(3) ?? "pending"} · {candidate.accepted ? "accepted" : (candidate.rejection_reasons ?? ["rejected"]).join(", ")}
                        </li>
                      ))}
                    </ol>
                  )}
                </>
              )}
              {recovery.input_preflight && (
                <>
                  <p><span className="quality-label">Input preflight</span> {recovery.input_preflight.state ?? "pending"}{recovery.input_lane === "production_multiview" && ` · ${recovery.input_preflight.selected_frame_count ?? 0}/${recovery.input_preflight.candidate_frame_count ?? 0} frames selected`}{recovery.input_lane !== "production_multiview" && ` · source ${recovery.source_pool?.selected_candidate_id ?? "pending"}`}</p>
                  {recovery.input_preflight.failure_reasons && recovery.input_preflight.failure_reasons.length > 0 && <p className="recovery-diagnosis">{recovery.input_preflight.failure_reasons.join(", ")}</p>}
                </>
              )}
              {recovery.post_panorama_preflight && (
                <>
                  <p><span className="quality-label">Generated panorama preflight</span> {recovery.post_panorama_preflight.state}{typeof recovery.post_panorama_preflight.composite_score === "number" && ` · score ${recovery.post_panorama_preflight.composite_score.toFixed(3)}`}</p>
                  {recovery.post_panorama_preflight.failure_reasons && recovery.post_panorama_preflight.failure_reasons.length > 0 && <p className="recovery-diagnosis">{recovery.post_panorama_preflight.failure_reasons.join(", ")}</p>}
                </>
              )}
              {recovery.pilot_validation && (
                <>
                  <p><span className="quality-label">500-step pilot</span> {recovery.pilot_validation.state} · {recovery.pilot_validation.validation_render_count ?? 0} renders · {recovery.pilot_validation.gaussian_count ?? 0} Gaussians · PSNR {recovery.pilot_validation.metrics?.psnr?.toFixed(2) ?? "n/a"} · SSIM {recovery.pilot_validation.metrics?.ssim?.toFixed(3) ?? "n/a"} · LPIPS {recovery.pilot_validation.metrics?.lpips?.toFixed(3) ?? "n/a"}</p>
                  {recovery.pilot_validation.failure_reasons && recovery.pilot_validation.failure_reasons.length > 0 && <p className="recovery-diagnosis">{recovery.pilot_validation.failure_reasons.join(", ")}</p>}
                </>
              )}
              {recovery.navigation_stress && (
                <>
                  <p><span className="quality-label">Navigation stress</span> {recovery.navigation_stress.state} · {recovery.navigation_stress.walk?.completed_steps ?? 0}/1000 steps ({recovery.navigation_stress.walk?.accepted_steps ?? 0} translated) · {recovery.navigation_stress.representative_views?.length ?? 0}/32 views · {recovery.navigation_stress.micro_translation_pairs?.length ?? 0}/16 micro pairs</p>
                  {recovery.navigation_stress.failure_reasons && recovery.navigation_stress.failure_reasons.length > 0 && <p className="recovery-diagnosis">{recovery.navigation_stress.failure_reasons.join(", ")}</p>}
                </>
              )}
              {recovery.repair_report && (
                <p>
                  <span className="quality-label">Gaussian repair</span>{" "}
                  attempt {recovery.repair_report.attempt ?? 1}/1 · {recovery.repair_report.state}
                  {typeof recovery.repair_report.removed_gaussian_count === "number" && ` · ${recovery.repair_report.removed_gaussian_count} sparse outliers removed`}
                  {recovery.repair_report.original_artifacts_preserved && " · originals archived"}
                </p>
              )}
              {recovery.training_mode && (
                <p><span className="quality-label">Training input</span> Strict trajectories + auxiliary panorama RGB-D</p>
              )}
              {typeof recovery.anchor_view_count === "number" && recovery.anchor_view_count > 0 && (
                <p><span className="quality-label">RGB-D anchors</span> {recovery.anchor_view_count} views</p>
              )}
              {recovery.artifact_validation && (
                <p><span className="quality-label">3D validation</span> SPZ decoded · {String(recovery.artifact_validation.gaussian_count ?? "validated")} Gaussians</p>
              )}
              {recovery.navigation_validation && (
                <p>
                  <span className="quality-label">Free-view validation</span>{" "}
                  {recovery.navigation_validation.unique_translated_camera_centers ?? 0} translated centers · {recovery.navigation_validation.azimuth_bin_count ?? 0}/8 azimuth bins · {recovery.navigation_validation.validation_render_count ?? 0} validation renders
                </p>
              )}
              {typeof recovery.best_source_attempt === "number" && (
                <p><span className="quality-label">Best source attempt</span> {recovery.best_source_attempt}</p>
              )}
              {typeof recovery.transient_retry_count === "number" && recovery.transient_retry_count > 0 && (
                <p><span className="quality-label">Infrastructure retries</span> {recovery.transient_retry_count}</p>
              )}
              {typeof recovery.pending_trajectories === "number" && recovery.pending_trajectories > 0 && (
                <p><span className="quality-label">Pending</span> {recovery.pending_trajectories} trajectories</p>
              )}
              {recovery.guidance_diagnostics?.best_minimum_guidance_mask != null && (
                <p>
                  <span className="quality-label">Source guidance</span>{" "}
                  best worst-frame {recovery.guidance_diagnostics.best_minimum_guidance_mask.toFixed(3)}
                  {typeof recovery.guidance_diagnostics.diagnostic_cutoff === "number" &&
                    ` · early-recovery cutoff ${recovery.guidance_diagnostics.diagnostic_cutoff.toFixed(2)}`}
                </p>
              )}
              {recovery.terminal_reason && <p className="recovery-diagnosis">{recovery.terminal_reason}</p>}
              {recovery.quality_warning && <p className="quality-warning">{recovery.quality_warning}</p>}
              {canRepairModel && (
                <div className="model-repair-controls">
                  <p>
                    This action removes only the renderer-proven sparse giant Gaussians, archives the
                    original PLY/SPZ/checkpoints, and reruns all 1,000 navigation steps and 48 renders.
                  </p>
                  <button className="btn btn-primary" onClick={handleRepairModel} disabled={repairingModel}>
                    {repairingModel ? "Queuing safe repair..." : "Revalidate and repair model"}
                  </button>
                </div>
              )}
              {canRetrySource && (
                <div className="source-retry-controls">
                  {(task.type === "worldgen_from_pano" || task.type === "worldgen") && (
                    <label className="form-label" htmlFor="replacement-panorama">
                      {task.type === "worldgen_from_pano"
                        ? "Replacement panorama"
                        : task.config.input_mode === "image"
                          ? "Replacement source image"
                          : "Replacement source image (recommended)"}
                      <input
                        id="replacement-panorama"
                        type="file"
                        accept="image/*"
                        onChange={(event) => setReplacementPanorama(event.target.files?.[0] ?? null)}
                      />
                    </label>
                  )}
                  {task.type === "worldrecon" && (
                    <>
                      <label className="form-label" htmlFor="replacement-images">
                        Replacement multi-view images (12+)
                        <input
                          id="replacement-images"
                          type="file"
                          accept="image/*"
                          multiple
                          onChange={(event) => setReplacementImages(Array.from(event.target.files ?? []))}
                        />
                      </label>
                      <label className="form-label" htmlFor="replacement-video">
                        Or replacement video
                        <input
                          id="replacement-video"
                          type="file"
                          accept="video/*"
                          onChange={(event) => setReplacementVideo(event.target.files?.[0] ?? null)}
                        />
                      </label>
                    </>
                  )}
                  <button
                    className="btn btn-primary"
                    onClick={handleRetrySource}
                    disabled={retryingSource
                      || ((task.type === "worldgen_from_pano" || task.config.input_mode === "image") && !replacementPanorama)
                      || (task.type === "worldrecon" && !replacementVideo && replacementImages.length < 12)}
                  >
                    {retryingSource
                      ? "Queuing source retry..."
                      : task.type === "worldgen_from_pano"
                        ? "Retry with replacement panorama"
                        : task.type === "worldrecon"
                          ? "Retry with replacement multi-view source"
                        : replacementPanorama || task.config.input_mode === "image"
                          ? "Retry with replacement source"
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
            {previewSplat && (
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
                {activePreview === "panorama" && previewPanorama && (
                  <PanoViewer url={getArtifactUrl(task.id, previewPanorama.path)} />
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
                      navigationEnabled={!legacyWorldrecon && recovery?.free_view_eligible === true}
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
