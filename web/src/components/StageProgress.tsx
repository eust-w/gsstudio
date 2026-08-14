import { StageInfo } from "../api";

interface StageProgressProps {
  stages: StageInfo[];
  currentStage?: string;
}

const STAGE_LABELS: Record<string, string> = {
  source_generate: "Source Candidate Generation",
  input_preflight: "Input Preflight",
  panogen: "Panorama Generation",
  post_panorama_preflight: "Generated Panorama Preflight",
  traj_generate: "Trajectory Planning",
  traj_render: "Trajectory Rendering",
  video_gen: "World Expansion",
  gen_gs_data: "GS Data Prep",
  gs_pilot: "3DGS Pilot",
  gs_training: "3DGS Training",
  worldrecon: "3D Reconstruction",
  navigation_stress: "Navigation Stress Test",
};

function stageErrorSummary(error: string): string {
  if (error.includes("post_panorama_preflight") || error.includes("source_replacement_required")) {
    return "Generated panorama is unsuitable for strict free-view reconstruction. Replace the source.";
  }
  if (error.includes("navigation_stress")) {
    return "Navigation validation did not pass. Review the guided repair or source replacement action below.";
  }
  if (error.includes("input_preflight")) {
    return "Input preflight did not pass. Review the rejected input conditions below.";
  }
  if (error.includes("pilot")) {
    return "The 500-step pilot did not meet the strict quality gate.";
  }
  return "This stage did not complete. Technical details are available in the task error panel.";
}

function StageIcon({ status }: { status: string }) {
  switch (status) {
    case "completed":
      return <span className="stage-icon completed">&#10003;</span>;
    case "running":
      return <span className="stage-icon running"><span className="spinner" /></span>;
    case "failed":
      return <span className="stage-icon failed">&#10007;</span>;
    case "skipped":
      return <span className="stage-icon skipped">-</span>;
    default:
      return <span className="stage-icon pending">&#9679;</span>;
  }
}

function formatDuration(seconds?: number): string {
  if (!seconds) return "";
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

export default function StageProgress({ stages, currentStage }: StageProgressProps) {
  return (
    <div className="stage-progress">
      <h3>Pipeline Stages</h3>
      <div className="stage-list">
        {stages.map((stage, idx) => (
          <div
            key={stage.name}
            className={`stage-item ${stage.status} ${stage.name === currentStage ? "current" : ""}`}
          >
            <div className="stage-indicator">
              <StageIcon status={stage.status} />
              {idx < stages.length - 1 && <div className="stage-connector" />}
            </div>
            <div className="stage-info">
              <div className="stage-name">
                {STAGE_LABELS[stage.name] || stage.name}
              </div>
              {stage.duration_seconds != null && (
                <div className="stage-duration">{formatDuration(stage.duration_seconds)}</div>
              )}
              {stage.error && <div className="stage-error">{stageErrorSummary(stage.error)}</div>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
