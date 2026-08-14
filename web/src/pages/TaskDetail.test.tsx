// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import TaskDetail, { preferredPanoramaArtifact, preferredSplatArtifact } from "./TaskDetail";

const { deleteTaskMock, getTaskMock, getTaskQualityMock, repairTaskModelMock, retryTaskSourceMock, updateTaskMock } = vi.hoisted(() => ({
  deleteTaskMock: vi.fn().mockResolvedValue(undefined),
  getTaskMock: vi.fn(),
  getTaskQualityMock: vi.fn().mockResolvedValue({ available: false }),
  repairTaskModelMock: vi.fn().mockResolvedValue(undefined),
  retryTaskSourceMock: vi.fn().mockResolvedValue(undefined),
  updateTaskMock: vi.fn().mockResolvedValue(undefined),
}));

const taskFixture = {
  id: "recon-task",
  name: "Reconstruction",
  type: "worldrecon",
  status: "completed",
  progress: 1,
  current_stage: null,
  created_at: "2026-08-09T00:00:00Z",
  updated_at: "2026-08-09T00:01:00Z",
  config: {
    quality_preset: "high",
    worldstereo_model_type: "worldstereo-memory",
  },
  error: null,
  stages: [],
  artifacts: [
    { name: "panorama.png", path: "output/panorama.png", type: "image", size: 256 },
    { name: "render.mp4", path: "output/render.mp4", type: "video", size: 2048 },
    { name: "gaussians.ply", path: "output/gaussians.ply", type: "3dgs", size: 1024 },
    { name: "gaussians.spz", path: "output/gaussians.spz", type: "3dgs", size: 512 },
    { name: "position_meta_info.json", path: "output/ply/position_meta_info.json", type: "data", size: 512 },
    { name: "fuse_simplified.ply", path: "output/ply/fuse_simplified.ply", type: "3dgs", size: 512 },
  ],
};

vi.mock("../api", () => ({
  cancelTask: vi.fn(),
  deleteTask: deleteTaskMock,
  getArtifactUrl: (_taskId: string, path: string) => `/api/artifacts/${path}`,
  getTask: getTaskMock,
  getTaskQuality: getTaskQualityMock,
  repairTaskModel: repairTaskModelMock,
  retryTaskSource: retryTaskSourceMock,
  updateTask: updateTaskMock,
}));

vi.mock("../components/LogStream", () => ({ default: () => <div>Logs</div> }));
vi.mock("../components/StageProgress", () => ({ default: () => <div>Stages</div> }));
vi.mock("../components/StatusBadge", () => ({ default: () => <div>Status</div> }));
vi.mock("../components/PanoViewer", () => ({
  default: ({ url }: { url: string }) => <div data-testid="panorama-viewer">{url}</div>,
}));
vi.mock("../components/SplatViewer", () => ({
  default: ({ url, metadataUrl, collisionMeshUrl, navigationEnabled }: { url: string; metadataUrl?: string; collisionMeshUrl?: string; navigationEnabled?: boolean }) => (
    <div data-testid="splat-viewer">{[url, metadataUrl, collisionMeshUrl, `nav:${Boolean(navigationEnabled)}`].filter(Boolean).join("|")}</div>
  ),
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.restoreAllMocks();
});

beforeEach(() => {
  getTaskMock.mockResolvedValue(taskFixture);
  getTaskQualityMock.mockResolvedValue({ available: false });
});

describe("TaskDetail", () => {
  it("shows a concise failure with expandable technical details", async () => {
    getTaskMock.mockResolvedValue({
      ...taskFixture,
      status: "failed",
      error_summary: "The source image could not be generated.",
      error: "Traceback: very long internal stack",
    });

    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes><Route path="/tasks/:taskId" element={<TaskDetail />} /></Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText("The source image could not be generated.")).toBeInTheDocument();
    expect(screen.queryByText(/Traceback: very long internal stack/)).not.toBeVisible();
    fireEvent.click(screen.getByText("Technical details"));
    expect(screen.getByText(/Traceback: very long internal stack/)).toBeVisible();
  });

  it("shows prompt source generation progress", async () => {
    getTaskQualityMock.mockResolvedValue({
      available: true,
      source_generation: {
        mode: "prompt",
        state: "expanding_panorama",
        seed: 1024,
        source_attempt: 0,
        artifact_path: "input/input.png",
      },
    });
    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes><Route path="/tasks/:taskId" element={<TaskDetail />} /></Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByRole("heading", { name: "Source generation" })).toBeInTheDocument();
    expect(screen.getByText(/Expanding source view into panorama/i)).toBeInTheDocument();
    expect(screen.getByText("Seed").closest("p")).toHaveTextContent("1024");
  });

  it("falls back only to a trained Gaussian PLY when no SPZ artifact exists", () => {
    const ply = {
      name: "gaussians.ply",
      path: "output/gaussians.ply",
      type: "3dgs" as const,
      size: 1024,
    };

    expect(preferredSplatArtifact([ply])).toEqual(ply);
    expect(preferredSplatArtifact([{
      name: "vis_mesh.ply",
      path: "scene/navmesh/vis_mesh.ply",
      type: "3dgs",
      size: 1024,
    }])).toBeUndefined();
  });

  it("prefers the source panorama over generated depth and normal maps", () => {
    const artifacts = [
      { name: "panorama_L00_A0000.png", path: "scene/gs_data/depths/panorama_L00_A0000.png", type: "image" as const, size: 100 },
      { name: "panorama_L00_A0000.png", path: "scene/gs_data/images/panorama_L00_A0000.png", type: "image" as const, size: 200 },
      { name: "panorama.png", path: "scene/panorama.png", type: "image" as const, size: 300 },
    ];

    expect(preferredPanoramaArtifact(artifacts)).toEqual(artifacts[2]);
  });

  it("renders the source panorama instead of the first depth artifact", async () => {
    getTaskMock.mockResolvedValue({
      ...taskFixture,
      artifacts: [
        { name: "panorama_L00_A0000.png", path: "scene/gs_data/depths/panorama_L00_A0000.png", type: "image", size: 100 },
        { name: "panorama.png", path: "scene/panorama.png", type: "image", size: 300 },
      ],
    });

    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes>
          <Route path="/tasks/:taskId" element={<TaskDetail />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByTestId("panorama-viewer")).toHaveTextContent(
      "/api/artifacts/scene/panorama.png",
    );
  });

  it("starts streaming SPZ instead of an earlier PLY artifact", async () => {
    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes>
          <Route path="/tasks/:taskId" element={<TaskDetail />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByTestId("splat-viewer")).toHaveTextContent(
      "/api/artifacts/output/gaussians.spz",
    );
    expect(screen.getByTestId("splat-viewer")).toHaveTextContent(
      "/api/artifacts/output/ply/position_meta_info.json",
    );
    expect(screen.getByTestId("splat-viewer")).toHaveTextContent(
      "/api/artifacts/output/ply/fuse_simplified.ply",
    );
  });

  it("shows the actual model, trajectory gates and fixed high-quality training plan", async () => {
    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes>
          <Route path="/tasks/:taskId" element={<TaskDetail />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText("worldstereo-memory")).toBeInTheDocument();
    expect(screen.getByText(/first batch 8 regular trajectories.*at least 6 alignment-approved.*target 10/i)).toBeInTheDocument();
    expect(screen.getByText(/target 10.*minimum 6.*24 trajectories.*4 hours.*3 ranked sources/i)).toBeInTheDocument();
    expect(screen.getByText(/500-step pilot.*2,000-step final training/i)).toBeInTheDocument();
  });

  it("shows live conservative-recovery progress and its GPU budget", async () => {
    getTaskQualityMock.mockResolvedValue({
      available: true,
      state: "generating_rescue",
      phase: "rescue_1",
      rescue_round: 1,
      accepted_trajectories: 2,
      required_trajectories: 6,
      attempted_trajectories: 8,
      pending_trajectories: 6,
      generated_video_trajectories: 8,
      video_seconds: 120,
      limits: { maximum_video_trajectories: 40, maximum_video_seconds: 7200 },
      terminal_reason: null,
    });
    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes>
          <Route path="/tasks/:taskId" element={<TaskDetail />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByRole("heading", { name: "Automatic recovery" })).toBeInTheDocument();
    expect(screen.getByText(/Conservative safe trajectories/i)).toBeInTheDocument();
    expect(screen.getByText(/2 accepted.*minimum 6.*target 10/i)).toBeInTheDocument();
    expect(screen.getByText(/8 of 40 WorldStereo trajectories/i)).toBeInTheDocument();
  });

  it("blocks a five-trajectory Worldgen delivery and hides 3D navigation", async () => {
    getTaskMock.mockResolvedValue({
      ...taskFixture,
      type: "worldgen",
      status: "failed",
      error_code: "insufficient_free_view_coverage",
      error: "insufficient_free_view_coverage",
    });
    getTaskQualityMock.mockResolvedValue({
      available: true,
      state: "quality_blocked",
      delivery_tier: null,
      free_view_eligible: false,
      best_source_attempt: 1,
      terminal_reason: "insufficient_free_view_coverage: only 5 passed. Replace the source and retry.",
      recovery_exhausted: true,
      transient_retry_count: 2,
      accepted_trajectories: 5,
      required_trajectories: 6,
      generated_video_trajectories: 31,
      video_seconds: 4000,
      limits: {
        maximum_video_trajectories: 32,
        maximum_video_seconds: 5400,
        maximum_source_regenerations: 1,
      },
    });

    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes>
          <Route path="/tasks/:taskId" element={<TaskDetail />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText("Source needs replacement")).toBeInTheDocument();
    expect(screen.getByText("Best source attempt").closest("p")).toHaveTextContent("1");
    expect(screen.getByText(/31 of 32 WorldStereo trajectories/i)).toBeInTheDocument();
    expect(screen.getByText(/only 5 passed/i)).toBeInTheDocument();
    expect(screen.getByText("Infrastructure retries").closest("p")).toHaveTextContent("2");
    expect(screen.queryByRole("button", { name: "3D Model" })).not.toBeInTheDocument();
  });

  it("offers one guided repair for a sparse giant-Gaussian-only failure", async () => {
    getTaskMock.mockResolvedValue({
      ...taskFixture,
      type: "worldgen",
      status: "failed",
      error_code: "navigation_stress_failed",
      error: "internal navigation traceback",
    });
    getTaskQualityMock.mockResolvedValue({
      available: true,
      product_state: "repair_available",
      next_action: "repair_model",
      user_message: "模型主体质量合格，仅检测到少量巨大 Gaussian。",
      free_view_eligible: false,
      repair_eligibility: {
        eligible: true,
        offending_gaussian_count: 119,
        offending_gaussian_fraction: 0.000179,
        attempts_used: 0,
        maximum_attempts: 1,
      },
      navigation_stress: {
        state: "failed",
        walk: { completed_steps: 1000, accepted_steps: 992 },
        representative_views: Array.from({ length: 32 }),
        micro_translation_pairs: Array.from({ length: 16 }),
        failure_reasons: ["view_01_giant_gaussian"],
      },
    });

    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes><Route path="/tasks/:taskId" element={<TaskDetail />} /></Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText("Safe repair available")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "3D Model" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Revalidate and repair model" }));
    await waitFor(() => expect(repairTaskModelMock).toHaveBeenCalledWith("recon-task"));
    expect(retryTaskSourceMock).not.toHaveBeenCalled();
  });

  it("requires source replacement when generated panorama preflight fails", async () => {
    getTaskMock.mockResolvedValue({
      ...taskFixture,
      type: "worldgen",
      status: "failed",
      error_code: "source_replacement_required",
      error: "post_panorama_preflight_failed",
    });
    getTaskQualityMock.mockResolvedValue({
      available: true,
      product_state: "source_replacement_required",
      next_action: "replace_source",
      user_message: "生成后的全景不适合稳定自由视角重建。",
      post_panorama_preflight: {
        state: "failed",
        composite_score: 0.41,
        failure_reasons: ["person", "perspective_conflict"],
      },
    });

    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes><Route path="/tasks/:taskId" element={<TaskDetail />} /></Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText("Source replacement required")).toBeInTheDocument();
    expect(screen.getByText("Generated panorama preflight").closest("p")).toHaveTextContent("failed");
    expect(screen.queryByRole("button", { name: "Revalidate and repair model" })).not.toBeInTheDocument();
  });

  it("shows 3D and enables navigation only for a validated strict Worldgen delivery", async () => {
    getTaskMock.mockResolvedValue({ ...taskFixture, type: "worldgen", status: "completed" });
    getTaskQualityMock.mockResolvedValue({
      available: true,
      state: "completed",
      delivery_tier: "strict_3d",
      free_view_eligible: true,
      accepted_trajectories: 6,
      required_trajectories: 6,
      navigation_validation: {
        free_view_eligible: true,
        unique_translated_camera_centers: 12,
        azimuth_bin_count: 6,
        validation_render_count: 4,
      },
    });

    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes><Route path="/tasks/:taskId" element={<TaskDetail />} /></Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText("Strict free-view 3DGS")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "3D Model" })).toBeInTheDocument();
    expect(screen.getByTestId("splat-viewer")).toHaveTextContent("nav:true");
    expect(screen.getByText(/12 translated centers.*6\/8 azimuth bins.*4 validation renders/i)).toBeInTheDocument();
  });

  it("hides v4 Worldrecon 3D when navigation stress did not pass", async () => {
    getTaskMock.mockResolvedValue({ ...taskFixture, type: "worldrecon", status: "failed", error_code: "navigation_stress_failed" });
    getTaskQualityMock.mockResolvedValue({
      available: true,
      contract_version: 4,
      input_lane: "production_multiview",
      free_view_eligible: false,
      input_preflight: { state: "passed", candidate_frame_count: 24, selected_frame_count: 16 },
      navigation_stress: { state: "failed", free_view_eligible: false, failure_reasons: ["view_03_giant_gaussian"] },
    });
    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes><Route path="/tasks/:taskId" element={<TaskDetail />} /></Routes>
      </MemoryRouter>,
    );
    expect(await screen.findByText(/v4.*production_multiview/i)).toBeInTheDocument();
    expect(screen.getByText(/passed.*16\/24 frames selected/i)).toBeInTheDocument();
    expect(screen.getByText(/failed.*0\/1000 steps.*0\/32 views.*0\/16 micro pairs/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "3D Model" })).not.toBeInTheDocument();
  });

  it("opens WASD for v4 Worldrecon only after every strict gate passes", async () => {
    getTaskQualityMock.mockResolvedValue({
      available: true,
      contract_version: 4,
      input_lane: "production_multiview",
      free_view_eligible: true,
      input_preflight: { state: "passed", candidate_frame_count: 24, selected_frame_count: 18 },
      navigation_stress: {
        state: "passed",
        free_view_eligible: true,
        walk: { completed_steps: 1000 },
        representative_views: Array.from({ length: 32 }),
        micro_translation_pairs: Array.from({ length: 16 }),
      },
    });
    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes><Route path="/tasks/:taskId" element={<TaskDetail />} /></Routes>
      </MemoryRouter>,
    );
    expect(await screen.findByRole("button", { name: "3D Model" })).toBeInTheDocument();
    expect(screen.getByTestId("splat-viewer")).toHaveTextContent("nav:true");
    expect(screen.queryByText("Legacy unvalidated")).not.toBeInTheDocument();
  });

  it("keeps legacy Worldrecon rotation-only and disables WASD", async () => {
    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes><Route path="/tasks/:taskId" element={<TaskDetail />} /></Routes>
      </MemoryRouter>,
    );
    expect(await screen.findByText("Legacy unvalidated")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "3D Model" })).toBeInTheDocument();
    expect(screen.getByTestId("splat-viewer")).toHaveTextContent("nav:false");
  });

  it("defaults a quality-blocked result to panorama and never renders an empty 3D viewer", async () => {
    getTaskMock.mockResolvedValue({
      ...taskFixture,
      type: "worldgen",
      status: "failed",
      error_code: "insufficient_free_view_coverage",
      artifacts: [
        { name: "panorama.png", path: "output/panorama.png", type: "image", size: 256 },
        { name: "render.mp4", path: "output/render.mp4", type: "video", size: 2048 },
        { name: "vis_mesh.ply", path: "scene/navmesh/vis_mesh.ply", type: "3dgs", size: 1024 },
      ],
    });
    getTaskQualityMock.mockResolvedValue({
      available: true,
      state: "quality_blocked",
      delivery_tier: null,
      free_view_eligible: false,
      terminal_reason: "insufficient_free_view_coverage: replace the source and retry.",
      accepted_trajectories: 3,
      required_trajectories: 6,
      generated_video_trajectories: 20,
      video_seconds: 1800,
      limits: { maximum_video_trajectories: 56, maximum_video_seconds: 10800 },
    });

    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes>
          <Route path="/tasks/:taskId" element={<TaskDetail />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText("Source needs replacement")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Panorama" })).toHaveClass("active");
    expect(screen.queryByRole("button", { name: "3D Model" })).not.toBeInTheDocument();
    expect(screen.queryByTestId("splat-viewer")).not.toBeInTheDocument();
    expect(screen.getByText(/insufficient_free_view_coverage/i)).toBeInTheDocument();
  });

  it("prefers panorama over PLY when no streaming SPZ is available", async () => {
    getTaskMock.mockResolvedValue({
      ...taskFixture,
      artifacts: taskFixture.artifacts.filter((artifact) => artifact.name !== "gaussians.spz"),
    });

    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes>
          <Route path="/tasks/:taskId" element={<TaskDetail />} />
        </Routes>
      </MemoryRouter>,
    );

    await screen.findByRole("button", { name: "Panorama" });
    expect(screen.getByRole("button", { name: "Panorama" })).toHaveClass("active");
  });

  it("offers a failed panorama task a replacement-source retry", async () => {
    getTaskMock.mockResolvedValue({
      ...taskFixture,
      type: "worldgen_from_pano",
      status: "failed",
      error: "source_quality_blocked",
    });
    getTaskQualityMock.mockResolvedValue({
      available: true,
      state: "source_quality_blocked",
      phase: null,
      accepted_trajectories: 2,
      required_trajectories: 6,
      attempted_trajectories: 20,
      pending_trajectories: 0,
      generated_video_trajectories: 20,
      video_seconds: 600,
      limits: { maximum_video_trajectories: 40, maximum_video_seconds: 7200 },
      terminal_reason: "Upload a replacement panorama to retry.",
    });
    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes>
          <Route path="/tasks/:taskId" element={<TaskDetail />} />
        </Routes>
      </MemoryRouter>,
    );
    await screen.findByRole("heading", { name: "Automatic recovery" });

    const panorama = new File(["new panorama"], "replacement.png", { type: "image/png" });
    fireEvent.change(screen.getByLabelText("Replacement panorama"), { target: { files: [panorama] } });
    fireEvent.click(screen.getByRole("button", { name: "Retry with replacement panorama" }));

    await waitFor(() => expect(retryTaskSourceMock).toHaveBeenCalledWith(
      "recon-task", panorama, "worldgen_from_pano", [], undefined,
    ));
  });

  it("offers a quality-blocked prompt task a real source replacement and shows source diagnostics", async () => {
    getTaskMock.mockResolvedValue({
      ...taskFixture,
      type: "worldgen",
      status: "failed",
      error_code: "insufficient_free_view_coverage",
      config: {
        ...taskFixture.config,
        input_mode: "prompt",
      },
    });
    getTaskQualityMock.mockResolvedValue({
      available: true,
      state: "quality_blocked",
      delivery_tier: null,
      free_view_eligible: false,
      recovery_exhausted: true,
      terminal_reason: "insufficient_free_view_coverage: replace the source and retry.",
      source_guidance_collapse: true,
      guidance_diagnostics: {
        detected: true,
        diagnostic_cutoff: 0.6,
        best_minimum_guidance_mask: 0.409,
      },
      source_generation: {
        mode: "prompt",
        state: "completed",
        source_attempt: 0,
        seed: 1024,
        original_prompt: "robotics office",
        effective_prompt: "robotics office with geometry guidance",
        requested_source_width: 1216,
        requested_source_height: 832,
      },
    });
    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes><Route path="/tasks/:taskId" element={<TaskDetail />} /></Routes>
      </MemoryRouter>,
    );
    await screen.findByRole("heading", { name: "Automatic recovery" });

    expect(screen.getByText(/Automatic geometry guidance applied/i)).toBeInTheDocument();
    expect(screen.getByText(/1216×832 fixed landscape/i)).toBeInTheDocument();
    expect(screen.getByText(/best worst-frame 0.409/i)).toBeInTheDocument();
    const replacement = new File(["real office"], "office.jpg", { type: "image/jpeg" });
    fireEvent.change(screen.getByLabelText("Replacement source image (recommended)"), {
      target: { files: [replacement] },
    });
    fireEvent.click(screen.getByRole("button", { name: "Retry with replacement source" }));

    await waitFor(() => expect(retryTaskSourceMock).toHaveBeenCalledWith(
      "recon-task", replacement, "worldgen", [], undefined,
    ));
  });

  it("defaults to 3D and exposes fullscreen for every preview type", async () => {
    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes>
          <Route path="/tasks/:taskId" element={<TaskDetail />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByTestId("splat-viewer")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Enter 3D Model fullscreen" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Video" }));
    expect(screen.getByRole("button", { name: "Enter Video fullscreen" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Panorama" }));
    expect(screen.getByRole("button", { name: "Enter Panorama fullscreen" })).toBeInTheDocument();
  });

  it("edits the task name from the detail page", async () => {
    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes>
          <Route path="/tasks/:taskId" element={<TaskDetail />} />
        </Routes>
      </MemoryRouter>,
    );
    await screen.findByRole("heading", { name: "Reconstruction" });

    fireEvent.click(screen.getByRole("button", { name: "Edit task" }));
    fireEvent.change(screen.getByLabelText("Task name"), { target: { value: "Renamed task" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(updateTaskMock).toHaveBeenCalledWith("recon-task", { name: "Renamed task" }));
  });

  it("deletes a completed task after confirmation", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes>
          <Route path="/tasks/:taskId" element={<TaskDetail />} />
          <Route path="/" element={<div>Task list</div>} />
        </Routes>
      </MemoryRouter>,
    );
    await screen.findByRole("heading", { name: "Reconstruction" });

    fireEvent.click(screen.getByRole("button", { name: "Delete task" }));

    await waitFor(() => expect(deleteTaskMock).toHaveBeenCalledWith("recon-task"));
    expect(await screen.findByText("Task list")).toBeInTheDocument();
  });
});
