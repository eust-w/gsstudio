// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import TaskDetail, { preferredPanoramaArtifact, preferredSplatArtifact } from "./TaskDetail";

const { deleteTaskMock, getTaskMock, getTaskQualityMock, retryTaskSourceMock, updateTaskMock } = vi.hoisted(() => ({
  deleteTaskMock: vi.fn().mockResolvedValue(undefined),
  getTaskMock: vi.fn(),
  getTaskQualityMock: vi.fn().mockResolvedValue({ available: false }),
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
  default: ({ url, metadataUrl, collisionMeshUrl }: { url: string; metadataUrl?: string; collisionMeshUrl?: string }) => (
    <div data-testid="splat-viewer">{[url, metadataUrl, collisionMeshUrl].filter(Boolean).join("|")}</div>
  ),
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.restoreAllMocks();
});

beforeEach(() => {
  getTaskMock.mockResolvedValue(taskFixture);
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

  it("falls back to PLY when no SPZ artifact exists", () => {
    const ply = {
      name: "gaussians.ply",
      path: "output/gaussians.ply",
      type: "3dgs" as const,
      size: 1024,
    };

    expect(preferredSplatArtifact([ply])).toEqual(ply);
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
    expect(screen.getByText(/minimum 8 valid trajectories/i)).toBeInTheDocument();
    expect(screen.getByText(/starts with 8 regular-first candidates/i)).toBeInTheDocument();
    expect(screen.getByText(/5 GPUs.*2,000 steps.*200.*1000.*133/i)).toBeInTheDocument();
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
    expect(screen.getByText(/2 of 6 accepted/i)).toBeInTheDocument();
    expect(screen.getByText(/8 of 40 WorldStereo trajectories/i)).toBeInTheDocument();
  });

  it("shows degraded 3D delivery, best attempt, warning, and v37 budget", async () => {
    getTaskQualityMock.mockResolvedValue({
      available: true,
      state: "completed",
      delivery_tier: "degraded_3d",
      best_source_attempt: 1,
      quality_warning: "Only 5 trajectories passed; 3D coverage is reduced.",
      recovery_exhausted: true,
      transient_retry_count: 2,
      accepted_trajectories: 5,
      required_trajectories: 6,
      generated_video_trajectories: 31,
      video_seconds: 4000,
      limits: {
        maximum_video_trajectories: 56,
        maximum_video_seconds: 10800,
        maximum_source_regenerations: 2,
      },
    });

    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes>
          <Route path="/tasks/:taskId" element={<TaskDetail />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText("Degraded 3D")).toBeInTheDocument();
    expect(screen.getByText("Best source attempt").closest("p")).toHaveTextContent("1");
    expect(screen.getByText(/31 of 56 WorldStereo trajectories/i)).toBeInTheDocument();
    expect(screen.getByText(/Only 5 trajectories passed/i)).toBeInTheDocument();
    expect(screen.getByText("Infrastructure retries").closest("p")).toHaveTextContent("2");
  });

  it("defaults a panorama-only completed result to panorama and never renders an empty 3D viewer", async () => {
    getTaskMock.mockResolvedValue({
      ...taskFixture,
      artifacts: [
        { name: "panorama.png", path: "output/panorama.png", type: "image", size: 256 },
        { name: "render.mp4", path: "output/render.mp4", type: "video", size: 2048 },
      ],
    });
    getTaskQualityMock.mockResolvedValue({
      available: true,
      state: "completed",
      delivery_tier: "panorama_only",
      quality_warning: "3D training was safely skipped.",
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

    expect(await screen.findByText("Panorama result")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Panorama" })).toHaveClass("active");
    expect(screen.queryByTestId("splat-viewer")).not.toBeInTheDocument();
    expect(screen.getByText(/3D training was safely skipped/i)).toBeInTheDocument();
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

    await waitFor(() => expect(retryTaskSourceMock).toHaveBeenCalledWith("recon-task", panorama, "worldgen_from_pano"));
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
