// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import TaskDetail, { preferredSplatArtifact } from "./TaskDetail";

const { deleteTaskMock, getTaskMock, updateTaskMock } = vi.hoisted(() => ({
  deleteTaskMock: vi.fn().mockResolvedValue(undefined),
  getTaskMock: vi.fn(),
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
  updateTask: updateTaskMock,
}));

vi.mock("../components/LogStream", () => ({ default: () => <div>Logs</div> }));
vi.mock("../components/StageProgress", () => ({ default: () => <div>Stages</div> }));
vi.mock("../components/StatusBadge", () => ({ default: () => <div>Status</div> }));
vi.mock("../components/PanoViewer", () => ({ default: () => <div>Panorama</div> }));
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
  it("falls back to PLY when no SPZ artifact exists", () => {
    const ply = {
      name: "gaussians.ply",
      path: "output/gaussians.ply",
      type: "3dgs" as const,
      size: 1024,
    };

    expect(preferredSplatArtifact([ply])).toEqual(ply);
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
    expect(screen.getByText(/adds untried trajectories in batches of 8/i)).toBeInTheDocument();
    expect(screen.getByText(/5 GPUs.*2,000 steps.*200.*1000.*133/i)).toBeInTheDocument();
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
