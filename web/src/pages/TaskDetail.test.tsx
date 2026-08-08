// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import TaskDetail from "./TaskDetail";

vi.mock("../api", () => ({
  cancelTask: vi.fn(),
  getArtifactUrl: (_taskId: string, path: string) => `/api/artifacts/${path}`,
  getTask: vi.fn().mockResolvedValue({
    id: "recon-task",
    name: "Reconstruction",
    type: "worldrecon",
    status: "completed",
    progress: 1,
    current_stage: null,
    created_at: "2026-08-09T00:00:00Z",
    updated_at: "2026-08-09T00:01:00Z",
    config: {},
    error: null,
    stages: [],
    artifacts: [
      {
        name: "gaussians.ply",
        path: "output/gaussians.ply",
        type: "3dgs",
        size: 1024,
      },
    ],
  }),
}));

vi.mock("../components/LogStream", () => ({ default: () => <div>Logs</div> }));
vi.mock("../components/StageProgress", () => ({ default: () => <div>Stages</div> }));
vi.mock("../components/StatusBadge", () => ({ default: () => <div>Status</div> }));
vi.mock("../components/PanoViewer", () => ({ default: () => <div>Panorama</div> }));
vi.mock("../components/SplatViewer", () => ({
  default: ({ url }: { url: string }) => <div data-testid="splat-viewer">{url}</div>,
}));

describe("TaskDetail", () => {
  it("starts loading the only 3D model as soon as the task page opens", async () => {
    render(
      <MemoryRouter initialEntries={["/tasks/recon-task"]}>
        <Routes>
          <Route path="/tasks/:taskId" element={<TaskDetail />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByTestId("splat-viewer")).toHaveTextContent(
      "/api/artifacts/output/gaussians.ply",
    );
  });
});
