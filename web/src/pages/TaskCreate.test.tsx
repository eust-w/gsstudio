// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import TaskCreate from "./TaskCreate";

const { createTaskMock, systemMock } = vi.hoisted(() => ({
  createTaskMock: vi.fn().mockResolvedValue({ task_id: "new-task" }),
  systemMock: vi.fn(),
}));

vi.mock("../api", () => ({
  createTask: createTaskMock,
}));
vi.mock("../components/SystemStatus", () => ({ useSystemStatus: systemMock }));

beforeEach(() => {
  systemMock.mockReturnValue({ system: null, error: "", refresh: vi.fn() });
  createTaskMock.mockResolvedValue({ task_id: "new-task" });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function openCreate() { return render(<MemoryRouter><TaskCreate /></MemoryRouter>); }
function uploadPhotos(count: number) {
  const files = Array.from({ length: count }, (_, index) => new File(["frame bytes"], `frame-${index}.png`, { type: "image/png", lastModified: index }));
  fireEvent.change(screen.getByLabelText("Drop your scene photos here"), { target: { files } });
  return files;
}

describe("TaskCreate upload workflow", () => {
  it("deduplicates filenames, rejects unsupported files, and permits removing a selected image", () => {
    openCreate();
    uploadPhotos(12);
    fireEvent.change(screen.getByLabelText("Drop your scene photos here"), { target: { files: [new File(["different"], "FRAME-0.png", { type: "image/png" }), new File(["text"], "notes.txt", { type: "text/plain" })] } });
    expect(screen.getByText(/12 files selected/)).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("already selected");
    expect(screen.getByRole("alert")).toHaveTextContent("supported image");
    fireEvent.click(screen.getByRole("button", { name: "Remove frame-0.png" }));
    expect(screen.getByText(/11 files selected/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Create Task" }));
    expect(screen.getByText(/requires at least 12/)).toBeInTheDocument();
    expect(createTaskMock).not.toHaveBeenCalled();
  });

  it("bounds drag-and-drop collections at 48 and rejects empty files", () => {
    openCreate();
    const empty = new File([], "empty.png", { type: "image/png" });
    fireEvent.drop(screen.getByRole("button", { name: /Drop your scene photos here/ }), { dataTransfer: { files: [empty] } });
    expect(screen.getByRole("alert")).toHaveTextContent("empty files");
    uploadPhotos(49);
    expect(screen.getByText(/48 files selected/)).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("at most 48");
  });

  it("only submits files from the active pipeline", async () => {
    openCreate();
    uploadPhotos(12);
    fireEvent.click(screen.getByRole("button", { name: /Text.*Image to World.*Generative/i }));
    fireEvent.change(screen.getByLabelText("Text Prompt"), { target: { value: "sunlit room" } });
    fireEvent.click(screen.getByRole("button", { name: "Create Task" }));
    await waitFor(() => expect(createTaskMock).toHaveBeenCalledOnce());
    const body = createTaskMock.mock.calls[0][0] as FormData;
    expect(body.get("type")).toBe("worldgen");
    expect(body.getAll("images")).toHaveLength(0);
    expect(body.has("video")).toBe(false);
    expect(body.get("prompt")).toBe("sunlit room");
  });

  it("submits either a video or photos, keeping inactive media out of the upload", async () => {
    openCreate();
    uploadPhotos(12);
    fireEvent.click(screen.getByRole("button", { name: "Video capture" }));
    fireEvent.change(screen.getByLabelText("Drop a video capture here"), { target: { files: [new File(["video bytes"], "walk.mp4", { type: "video/mp4" })] } });
    fireEvent.click(screen.getByRole("button", { name: "Create Task" }));
    await waitFor(() => expect(createTaskMock).toHaveBeenCalledOnce());
    const body = createTaskMock.mock.calls[0][0] as FormData;
    expect(body.getAll("images")).toHaveLength(0);
    expect((body.get("video") as File).name).toBe("walk.mp4");
  });

  it("does not carry an invisible prompt from text generation into a panorama task", async () => {
    openCreate();
    fireEvent.click(screen.getByRole("button", { name: /Text.*Image to World.*Generative/i }));
    fireEvent.change(screen.getByLabelText("Text Prompt"), { target: { value: "old prompt" } });
    fireEvent.click(screen.getByRole("button", { name: /Panorama to World.*Generative/i }));
    fireEvent.change(screen.getByLabelText("Drop a 360° panorama here"), { target: { files: [new File(["panorama"], "pano.png", { type: "image/png" })] } });
    fireEvent.click(screen.getByRole("button", { name: "Create Task" }));
    await waitFor(() => expect(createTaskMock).toHaveBeenCalledOnce());
    const body = createTaskMock.mock.calls[0][0] as FormData;
    expect(body.has("prompt")).toBe(false);
    expect(JSON.parse(body.get("config") as string)).not.toHaveProperty("prompt");
  });

  it("rejects unsupported server settings and non-integral seeds before uploading", () => {
    openCreate();
    uploadPhotos(12);
    fireEvent.click(screen.getByRole("button", { name: /Show Advanced Config/ }));
    fireEvent.change(screen.getByLabelText("Advanced JSON config"), { target: { value: '{"model_path":"/tmp/model"}' } });
    fireEvent.click(screen.getByRole("button", { name: "Create Task" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Unsupported config fields: model_path");
    fireEvent.change(screen.getByLabelText("Advanced JSON config"), { target: { value: '{"seed":1.5}' } });
    fireEvent.click(screen.getByRole("button", { name: "Create Task" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Seed must be an integer");
    expect(createTaskMock).not.toHaveBeenCalled();
  });

  it("reuses the idempotency key for a retry and changes it when the request changes", async () => {
    createTaskMock.mockRejectedValue(new Error("Connection interrupted"));
    openCreate();
    uploadPhotos(12);
    fireEvent.click(screen.getByRole("button", { name: "Create Task" }));
    await screen.findByText("Connection interrupted");
    const firstKey = createTaskMock.mock.calls[0][1].idempotencyKey;
    fireEvent.click(screen.getByRole("button", { name: "Create Task" }));
    await waitFor(() => expect(createTaskMock).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.getByRole("button", { name: "Create Task" })).not.toBeDisabled());
    expect(createTaskMock.mock.calls[1][1].idempotencyKey).toBe(firstKey);
    fireEvent.change(screen.getByLabelText("Task Name"), { target: { value: "changed name" } });
    fireEvent.click(screen.getByRole("button", { name: "Create Task" }));
    await waitFor(() => expect(createTaskMock).toHaveBeenCalledTimes(3));
    expect(createTaskMock.mock.calls[2][1].idempotencyKey).not.toBe(firstKey);
  });

  it("prevents submission when the selected real pipeline lacks a ready runtime", () => {
    const refresh = vi.fn();
    systemMock.mockReturnValue({ system: { mode: "real", capabilities: [{ task_type: "worldrecon", ready: false, blockers: [{ code: "gpu", message: "No NVIDIA GPU is available." }] }] }, error: "", refresh });
    openCreate();
    expect(screen.getByRole("alert")).toHaveTextContent("No NVIDIA GPU is available.");
    expect(screen.getByRole("button", { name: "Create Task" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Check again" }));
    expect(refresh).toHaveBeenCalledOnce();
  });
});

describe("TaskCreate quality configuration", () => {
  it("defaults to high quality and merges the form prompt over advanced JSON", async () => {
    render(
      <MemoryRouter>
        <TaskCreate />
      </MemoryRouter>,
    );

    expect(screen.getByRole("button", { name: /Multi-View Reconstruction.*Production recommended/i })).toHaveClass("active");
    fireEvent.click(screen.getByRole("button", { name: /Text.*Image to World.*Generative/i }));
    expect(screen.getByText("Quality preset")).toBeInTheDocument();
    expect(screen.queryByText(/Fast —/i)).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Task Name"), { target: { value: "quality task" } });
    fireEvent.change(screen.getByLabelText("Text Prompt"), { target: { value: "form prompt" } });
    fireEvent.click(screen.getByRole("button", { name: /show advanced config/i }));
    fireEvent.change(screen.getByLabelText("Advanced JSON config"), {
      target: { value: '{"prompt":"advanced prompt","quality_preset":"fast","seed":17}' },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create Task" }));

    await waitFor(() => expect(createTaskMock).toHaveBeenCalledTimes(1));
    const formData = createTaskMock.mock.calls[0][0] as FormData;
    expect(JSON.parse(formData.get("config") as string)).toEqual({
      prompt: "form prompt",
      quality_preset: "high",
      recovery_policy: "auto",
      seed: 17,
      success_policy: "strict_free_view",
      input_lane: "generative_worldgen",
    });
    expect(screen.getByText(/Six trajectories are the training minimum/i)).toBeInTheDocument();
    expect(screen.getByText(/four ranked sources/i)).toBeInTheDocument();
    expect(screen.getByText(/without an image.*generates a source view first/i)).toBeInTheDocument();
  });

  it("requires twelve images when no production video is provided", async () => {
    render(<MemoryRouter><TaskCreate /></MemoryRouter>);
    fireEvent.change(screen.getByLabelText("Task Name"), { target: { value: "too-few" } });
    const input = document.querySelector('input[type="file"][multiple]') as HTMLInputElement;
    const files = Array.from({ length: 11 }, (_, index) => new File(["x"], `frame-${index}.png`, { type: "image/png" }));
    fireEvent.change(input, { target: { files } });
    fireEvent.click(screen.getByRole("button", { name: "Create Task" }));
    expect(await screen.findByText(/requires at least 12 multi-view images/i)).toBeInTheDocument();
    expect(createTaskMock).not.toHaveBeenCalled();
  });
});
