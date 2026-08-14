// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import TaskCreate from "./TaskCreate";

const { createTaskMock } = vi.hoisted(() => ({
  createTaskMock: vi.fn().mockResolvedValue({ task_id: "new-task" }),
}));

vi.mock("../api", () => ({
  createTask: createTaskMock,
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
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
