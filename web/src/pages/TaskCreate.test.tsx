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

    expect(screen.getByLabelText("Quality preset")).toHaveValue("high");
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
      seed: 17,
    });
  });
});
