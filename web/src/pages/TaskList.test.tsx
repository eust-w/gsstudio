// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import TaskList from "./TaskList";

const { cancelTaskMock, deleteTaskMock, listTasksMock, retryTaskMock } = vi.hoisted(() => ({
  cancelTaskMock: vi.fn().mockResolvedValue(undefined),
  deleteTaskMock: vi.fn().mockResolvedValue(undefined),
  listTasksMock: vi.fn(),
  retryTaskMock: vi.fn().mockResolvedValue(undefined),
}));

const tasks = [
  { id: "pending-1", name: "Pending task", type: "worldgen", status: "pending", progress: 0, created_at: "2026-08-09T00:00:00Z" },
  { id: "running-1", name: "Running task", type: "worldgen", status: "running", progress: 0.5, created_at: "2026-08-09T00:01:00Z" },
  { id: "completed-1", name: "Completed task", type: "worldrecon", status: "completed", progress: 1, created_at: "2026-08-09T00:02:00Z" },
  { id: "failed-1", name: "Failed task", type: "worldgen_from_pano", status: "failed", progress: 0.4, created_at: "2026-08-09T00:03:00Z" },
  { id: "cancelled-1", name: "Cancelled task", type: "worldrecon", status: "cancelled", progress: 0.2, created_at: "2026-08-09T00:04:00Z" },
] as const;

vi.mock("../api", () => ({
  cancelTask: cancelTaskMock,
  deleteTask: deleteTaskMock,
  listTasks: listTasksMock,
  retryTask: retryTaskMock,
}));

vi.mock("../components/StatusBadge", () => ({
  default: ({ status }: { status: string }) => <span>{status}</span>,
}));

function renderList() {
  listTasksMock.mockResolvedValue({ tasks, total: tasks.length });
  return render(
    <MemoryRouter>
      <TaskList />
    </MemoryRouter>,
  );
}

async function selectVisibleTasks() {
  await screen.findByRole("heading", { name: "Your scenes" });
  fireEvent.click(screen.getByRole("checkbox", { name: "Select visible tasks" }));
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.restoreAllMocks();
});

describe("TaskList batch actions", () => {
  it("selects visible tasks and shows status-aware action counts", async () => {
    renderList();

    await selectVisibleTasks();

    expect(screen.getByText("5 selected")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel active (2)" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Retry failed (2)" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Delete finished (3)" })).toBeEnabled();
  });

  it("cancels only selected pending and running tasks", async () => {
    renderList();
    await selectVisibleTasks();

    fireEvent.click(screen.getByRole("button", { name: "Cancel active (2)" }));

    await waitFor(() => {
      expect(cancelTaskMock).toHaveBeenCalledTimes(2);
      expect(cancelTaskMock).toHaveBeenCalledWith("pending-1");
      expect(cancelTaskMock).toHaveBeenCalledWith("running-1");
    });
  });

  it("deletes only terminal tasks after confirmation", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    renderList();
    await selectVisibleTasks();

    fireEvent.click(screen.getByRole("button", { name: "Delete finished (3)" }));

    await waitFor(() => {
      expect(deleteTaskMock).toHaveBeenCalledTimes(3);
      expect(deleteTaskMock).toHaveBeenCalledWith("completed-1");
      expect(deleteTaskMock).toHaveBeenCalledWith("failed-1");
      expect(deleteTaskMock).toHaveBeenCalledWith("cancelled-1");
    });
  });

  it("retries only failed and cancelled tasks", async () => {
    renderList();
    await selectVisibleTasks();

    fireEvent.click(screen.getByRole("button", { name: "Retry failed (2)" }));

    await waitFor(() => {
      expect(retryTaskMock).toHaveBeenCalledTimes(2);
      expect(retryTaskMock).toHaveBeenCalledWith("failed-1");
      expect(retryTaskMock).toHaveBeenCalledWith("cancelled-1");
    });
  });

  it("selecting a card checkbox does not navigate to its detail page", async () => {
    renderList();
    await screen.findByRole("heading", { name: "Your scenes" });

    fireEvent.click(screen.getByRole("checkbox", { name: "Select task Pending task" }));

    expect(screen.getByRole("heading", { name: "Your scenes" })).toBeInTheDocument();
    expect(screen.getByText("1 selected")).toBeInTheDocument();
  });
});

describe("TaskList server-backed discovery", () => {
  it("sends trimmed search text, pipeline and status filters to the server and resets pagination", async () => {
    listTasksMock.mockImplementation(async (params: { q?: string }) => ({
      tasks: params.q ? [{ ...tasks[2], id: "remote-match", name: "Office on a later page" }] : tasks,
      total: params.q ? 14 : 25,
    }));
    render(<MemoryRouter><TaskList /></MemoryRouter>);
    await screen.findByRole("heading", { name: "Completed task" });
    expect(listTasksMock).toHaveBeenLastCalledWith(expect.objectContaining({ limit: 12, offset: 0, q: undefined }));

    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() => expect(listTasksMock).toHaveBeenLastCalledWith(expect.objectContaining({ limit: 12, offset: 12 })));
    expect(await screen.findByText("Page 2 of 3")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox", { name: "Search scenes" }), { target: { value: "  office  " } });
    await waitFor(() => expect(listTasksMock).toHaveBeenLastCalledWith(expect.objectContaining({ q: "office", limit: 12, offset: 0 })));
    expect(await screen.findByRole("heading", { name: "Office on a later page" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Completed task" })).not.toBeInTheDocument();
    expect(screen.getByText("Page 1 of 2")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox", { name: "Filter by pipeline" }), { target: { value: "worldrecon" } });
    await waitFor(() => expect(listTasksMock).toHaveBeenLastCalledWith(expect.objectContaining({ q: "office", type: "worldrecon", offset: 0 })));
    fireEvent.click(screen.getByRole("button", { name: "Completed" }));
    await waitFor(() => expect(listTasksMock).toHaveBeenLastCalledWith(expect.objectContaining({ q: "office", type: "worldrecon", status: "completed", offset: 0 })));
  });

  it("shows an API failure without claiming that the workspace has no scenes", async () => {
    listTasksMock.mockRejectedValue(new Error("The task API is unavailable."));
    render(<MemoryRouter><TaskList /></MemoryRouter>);
    expect(await screen.findByRole("alert")).toHaveTextContent("The task API is unavailable.");
    expect(screen.getByRole("heading", { name: "Your scenes are temporarily unavailable" })).toBeInTheDocument();
    expect(screen.queryByText("A new perspective starts here")).not.toBeInTheDocument();
    expect(screen.queryByText("No scenes yet")).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Create your first scene" })).not.toBeInTheDocument();

    listTasksMock.mockResolvedValue({ tasks: [tasks[2]], total: 1 });
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("heading", { name: "Completed task" })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
