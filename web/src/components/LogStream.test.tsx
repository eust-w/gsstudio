// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import LogStream from "./LogStream";

vi.mock("../api", () => ({ getLogWebSocketUrl: (id: string) => `ws://localhost/api/tasks/${id}/logs` }));
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: ((event: { code: number }) => void) | null = null;
  onerror: (() => void) | null = null;
  close = vi.fn(() => this.onclose?.({ code: 1000 }));
  constructor(public url: string) { MockWebSocket.instances.push(this); }
  message(message: unknown) { this.onmessage?.({ data: JSON.stringify(message) }); }
}

beforeEach(() => {
  vi.useFakeTimers();
  MockWebSocket.instances = [];
  vi.stubGlobal("WebSocket", MockWebSocket);
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });

describe("LogStream", () => {
  it("cleans up the socket without scheduling a reconnect after unmount", () => {
    const { unmount } = render(<LogStream taskId="scene" />);
    const socket = MockWebSocket.instances[0];
    act(() => socket.onopen?.());
    expect(screen.getByRole("status")).toHaveTextContent("Live");
    unmount();
    expect(socket.close).toHaveBeenCalledOnce();
    act(() => vi.advanceTimersByTime(60000));
    expect(MockWebSocket.instances).toHaveLength(1);
  });

  it("resumes from the last cursor and stops reconnecting after a terminal frame", () => {
    render(<LogStream taskId="scene" />);
    const socket = MockWebSocket.instances[0];
    act(() => {
      socket.onopen?.();
      socket.message({ stage: "input_preflight", line: "Inputs validated", cursor: "cursor-1" });
      socket.onclose?.({ code: 1006 });
    });
    expect(screen.getByText("Inputs validated")).toBeInTheDocument();
    act(() => vi.advanceTimersByTime(1000));
    const resumed = MockWebSocket.instances[1];
    expect(resumed.url).toContain("cursor=cursor-1");
    act(() => {
      resumed.onopen?.();
      resumed.message({ stage: "input_preflight", line: "Inputs validated", cursor: "cursor-1" });
      resumed.message({ stage: "worldrecon", line: "Reconstruction completed", cursor: "cursor-2" });
      resumed.message({ type: "complete", status: "completed" });
    });
    expect(screen.getAllByText("Inputs validated")).toHaveLength(1);
    expect(screen.getByText("Reconstruction completed")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Finished");
    act(() => vi.advanceTimersByTime(60000));
    expect(MockWebSocket.instances).toHaveLength(2);
  });

  it("validates messages and filters by stage or content", () => {
    render(<LogStream taskId="scene" />);
    act(() => {
      const socket = MockWebSocket.instances[0];
      socket.message(null);
      socket.message({ line: 123 });
      socket.message({ stage: "worldrecon", line: "Camera positions recovered", cursor: "a" });
      socket.message({ stage: "navigation_stress", line: "Warning: boundary reached", cursor: "b" });
    });
    fireEvent.change(screen.getByLabelText("Filter logs"), { target: { value: "worldrecon" } });
    expect(screen.getByText("Camera positions recovered")).toBeInTheDocument();
    expect(screen.queryByText("Warning: boundary reached")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Filter logs"), { target: { value: "missing" } });
    expect(screen.getByText("No logs match your filter.")).toBeInTheDocument();
  });

  it("does not endlessly retry a task-not-found close", () => {
    render(<LogStream taskId="deleted" />);
    act(() => MockWebSocket.instances[0].onclose?.({ code: 4004 }));
    expect(screen.getByText("This task no longer exists.")).toBeInTheDocument();
    act(() => vi.advanceTimersByTime(60000));
    expect(MockWebSocket.instances).toHaveLength(1);
  });

  it("resets records and resume position when a different task is displayed", () => {
    const { rerender } = render(<LogStream taskId="first" />);
    act(() => MockWebSocket.instances[0].message({ line: "First task output", cursor: "first-cursor" }));
    rerender(<LogStream taskId="second" />);
    expect(screen.queryByText("First task output")).not.toBeInTheDocument();
    expect(MockWebSocket.instances[1].url).toBe("ws://localhost/api/tasks/second/logs");
    expect(MockWebSocket.instances[0].onclose).toBeNull();
  });
});
