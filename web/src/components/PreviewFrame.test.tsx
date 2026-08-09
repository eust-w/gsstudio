// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import PreviewFrame from "./PreviewFrame";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("PreviewFrame", () => {
  it("enters and exits native fullscreen", async () => {
    const requestFullscreen = vi.fn().mockResolvedValue(undefined);
    const exitFullscreen = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(HTMLElement.prototype, "requestFullscreen", {
      configurable: true,
      value: requestFullscreen,
    });
    Object.defineProperty(document, "exitFullscreen", {
      configurable: true,
      value: exitFullscreen,
    });

    render(<PreviewFrame label="3D Model"><div>Preview</div></PreviewFrame>);
    fireEvent.click(screen.getByRole("button", { name: "Enter 3D Model fullscreen" }));
    await waitFor(() => expect(requestFullscreen).toHaveBeenCalledTimes(1));

    Object.defineProperty(document, "fullscreenElement", {
      configurable: true,
      value: screen.getByTestId("preview-frame"),
    });
    fireEvent(document, new Event("fullscreenchange"));
    fireEvent.click(screen.getByRole("button", { name: "Exit 3D Model fullscreen" }));

    expect(exitFullscreen).toHaveBeenCalledTimes(1);
  });

  it("falls back to an in-page fullscreen overlay and exits with Escape", async () => {
    Object.defineProperty(HTMLElement.prototype, "requestFullscreen", {
      configurable: true,
      value: undefined,
    });

    render(<PreviewFrame label="Video"><div>Preview</div></PreviewFrame>);
    fireEvent.click(screen.getByRole("button", { name: "Enter Video fullscreen" }));

    const frame = screen.getByTestId("preview-frame");
    expect(frame).toHaveClass("preview-frame-fallback");
    fireEvent.keyDown(document, { key: "Escape" });
    expect(frame).not.toHaveClass("preview-frame-fallback");
  });

  it("focuses a keyboard viewer after expansion", async () => {
    Object.defineProperty(HTMLElement.prototype, "requestFullscreen", {
      configurable: true,
      value: undefined,
    });

    render(
      <PreviewFrame label="3D Model">
        <div tabIndex={0} data-keyboard-viewer>Viewer</div>
      </PreviewFrame>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Enter 3D Model fullscreen" }));

    await waitFor(() => expect(screen.getByText("Viewer")).toHaveFocus());
  });
});
