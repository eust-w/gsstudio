// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import PanoViewer from "./PanoViewer";

const { dimensions, rendererSetSize, resizeObserverDisconnect, resizeObserverObserve, state } = vi.hoisted(() => ({
  dimensions: { width: 640, height: 400 },
  rendererSetSize: vi.fn(),
  resizeObserverDisconnect: vi.fn(),
  resizeObserverObserve: vi.fn(),
  state: { resizeCallback: undefined as ResizeObserverCallback | undefined },
}));

vi.mock("three", () => ({
  MathUtils: { degToRad: (value: number) => value * Math.PI / 180 },
  Mesh: class {},
  MeshBasicMaterial: class {},
  PerspectiveCamera: class {
    aspect = 1;
    position = { set: vi.fn() };
    lookAt = vi.fn();
    updateProjectionMatrix = vi.fn();
  },
  Scene: class { add = vi.fn(); },
  SphereGeometry: class { scale = vi.fn(); },
  SRGBColorSpace: "srgb",
  TextureLoader: class {
    load(_url: string, onLoad: (texture: { colorSpace: string }) => void) {
      onLoad({ colorSpace: "" });
    }
  },
  WebGLRenderer: class {
    domElement = document.createElement("canvas");
    dispose = vi.fn();
    render = vi.fn();
    setPixelRatio = vi.fn();
    setSize = rendererSetSize;
  },
}));

beforeEach(() => {
  dimensions.width = 640;
  dimensions.height = 400;
  Object.defineProperty(HTMLElement.prototype, "clientWidth", {
    configurable: true,
    get: () => dimensions.width,
  });
  Object.defineProperty(HTMLElement.prototype, "clientHeight", {
    configurable: true,
    get: () => dimensions.height,
  });
  globalThis.requestAnimationFrame = vi.fn().mockReturnValue(1);
  globalThis.cancelAnimationFrame = vi.fn();
  globalThis.ResizeObserver = class ResizeObserverMock {
    constructor(callback: ResizeObserverCallback) {
      state.resizeCallback = callback;
    }
    observe = resizeObserverObserve;
    disconnect = resizeObserverDisconnect;
    unobserve = vi.fn();
  };
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.restoreAllMocks();
  state.resizeCallback = undefined;
});

describe("PanoViewer", () => {
  it("resizes its renderer whenever its container changes size", async () => {
    const { container } = render(<PanoViewer url="/panorama.png" />);
    const canvasContainer = container.querySelector(".pano-canvas") as HTMLElement;

    await waitFor(() => expect(resizeObserverObserve).toHaveBeenCalledWith(canvasContainer));
    dimensions.width = 1200;
    dimensions.height = 800;
    state.resizeCallback?.([], {} as ResizeObserver);

    expect(rendererSetSize).toHaveBeenLastCalledWith(1200, 800);
  });
});
