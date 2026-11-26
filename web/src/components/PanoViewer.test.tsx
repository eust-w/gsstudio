// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import PanoViewer from "./PanoViewer";

const { dimensions, rendererSetSize, rendererDispose, geometryDispose, materialDispose, resizeObserverDisconnect, resizeObserverObserve, state } = vi.hoisted(() => ({
  dimensions: { width: 640, height: 400 },
  rendererSetSize: vi.fn(), rendererDispose: vi.fn(), geometryDispose: vi.fn(), materialDispose: vi.fn(), resizeObserverDisconnect: vi.fn(), resizeObserverObserve: vi.fn(),
  state: {
    resizeCallback: undefined as ResizeObserverCallback | undefined,
    deferred: false,
    loads: [] as Array<{ texture: { colorSpace: string; dispose: ReturnType<typeof vi.fn> }; finish: () => void; fail: () => void }>,
    cameras: [] as Array<{ fov: number; aspect: number }>,
  },
}));
vi.mock("three", () => ({
  MathUtils: { degToRad: (value: number) => value * Math.PI / 180 }, Mesh: class {},
  MeshBasicMaterial: class { dispose = materialDispose; },
  PerspectiveCamera: class {
    aspect = 1;
    position = { set: vi.fn() }; lookAt = vi.fn(); updateProjectionMatrix = vi.fn();
    constructor(public fov: number) { state.cameras.push(this); }
  },
  Scene: class { add = vi.fn(); },
  SphereGeometry: class { scale = vi.fn(); dispose = geometryDispose; },
  SRGBColorSpace: "srgb",
  TextureLoader: class {
    load(_url: string, onLoad: (texture: { colorSpace: string; dispose: ReturnType<typeof vi.fn> }) => void, _progress: unknown, onError: () => void) {
      const texture = { colorSpace: "", dispose: vi.fn() };
      state.loads.push({ texture, finish: () => onLoad(texture), fail: onError });
      if (!state.deferred) onLoad(texture);
      return texture;
    }
  },
  WebGLRenderer: class {
    domElement = document.createElement("canvas"); dispose = rendererDispose; render = vi.fn(); setPixelRatio = vi.fn(); setSize = rendererSetSize;
  },
}));

beforeEach(() => {
  dimensions.width = 640; dimensions.height = 400;
  state.deferred = false; state.loads = []; state.cameras = [];
  Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, get: () => dimensions.width });
  Object.defineProperty(HTMLElement.prototype, "clientHeight", { configurable: true, get: () => dimensions.height });
  vi.stubGlobal("requestAnimationFrame", vi.fn().mockReturnValue(1));
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  vi.stubGlobal("ResizeObserver", class {
    constructor(callback: ResizeObserverCallback) { state.resizeCallback = callback; }
    observe = resizeObserverObserve; disconnect = resizeObserverDisconnect; unobserve = vi.fn();
  });
});
afterEach(() => { cleanup(); vi.clearAllMocks(); vi.restoreAllMocks(); vi.unstubAllGlobals(); state.resizeCallback = undefined; });

describe("PanoViewer", () => {
  it("resizes its renderer whenever its container changes size", async () => {
    const { container } = render(<PanoViewer url="/panorama.png" />);
    const canvasContainer = container.querySelector(".pano-canvas") as HTMLElement;
    await waitFor(() => expect(resizeObserverObserve).toHaveBeenCalledWith(canvasContainer));
    dimensions.width = 1200; dimensions.height = 800;
    state.resizeCallback?.([], {} as ResizeObserver);
    expect(rendererSetSize).toHaveBeenLastCalledWith(1200, 800);
  });

  it("disposes the loaded texture, material, geometry and renderer on unmount", () => {
    const { unmount, container } = render(<PanoViewer url="/panorama.png" />);
    expect(container.querySelector("canvas")).toBeInTheDocument();
    unmount();
    expect(state.loads[0].texture.dispose).toHaveBeenCalledOnce();
    expect(geometryDispose).toHaveBeenCalledOnce();
    expect(materialDispose).toHaveBeenCalledOnce();
    expect(rendererDispose).toHaveBeenCalledOnce();
    expect(resizeObserverDisconnect).toHaveBeenCalledOnce();
  });

  it("does not create geometry or leak a texture when loading finishes after unmount", () => {
    state.deferred = true;
    const { unmount } = render(<PanoViewer url="/slow.png" />);
    const load = state.loads[0];
    unmount();
    act(() => load.finish());
    expect(load.texture.dispose).toHaveBeenCalledOnce();
    expect(geometryDispose).not.toHaveBeenCalled();
    expect(materialDispose).not.toHaveBeenCalled();
    expect(rendererDispose).toHaveBeenCalledOnce();
  });

  it("resets error and loading state when a new panorama URL is supplied", () => {
    state.deferred = true;
    const { rerender } = render(<PanoViewer url="/failed.png" />);
    act(() => state.loads[0].fail());
    expect(screen.getByRole("alert")).toHaveTextContent("Failed to load panorama image");
    rerender(<PanoViewer url="/replacement.png" />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Loading panorama");
    act(() => state.loads[1].finish());
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(state.loads[0].texture.dispose).toHaveBeenCalledOnce();
  });

  it("releases a partially initialized renderer after setup fails", () => {
    rendererSetSize.mockImplementationOnce(() => { throw new Error("context lost"); });
    const { container } = render(<PanoViewer url="/panorama.png" />);
    expect(screen.getByRole("alert")).toHaveTextContent("Failed to initialize 3D viewer");
    expect(rendererDispose).toHaveBeenCalledOnce();
    expect(container.querySelector("canvas")).not.toBeInTheDocument();
  });

  it("clamps zero-size viewports and supports bounded zoom and reset", () => {
    dimensions.width = 0; dimensions.height = 0;
    render(<PanoViewer url="/panorama.png" />);
    expect(rendererSetSize).toHaveBeenLastCalledWith(1, 1);
    expect(Number.isFinite(state.cameras[0].aspect)).toBe(true);
    const viewport = screen.getByLabelText(/Panorama viewport/);
    fireEvent.wheel(viewport, { deltaY: -10000 });
    expect(state.cameras[0].fov).toBe(30);
    expect(screen.getByRole("button", { name: "Zoom in panorama" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Reset view" }));
    expect(state.cameras[0].fov).toBe(75);
    fireEvent.keyDown(viewport, { key: "+" });
    expect(state.cameras[0].fov).toBe(70);
  });
});
