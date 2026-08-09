import { useEffect, useRef, useState } from "react";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { isEditableTarget, moveCamera } from "./splatNavigation";

interface SplatViewerProps {
  url: string;
  onLoad?: () => void;
  onError?: (err: Error) => void;
}

export default function SplatViewer({ url, onLoad, onError }: SplatViewerProps) {
  const viewerRef = useRef<HTMLDivElement>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const pressedKeysRef = useRef(new Set<string>());
  const moveSpeedRef = useRef(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);
  const [loadedBytes, setLoadedBytes] = useState(0);
  const [keyboardActive, setKeyboardActive] = useState(false);

  useEffect(() => {
    const container = canvasContainerRef.current;
    const viewer = viewerRef.current;
    if (!container || !viewer || !url) return;

    let disposed = false;
    setLoading(true);
    setError("");
    setProgress(0);
    setLoadedBytes(0);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 10000);
    camera.position.set(0, 0, 5);
    camera.up.set(0, -1, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const spark = new SparkRenderer({
      renderer,
      lodSplatScale: 1,
      pagedExtSplats: true,
    });
    scene.add(spark);

    const handleError = (cause: unknown) => {
      if (disposed) return;
      const nextError = cause instanceof Error ? cause : new Error(String(cause));
      setError(nextError.message);
      setLoading(false);
      onError?.(nextError);
    };

    const splat = new SplatMesh({
      url,
      lod: true,
      onProgress: (event) => {
        if (disposed) return;
        setLoadedBytes(event.loaded);
        if (event.lengthComputable && event.total > 0) {
          setProgress(Math.min(100, (event.loaded / event.total) * 100));
        }
      },
      onLoad: (mesh) => {
        if (disposed) return;
        const bounds = mesh.getBoundingBox();
        const center = bounds.getCenter(new THREE.Vector3());
        const diagonal = Math.max(bounds.getSize(new THREE.Vector3()).length(), 1);
        moveSpeedRef.current = Math.max(diagonal * 0.3, 0.1);
        controls.target.copy(center);
        camera.position.copy(center).add(new THREE.Vector3(0, 0, diagonal * 1.2));
        camera.near = Math.max(diagonal / 10000, 0.001);
        camera.far = Math.max(diagonal * 100, 1000);
        camera.updateProjectionMatrix();
        controls.update();
        setProgress(100);
        setLoading(false);
        onLoad?.();
      },
    });
    scene.add(splat);
    void splat.initialized.catch(handleError);

    const resize = () => {
      const width = Math.max(container.clientWidth, 1);
      const height = Math.max(container.clientHeight, 1);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    resize();

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);
    const navigationCodes = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "ShiftLeft", "ShiftRight"]);
    const keyboardControlsActive = () => (
      document.activeElement === viewer || Boolean(document.fullscreenElement?.contains(viewer))
    );
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!navigationCodes.has(event.code) || !keyboardControlsActive() || isEditableTarget(event.target)) return;
      event.preventDefault();
      pressedKeysRef.current.add(event.code);
      setKeyboardActive(true);
    };
    const handleKeyUp = (event: KeyboardEvent) => {
      if (!navigationCodes.has(event.code)) return;
      pressedKeysRef.current.delete(event.code);
    };
    const clearPressedKeys = () => pressedKeysRef.current.clear();
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", clearPressedKeys);
    document.addEventListener("visibilitychange", clearPressedKeys);

    let previousFrameTime = 0;
    renderer.setAnimationLoop((frameTime) => {
      const deltaSeconds = previousFrameTime === 0
        ? 0
        : Math.min((frameTime - previousFrameTime) / 1000, 0.1);
      previousFrameTime = frameTime;
      moveCamera(camera, controls.target, pressedKeysRef.current, deltaSeconds, moveSpeedRef.current);
      controls.update();
      renderer.render(scene, camera);
    });

    return () => {
      disposed = true;
      pressedKeysRef.current.clear();
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", clearPressedKeys);
      document.removeEventListener("visibilitychange", clearPressedKeys);
      resizeObserver.disconnect();
      renderer.setAnimationLoop(null);
      controls.dispose();
      splat.dispose();
      spark.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [url, onLoad, onError]);

  const loadedMegabytes = loadedBytes / 1024 / 1024;

  return (
    <div
      ref={viewerRef}
      className={`splat-viewer ${keyboardActive ? "keyboard-active" : ""}`}
      tabIndex={0}
      data-keyboard-viewer
      aria-label="Interactive 3D model viewer. Use WASD to move and Shift to move faster."
      onFocus={() => setKeyboardActive(true)}
      onBlur={() => {
        pressedKeysRef.current.clear();
        setKeyboardActive(false);
      }}
      onPointerDown={() => viewerRef.current?.focus({ preventScroll: true })}
    >
      <div ref={canvasContainerRef} className="splat-canvas" />
      {loading && (
        <div className="viewer-loading viewer-loading-overlay">
          <div className="loading-spinner" />
          <span>
            Streaming 3D model… {progress > 0 ? `${Math.round(progress)}%` : ""}
          </span>
          {loadedBytes > 0 && <span className="viewer-loading-bytes">{loadedMegabytes.toFixed(1)} MB received</span>}
        </div>
      )}
      {error && (
        <div className="viewer-error viewer-loading-overlay">
          <p>Failed to load 3D model</p>
          <p className="error-detail">{error}</p>
          <p className="error-hint">Check that the model artifact is available and supported.</p>
        </div>
      )}
      {!loading && !error && (
        <div className="splat-hint">Click to control | WASD move | Shift boost | Drag to look | Scroll to zoom</div>
      )}
    </div>
  );
}
