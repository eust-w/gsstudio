import { useEffect, useRef, useState } from "react";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { PointerLockControls } from "three/examples/jsm/controls/PointerLockControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { MeshBVH, acceleratedRaycast } from "three-mesh-bvh";
import { areSplatWorkersIdle, disposeSplatResourcesWhenIdle } from "./splatCleanup";
import { isEditableTarget } from "./splatNavigation";
import {
  clampToNavigationBounds,
  hasNavigableBounds,
  initialCameraPose,
  moveOnGround,
  parseViewerMetadata,
  requestPointerLockSafely,
  supportsPointerLock,
  ViewerMetadata,
} from "./splatFirstPerson";

THREE.Mesh.prototype.raycast = acceleratedRaycast;

interface SplatViewerProps {
  url: string;
  metadataUrl?: string;
  collisionMeshUrl?: string;
  navigationEnabled?: boolean;
  onLoad?: () => void;
  onError?: (err: Error) => void;
}

type NavigationStatus = "loading" | "ready" | "unavailable" | "error";

function safeFallbackPose(mesh: SplatMesh, camera: THREE.PerspectiveCamera): () => void {
  const bounds = mesh.getBoundingBox();
  const center = bounds.getCenter(new THREE.Vector3());
  const diagonal = Math.max(bounds.getSize(new THREE.Vector3()).length(), 1);
  const lookAt = center.clone().add(new THREE.Vector3(0, 0, -1));
  return () => {
    camera.fov = 70;
    camera.up.set(0, 1, 0);
    // Keeping a legacy task at the point-cloud center is intentionally safer
    // than the former bounding-box exterior guess for indoor scenes.
    camera.position.copy(center);
    camera.near = Math.max(diagonal / 10000, 0.001);
    camera.far = Math.max(diagonal * 100, 1000);
    camera.lookAt(lookAt);
    camera.updateProjectionMatrix();
  };
}

function resolveCollision(
  current: THREE.Vector3,
  candidate: THREE.Vector3,
  metadata: ViewerMetadata,
  collisionMesh: THREE.Mesh | null,
  raycaster: THREE.Raycaster,
): THREE.Vector3 {
  const bounded = clampToNavigationBounds(current, candidate, metadata);
  if (!collisionMesh) return bounded;

  const up = metadata.upDirection;
  const horizontalMove = bounded.clone().sub(current).projectOnPlane(up);
  if (horizontalMove.lengthSq() > 0) {
    const moveDistance = horizontalMove.length();
    raycaster.set(current.clone().addScaledVector(up, Math.max(metadata.humanScale * 0.05, 0.02)), horizontalMove.normalize());
    raycaster.near = 0.02;
    raycaster.far = moveDistance + Math.max(metadata.humanScale * 0.2, 0.05);
    const wallHit = raycaster.intersectObject(collisionMesh, false)[0];
    if (wallHit) return current.clone();
  }

  return bounded;
}

export default function SplatViewer({
  url,
  metadataUrl,
  collisionMeshUrl,
  navigationEnabled = false,
  onLoad,
  onError,
}: SplatViewerProps) {
  const viewerRef = useRef<HTMLDivElement>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const pressedKeysRef = useRef(new Set<string>());
  const controlsRef = useRef<PointerLockControls | null>(null);
  const resetViewRef = useRef<() => void>(() => {});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);
  const [loadedBytes, setLoadedBytes] = useState(0);
  const [keyboardActive, setKeyboardActive] = useState(false);
  const [pointerLockAvailable, setPointerLockAvailable] = useState(true);
  const [navigationStatus, setNavigationStatus] = useState<NavigationStatus>(
    navigationEnabled && collisionMeshUrl ? "loading" : "unavailable",
  );
  const [metadataMode, setMetadataMode] = useState<"loading" | "official" | "fallback">(
    metadataUrl ? "loading" : "fallback",
  );

  useEffect(() => {
    const container = canvasContainerRef.current;
    const viewer = viewerRef.current;
    if (!container || !viewer || !url) return;

    let disposed = false;
    let metadata: ViewerMetadata | null = null;
    let fallbackReset: (() => void) | null = null;
    let collisionMesh: THREE.Mesh | null = null;
    let collisionReady = false;
    let navigationReady = false;
    let splatReady = false;
    const raycaster = new THREE.Raycaster();
    (raycaster as THREE.Raycaster & { firstHitOnly?: boolean }).firstHitOnly = true;

    setLoading(true);
    setError("");
    setProgress(0);
    setLoadedBytes(0);
    setNavigationStatus(navigationEnabled && collisionMeshUrl ? "loading" : "unavailable");
    setMetadataMode(metadataUrl ? "loading" : "fallback");

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(70, 1, 0.01, 10000);
    camera.position.set(0, 0, 0);
    camera.up.set(0, 1, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new PointerLockControls(camera, renderer.domElement);
    controlsRef.current = controls;
    const pointerLockSupported = supportsPointerLock(
      renderer.domElement,
      renderer.domElement.ownerDocument,
      window,
    );
    setPointerLockAvailable(pointerLockSupported);
    const onPointerLock = () => setKeyboardActive(true);
    const onPointerUnlock = () => {
      pressedKeysRef.current.clear();
      setKeyboardActive(false);
    };
    // PointerLockControls logs an error on `pointerlockerror`.  Intercept it
    // in capture phase so embedded browsers degrade to focused WASD controls
    // without a noisy console error, while normal browsers still lock the
    // cursor when the capability is present.
    const onPointerLockError = (event: Event) => {
      event.preventDefault();
      event.stopImmediatePropagation();
      if (!disposed) setPointerLockAvailable(false);
    };
    controls.addEventListener("lock", onPointerLock);
    controls.addEventListener("unlock", onPointerUnlock);
    renderer.domElement.ownerDocument.addEventListener("pointerlockerror", onPointerLockError, true);

    const spark = new SparkRenderer({
      renderer,
      lodSplatScale: 1,
      pagedExtSplats: true,
    });
    scene.add(spark);

    const applyInitialPose = () => {
      if (!splatReady) return;
      if (metadata) {
        const pose = initialCameraPose(metadata);
        camera.position.copy(pose.position);
        camera.up.copy(pose.up);
        camera.fov = pose.fov;
        camera.lookAt(pose.lookAt);
        camera.updateProjectionMatrix();
      } else {
        fallbackReset?.();
      }
    };
    resetViewRef.current = applyInitialPose;

    const updateNavigationReadiness = () => {
      navigationReady = Boolean(
        navigationEnabled
        && collisionReady
        && metadata
        && hasNavigableBounds(metadata),
      );
      if (!navigationEnabled || !collisionMeshUrl) {
        setNavigationStatus("unavailable");
      } else if (navigationReady) {
        setNavigationStatus("ready");
      } else if (metadata && !hasNavigableBounds(metadata)) {
        setNavigationStatus("unavailable");
      } else {
        setNavigationStatus("loading");
      }
    };

    const handleError = (cause: unknown) => {
      if (disposed) return;
      const nextError = cause instanceof Error ? cause : new Error(String(cause));
      setError(nextError.message);
      setLoading(false);
      onError?.(nextError);
    };

    if (metadataUrl) {
      void fetch(metadataUrl, { credentials: "same-origin" })
        .then((response) => {
          if (!response.ok) throw new Error(`metadata request failed (${response.status})`);
          return response.json();
        })
        .then((payload) => {
          if (disposed) return;
          metadata = parseViewerMetadata(payload);
          setMetadataMode(metadata ? "official" : "fallback");
          updateNavigationReadiness();
          applyInitialPose();
        })
        .catch(() => {
          if (!disposed) {
            setMetadataMode("fallback");
            setNavigationStatus(navigationEnabled ? "error" : "unavailable");
          }
        });
    }

    if (collisionMeshUrl) {
      new PLYLoader().load(
        collisionMeshUrl,
        (geometry) => {
          if (disposed) {
            geometry.dispose();
            return;
          }
          geometry.computeVertexNormals();
          geometry.boundsTree = new MeshBVH(geometry);
          collisionMesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
          collisionMesh.updateMatrixWorld(true);
          collisionReady = true;
          updateNavigationReadiness();
        },
        undefined,
        () => {
          if (!disposed) setNavigationStatus(navigationEnabled ? "error" : "unavailable");
        },
      );
    }

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
        fallbackReset = safeFallbackPose(mesh, camera);
        splatReady = true;
        applyInitialPose();
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
      controls.isLocked || document.activeElement === viewer || Boolean(document.fullscreenElement?.contains(viewer))
    );
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!keyboardControlsActive() || isEditableTarget(event.target)) return;
      if (event.code === "KeyR") {
        event.preventDefault();
        resetViewRef.current();
        return;
      }
      if (!navigationCodes.has(event.code) || !navigationReady) return;
      event.preventDefault();
      pressedKeysRef.current.add(event.code);
      setKeyboardActive(true);
    };
    const handleKeyUp = (event: KeyboardEvent) => {
      if (navigationCodes.has(event.code)) pressedKeysRef.current.delete(event.code);
    };
    const clearPressedKeys = () => pressedKeysRef.current.clear();
    const activatePointerLock = () => {
      viewer.focus({ preventScroll: true });
      if (!controls.isLocked && pointerLockSupported && renderer.domElement.isConnected) {
        requestPointerLockSafely(renderer.domElement, () => setPointerLockAvailable(false));
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", clearPressedKeys);
    document.addEventListener("visibilitychange", clearPressedKeys);
    viewer.addEventListener("pointerdown", activatePointerLock);

    let previousFrameTime = 0;
    renderer.setAnimationLoop((frameTime) => {
      const deltaSeconds = previousFrameTime === 0
        ? 0
        : Math.min((frameTime - previousFrameTime) / 1000, 0.1);
      previousFrameTime = frameTime;
      if (navigationReady && metadata && pressedKeysRef.current.size > 0) {
        const candidate = moveOnGround(
          camera.position,
          camera.getWorldDirection(new THREE.Vector3()),
          metadata.upDirection,
          pressedKeysRef.current,
          deltaSeconds,
          metadata.humanScale,
        );
        camera.position.copy(resolveCollision(camera.position, candidate, metadata, collisionMesh, raycaster));
      }
      renderer.render(scene, camera);
    });

    return () => {
      disposed = true;
      pressedKeysRef.current.clear();
      resetViewRef.current = () => {};
      controlsRef.current = null;
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", clearPressedKeys);
      document.removeEventListener("visibilitychange", clearPressedKeys);
      viewer.removeEventListener("pointerdown", activatePointerLock);
      resizeObserver.disconnect();
      renderer.setAnimationLoop(null);
      controls.removeEventListener("lock", onPointerLock);
      controls.removeEventListener("unlock", onPointerUnlock);
      renderer.domElement.ownerDocument.removeEventListener("pointerlockerror", onPointerLockError, true);
      controls.disconnect();
      collisionMesh?.geometry.dispose();
      (collisionMesh?.material as THREE.Material | undefined)?.dispose();
      spark.autoUpdate = false;
      if (spark.updateTimeoutId !== -1) {
        window.clearTimeout(spark.updateTimeoutId);
        spark.updateTimeoutId = -1;
      }
      if (spark.sortTimeoutId !== -1) {
        window.clearTimeout(spark.sortTimeoutId);
        spark.sortTimeoutId = -1;
      }
      spark.sortDirty = false;
      spark.enableDriveLod = false;
      spark.lodDirty = false;
      spark.lodInitQueue = [];
      spark.lodUpdates = [];
      disposeSplatResourcesWhenIdle(
        () => !spark.sorting && areSplatWorkersIdle(spark.sortWorker, spark.lodWorker),
        () => {
          controls.dispose();
          splat.dispose();
          spark.dispose();
          renderer.dispose();
          renderer.domElement.remove();
        },
      );
    };
  }, [url, metadataUrl, collisionMeshUrl, navigationEnabled, onLoad, onError]);

  const loadedMegabytes = loadedBytes / 1024 / 1024;
  const navigationHint = navigationStatus === "ready"
    ? pointerLockAvailable
      ? "Click to lock mouse | WASD move | Shift boost | R reset view"
      : "Click viewer to focus | WASD move | Shift boost | R reset view"
    : navigationStatus === "loading"
      ? "Preparing collision mesh… WASD is temporarily disabled"
      : navigationStatus === "error"
        ? "Collision mesh failed to load — WASD is disabled"
        : navigationEnabled
          ? "Validated navigation bounds are unavailable — WASD is disabled"
          : "Free-view validation is unavailable — rotation and R reset remain enabled";

  return (
    <div
      ref={viewerRef}
      className={`splat-viewer ${keyboardActive ? "keyboard-active" : ""}`}
      tabIndex={0}
      data-keyboard-viewer
      aria-label={navigationEnabled
        ? "Interactive first-person 3D model viewer. Use WASD to move, Shift to move faster, and R to reset the official view."
        : "Interactive 3D model viewer. Rotate the view and press R to reset the official view; WASD is disabled."}
      onFocus={() => setKeyboardActive(true)}
      onBlur={() => {
        pressedKeysRef.current.clear();
        setKeyboardActive(false);
      }}
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
        <>
          <div className="splat-hint">{navigationHint} · {metadataMode === "official" ? "official scene pose" : "safe legacy pose"}</div>
          <button
            type="button"
            className="splat-reset-button"
            onPointerDown={(event) => event.stopPropagation()}
            onClick={() => resetViewRef.current()}
          >
            Reset view (R)
          </button>
        </>
      )}
    </div>
  );
}
