import { useEffect, useRef, useState } from "react";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

interface SplatViewerProps {
  url: string;
  onLoad?: () => void;
  onError?: (err: Error) => void;
}

export default function SplatViewer({ url, onLoad, onError }: SplatViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);
  const [loadedBytes, setLoadedBytes] = useState(0);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !url) return;

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
    renderer.setAnimationLoop(() => {
      controls.update();
      renderer.render(scene, camera);
    });

    return () => {
      disposed = true;
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
    <div className="splat-viewer">
      <div ref={containerRef} className="splat-canvas" />
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
        <div className="splat-hint">Left-drag to rotate | Right-drag to pan | Scroll to zoom</div>
      )}
    </div>
  );
}
