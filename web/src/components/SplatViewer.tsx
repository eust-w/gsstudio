import { useEffect, useRef, useState } from "react";

interface SplatViewerProps {
  url: string;
  onLoad?: () => void;
  onError?: (err: Error) => void;
}

export default function SplatViewer({ url, onLoad, onError }: SplatViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!containerRef.current || !url) return;

    let disposed = false;

    const initViewer = async () => {
      try {
        const GaussianSplats3D = await import("@mkkellogg/gaussian-splats-3d");
        const THREE = await import("three");

        if (disposed || !containerRef.current) return;

        const viewer = new GaussianSplats3D.Viewer({
          cameraUp: [0, -1, 0],
          initialCameraPosition: [0, 0, 5],
          initialCameraLookAt: [0, 0, 0],
          rootElement: containerRef.current,
          selfDrivenMode: true,
          sharedMemoryForWorkers: false,
        });

        viewerRef.current = viewer;

        const isSpz = url.endsWith(".spz");
        const format = isSpz
          ? GaussianSplats3D.SceneFormat.Spz
          : GaussianSplats3D.SceneFormat.Ply;

        await viewer.addSplatScene(url, {
          showLoadingUI: true,
          format,
          onProgress: (pct: number) => {
            if (!disposed) setProgress(pct);
          },
        });
        viewer.start();

        if (!disposed) {
          setLoading(false);
          onLoad?.();
        }
      } catch (e) {
        if (!disposed) {
          const err = e instanceof Error ? e : new Error(String(e));
          setError(err.message);
          setLoading(false);
          onError?.(err);
        }
      }
    };

    initViewer();

    return () => {
      disposed = true;
      if (viewerRef.current) {
        try {
          (viewerRef.current as { dispose: () => void }).dispose();
        } catch {
          // ignore dispose errors
        }
        viewerRef.current = null;
      }
    };
  }, [url]);

  return (
    <div className="splat-viewer">
      {loading && (
        <div className="viewer-loading">
          <div className="loading-spinner" />
          <span>Loading 3D model... {progress > 0 ? `${Math.round(progress)}%` : ""}</span>
        </div>
      )}
      {error && (
        <div className="viewer-error">
          <p>Failed to load 3D model</p>
          <p className="error-detail">{error}</p>
          <p className="error-hint">
            The model may still be generating, or the format may not be supported.
          </p>
        </div>
      )}
      <div ref={containerRef} className="splat-canvas" />
      {!loading && !error && (
        <div className="splat-hint">
          Left-drag to rotate | Right-drag to pan | Scroll to zoom
        </div>
      )}
    </div>
  );
}
