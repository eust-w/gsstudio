import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

interface PanoViewerProps { url: string; }
interface ViewControls { zoom: (delta: number) => void; reset: () => void; }
const DEFAULT_FOV = 75;

export default function PanoViewer({ url }: PanoViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const controlsRef = useRef<ViewControls | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [fov, setFov] = useState(DEFAULT_FOV);

  useEffect(() => {
    setLoading(true);
    setError("");
    setFov(DEFAULT_FOV);
    if (!containerRef.current || !url) {
      setLoading(false);
      setError("No panorama image is available");
      return;
    }
    const container = containerRef.current;
    let disposed = false;
    let renderer: THREE.WebGLRenderer | undefined;
    let animationId: number | undefined;
    let observer: ResizeObserver | undefined;
    let geometry: THREE.SphereGeometry | undefined;
    let material: THREE.MeshBasicMaterial | undefined;
    const textures = new Set<THREE.Texture>();
    const disposedTextures = new WeakSet<THREE.Texture>();
    const listenerCleanup: Array<() => void> = [];
    const disposeTexture = (texture: THREE.Texture) => {
      if (disposedTextures.has(texture)) return;
      disposedTextures.add(texture);
      texture.dispose();
    };
    const cleanup = () => {
      if (disposed) return;
      disposed = true;
      controlsRef.current = null;
      if (animationId !== undefined) cancelAnimationFrame(animationId);
      observer?.disconnect();
      listenerCleanup.forEach((remove) => remove());
      textures.forEach(disposeTexture);
      geometry?.dispose();
      material?.dispose();
      renderer?.dispose();
      renderer?.domElement.remove();
    };
    const fail = (message: string) => {
      if (disposed) return;
      cleanup();
      setError(message);
      setLoading(false);
    };

    try {
      const scene = new THREE.Scene();
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);
      const camera = new THREE.PerspectiveCamera(DEFAULT_FOV, width / height, 0.1, 1100);
      camera.position.set(0, 0, 0.1);
      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      container.appendChild(renderer.domElement);
      let draggingPointer: number | null = null;
      let prevX = 0;
      let prevY = 0;
      let lon = 0;
      let lat = 0;
      let targetLon = 0;
      let targetLat = 0;
      const zoom = (delta: number) => {
        camera.fov = Math.max(30, Math.min(100, camera.fov + delta));
        camera.updateProjectionMatrix();
        setFov(camera.fov);
      };
      controlsRef.current = {
        zoom,
        reset: () => {
          targetLon = 0;
          targetLat = 0;
          camera.fov = DEFAULT_FOV;
          camera.updateProjectionMatrix();
          setFov(DEFAULT_FOV);
        },
      };
      const onPointerDown = (event: PointerEvent) => {
        if (event.button !== 0 || draggingPointer !== null) return;
        draggingPointer = event.pointerId;
        prevX = event.clientX;
        prevY = event.clientY;
        container.setPointerCapture?.(event.pointerId);
        container.focus({ preventScroll: true });
      };
      const onPointerMove = (event: PointerEvent) => {
        if (draggingPointer !== event.pointerId) return;
        targetLon += (prevX - event.clientX) * 0.15;
        targetLat = Math.max(-85, Math.min(85, targetLat + (event.clientY - prevY) * 0.15));
        prevX = event.clientX;
        prevY = event.clientY;
      };
      const onPointerUp = (event: PointerEvent) => {
        if (draggingPointer !== event.pointerId) return;
        draggingPointer = null;
        if (container.hasPointerCapture?.(event.pointerId)) container.releasePointerCapture(event.pointerId);
      };
      const onWheel = (event: WheelEvent) => { event.preventDefault(); zoom(event.deltaY * 0.04); };
      const onKeyDown = (event: KeyboardEvent) => {
        if (event.key === "ArrowLeft") targetLon -= 5;
        else if (event.key === "ArrowRight") targetLon += 5;
        else if (event.key === "ArrowUp") targetLat = Math.min(85, targetLat + 5);
        else if (event.key === "ArrowDown") targetLat = Math.max(-85, targetLat - 5);
        else if (event.key === "+" || event.key === "=") zoom(-5);
        else if (event.key === "-") zoom(5);
        else if (event.key === "0") controlsRef.current?.reset();
        else return;
        event.preventDefault();
      };
      container.addEventListener("pointerdown", onPointerDown);
      container.addEventListener("pointermove", onPointerMove);
      container.addEventListener("pointerup", onPointerUp);
      container.addEventListener("pointercancel", onPointerUp);
      container.addEventListener("lostpointercapture", onPointerUp);
      container.addEventListener("wheel", onWheel, { passive: false });
      container.addEventListener("keydown", onKeyDown);
      listenerCleanup.push(() => {
        container.removeEventListener("pointerdown", onPointerDown);
        container.removeEventListener("pointermove", onPointerMove);
        container.removeEventListener("pointerup", onPointerUp);
        container.removeEventListener("pointercancel", onPointerUp);
        container.removeEventListener("lostpointercapture", onPointerUp);
        container.removeEventListener("wheel", onWheel);
        container.removeEventListener("keydown", onKeyDown);
      });
      const animate = () => {
        if (disposed) return;
        animationId = requestAnimationFrame(animate);
        lon += (targetLon - lon) * 0.1;
        lat += (targetLat - lat) * 0.1;
        const phi = THREE.MathUtils.degToRad(90 - lat);
        const theta = THREE.MathUtils.degToRad(lon);
        camera.lookAt(500 * Math.sin(phi) * Math.cos(theta), 500 * Math.cos(phi), 500 * Math.sin(phi) * Math.sin(theta));
        renderer!.render(scene, camera);
      };
      const onResize = () => {
        if (disposed) return;
        const nextWidth = Math.max(1, container.clientWidth);
        const nextHeight = Math.max(1, container.clientHeight);
        camera.aspect = nextWidth / nextHeight;
        camera.updateProjectionMatrix();
        renderer!.setSize(nextWidth, nextHeight);
      };
      observer = new ResizeObserver(onResize);
      observer.observe(container);
      animate();
      const pendingTexture = new THREE.TextureLoader().load(url, (texture) => {
        if (disposed) { disposeTexture(texture); return; }
        textures.add(texture);
        try {
          texture.colorSpace = THREE.SRGBColorSpace;
          geometry = new THREE.SphereGeometry(500, 60, 40);
          geometry.scale(-1, 1, 1);
          material = new THREE.MeshBasicMaterial({ map: texture });
          scene.add(new THREE.Mesh(geometry, material));
          setLoading(false);
        } catch { fail("Failed to prepare panorama image"); }
      }, undefined, () => fail("Failed to load panorama image"));
      if (pendingTexture) {
        if (disposed) disposeTexture(pendingTexture);
        else textures.add(pendingTexture);
      }
    } catch { fail("Failed to initialize 3D viewer"); }
    return cleanup;
  }, [url]);

  return <div className="pano-viewer">
    {loading && <div className="viewer-loading" role="status">Loading panorama...</div>}
    {error && <div className="viewer-error" role="alert">{error}</div>}
    <div ref={containerRef} className="pano-canvas" tabIndex={0} aria-label="Panorama viewport. Drag or use arrow keys to look around. Scroll or use plus and minus to zoom." style={{ touchAction: "none" }} />
    {!error && <div className="pano-toolbar"><span className="pano-hint">Drag to look around · scroll to zoom</span><div className="pano-controls"><button className="btn btn-sm" type="button" disabled={loading || fov <= 30} aria-label="Zoom in panorama" onClick={() => controlsRef.current?.zoom(-10)}>+</button><button className="btn btn-sm" type="button" disabled={loading || fov >= 100} aria-label="Zoom out panorama" onClick={() => controlsRef.current?.zoom(10)}>−</button><button className="btn btn-sm" type="button" disabled={loading} onClick={() => controlsRef.current?.reset()}>Reset view</button></div></div>}
  </div>;
}
