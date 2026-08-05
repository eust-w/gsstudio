import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

interface PanoViewerProps {
  url: string;
}

export default function PanoViewer({ url }: PanoViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!containerRef.current || !url) return;

    const container = containerRef.current;
    let renderer: THREE.WebGLRenderer;
    let animationId: number;

    try {
      // Scene setup
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(
        75,
        container.clientWidth / container.clientHeight,
        0.1,
        1100
      );
      camera.position.set(0, 0, 0.1);

      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
      container.appendChild(renderer.domElement);

      // Load panorama texture
      const loader = new THREE.TextureLoader();
      loader.load(
        url,
        (texture) => {
          texture.colorSpace = THREE.SRGBColorSpace;
          const geometry = new THREE.SphereGeometry(500, 60, 40);
          geometry.scale(-1, 1, 1); // Invert for inside view
          const material = new THREE.MeshBasicMaterial({ map: texture });
          const sphere = new THREE.Mesh(geometry, material);
          scene.add(sphere);
          setLoading(false);
        },
        undefined,
        (err) => {
          setError("Failed to load panorama image");
          setLoading(false);
        }
      );

      // Mouse/touch controls
      let isDragging = false;
      let prevX = 0;
      let prevY = 0;
      let lon = 0;
      let lat = 0;
      let targetLon = 0;
      let targetLat = 0;

      const onPointerDown = (e: PointerEvent) => {
        isDragging = true;
        prevX = e.clientX;
        prevY = e.clientY;
      };

      const onPointerMove = (e: PointerEvent) => {
        if (!isDragging) return;
        targetLon += (prevX - e.clientX) * 0.15;
        targetLat += (e.clientY - prevY) * 0.15;
        targetLat = Math.max(-85, Math.min(85, targetLat));
        prevX = e.clientX;
        prevY = e.clientY;
      };

      const onPointerUp = () => {
        isDragging = false;
      };

      container.addEventListener("pointerdown", onPointerDown);
      container.addEventListener("pointermove", onPointerMove);
      container.addEventListener("pointerup", onPointerUp);
      container.addEventListener("pointerleave", onPointerUp);

      // Animation loop
      const animate = () => {
        animationId = requestAnimationFrame(animate);
        lon += (targetLon - lon) * 0.1;
        lat += (targetLat - lat) * 0.1;

        const phi = THREE.MathUtils.degToRad(90 - lat);
        const theta = THREE.MathUtils.degToRad(lon);

        camera.lookAt(
          500 * Math.sin(phi) * Math.cos(theta),
          500 * Math.cos(phi),
          500 * Math.sin(phi) * Math.sin(theta)
        );

        renderer.render(scene, camera);
      };
      animate();

      // Resize handler
      const onResize = () => {
        if (!container) return;
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
      };
      window.addEventListener("resize", onResize);

      return () => {
        cancelAnimationFrame(animationId);
        window.removeEventListener("resize", onResize);
        container.removeEventListener("pointerdown", onPointerDown);
        container.removeEventListener("pointermove", onPointerMove);
        container.removeEventListener("pointerup", onPointerUp);
        container.removeEventListener("pointerleave", onPointerUp);
        renderer.dispose();
        if (container.contains(renderer.domElement)) {
          container.removeChild(renderer.domElement);
        }
      };
    } catch (e) {
      setError("Failed to initialize 3D viewer");
      setLoading(false);
    }
  }, [url]);

  return (
    <div className="pano-viewer">
      {loading && <div className="viewer-loading">Loading panorama...</div>}
      {error && <div className="viewer-error">{error}</div>}
      <div ref={containerRef} className="pano-canvas" />
      <div className="pano-hint">Drag to look around</div>
    </div>
  );
}
