import { ReactNode, useCallback, useEffect, useRef, useState } from "react";

interface PreviewFrameProps {
  label: string;
  children: ReactNode;
}

export default function PreviewFrame({ label, children }: PreviewFrameProps) {
  const frameRef = useRef<HTMLDivElement>(null);
  const [nativeFullscreen, setNativeFullscreen] = useState(false);
  const [fallbackFullscreen, setFallbackFullscreen] = useState(false);
  const expanded = nativeFullscreen || fallbackFullscreen;

  const focusKeyboardViewer = useCallback(() => {
    window.setTimeout(() => {
      frameRef.current
        ?.querySelector<HTMLElement>("[data-keyboard-viewer]")
        ?.focus({ preventScroll: true });
    }, 0);
  }, []);

  useEffect(() => {
    const handleFullscreenChange = () => {
      const active = document.fullscreenElement === frameRef.current;
      setNativeFullscreen(active);
      if (active) focusKeyboardViewer();
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && fallbackFullscreen) {
        setFallbackFullscreen(false);
      }
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [fallbackFullscreen, focusKeyboardViewer]);

  const toggleFullscreen = async () => {
    const frame = frameRef.current;
    if (!frame) return;

    if (nativeFullscreen) {
      await document.exitFullscreen?.();
      return;
    }
    if (fallbackFullscreen) {
      setFallbackFullscreen(false);
      return;
    }

    if (typeof frame.requestFullscreen === "function") {
      try {
        await frame.requestFullscreen();
        focusKeyboardViewer();
        return;
      } catch {
        // Browser policy may reject native fullscreen; preserve the interaction
        // with the in-page fallback below.
      }
    }
    setFallbackFullscreen(true);
    focusKeyboardViewer();
  };

  return (
    <div
      ref={frameRef}
      className={`preview-frame ${fallbackFullscreen ? "preview-frame-fallback" : ""}`}
      data-testid="preview-frame"
    >
      {children}
      <button
        type="button"
        className="preview-fullscreen-button"
        aria-label={`${expanded ? "Exit" : "Enter"} ${label} fullscreen`}
        aria-pressed={expanded}
        onClick={toggleFullscreen}
      >
        <span aria-hidden="true">{expanded ? "↙" : "⛶"}</span>
        <span>{expanded ? "Exit fullscreen" : "Fullscreen"}</span>
      </button>
    </div>
  );
}
