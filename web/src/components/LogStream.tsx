import { useCallback, useEffect, useRef, useState } from "react";
import { getLogWebSocketUrl } from "../api";

interface LogStreamProps {
  taskId: string;
}

interface LogLine {
  stage: string;
  timestamp: string;
  line: string;
}

const MAX_LINES = 500;

export default function LogStream({ taskId }: LogStreamProps) {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [connected, setConnected] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>();
  const reconnectDelayRef = useRef(1000);

  const connect = useCallback(() => {
    if (!taskId) return;

    const url = getLogWebSocketUrl(taskId);
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConnected(true);
      reconnectDelayRef.current = 1000;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setLines((prev) => {
          const next = [...prev, data];
          return next.length > MAX_LINES ? next.slice(-MAX_LINES) : next;
        });
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      // Reconnect with exponential backoff
      reconnectTimeoutRef.current = setTimeout(() => {
        reconnectDelayRef.current = Math.min(reconnectDelayRef.current * 2, 30000);
        connect();
      }, reconnectDelayRef.current);
    };

    ws.onerror = () => {
      ws.close();
    };

    wsRef.current = ws;
  }, [taskId]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [lines]);

  return (
    <div className="log-stream">
      <div className="log-header">
        <span className={`log-status ${connected ? "connected" : "disconnected"}`}>
          {connected ? "Live" : "Disconnected"}
        </span>
        <button className="btn btn-sm" onClick={() => setLines([])}>
          Clear
        </button>
      </div>
      <div className="log-container" ref={containerRef}>
        {lines.length === 0 ? (
          <div className="log-empty">Waiting for logs...</div>
        ) : (
          lines.map((entry, i) => (
            <div key={i} className={`log-line ${getLogLevel(entry.line)}`}>
              {entry.stage && <span className="log-stage">[{entry.stage}]</span>}
              <span className="log-text">{entry.line}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function getLogLevel(line: string): string {
  const lower = line.toLowerCase();
  if (lower.includes("error") || lower.includes("fail")) return "log-error";
  if (lower.includes("warn")) return "log-warning";
  if (lower.includes("info")) return "log-info";
  return "";
}
