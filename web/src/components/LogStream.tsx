import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { getLogWebSocketUrl } from "../api";

interface LogStreamProps { taskId: string; }
interface LogLine { stage: string; timestamp: string; line: string; id: number; }
type ConnectionState = "connecting" | "live" | "reconnecting" | "complete" | "unavailable";
const MAX_LINES = 1000;
const STATUS_LABEL: Record<ConnectionState, string> = { connecting: "Connecting", live: "Live", reconnecting: "Reconnecting", complete: "Finished", unavailable: "Unavailable" };

export default function LogStream({ taskId }: LogStreamProps) {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [connectionDetail, setConnectionDetail] = useState("");
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const [search, setSearch] = useState("");
  const [follow, setFollow] = useState(true);
  const [dropped, setDropped] = useState(0);
  const [exportError, setExportError] = useState("");
  const filter = useDeferredValue(search.toLocaleLowerCase());
  const containerRef = useRef<HTMLDivElement>(null);
  const cursorRef = useRef("");
  const nextIdRef = useRef(0);
  const seenRef = useRef(new Set<string>());
  const filteredLines = useMemo(() => filter ? lines.filter((entry) => `${entry.stage} ${entry.line}`.toLocaleLowerCase().includes(filter)) : lines, [lines, filter]);

  useEffect(() => {
    setLines([]);
    setDropped(0);
    cursorRef.current = "";
    seenRef.current.clear();
    nextIdRef.current = 0;
  }, [taskId]);

  useEffect(() => {
    let disposed = false;
    let terminal = false;
    let socket: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let delay = 1000;
    setConnectionDetail("");
    const scheduleReconnect = () => {
      if (disposed || terminal) return;
      setConnection("reconnecting");
      setConnectionDetail(`Connection interrupted. Retrying in ${Math.round(delay / 1000)}s.`);
      timer = setTimeout(connect, delay);
      delay = Math.min(delay * 2, 30000);
    };
    const connect = () => {
      if (disposed || terminal || !taskId) return;
      setConnection("connecting");
      const url = new URL(getLogWebSocketUrl(taskId), window.location.href);
      if (cursorRef.current) url.searchParams.set("cursor", cursorRef.current);
      try { socket = new WebSocket(url.toString()); }
      catch { scheduleReconnect(); return; }
      const currentSocket = socket;
      socket.onopen = () => {
        if (disposed) return;
        setConnection("live");
        setConnectionDetail("");
      };
      socket.onmessage = (event) => {
        if (disposed || terminal) return;
        try {
          const data: unknown = JSON.parse(event.data);
          if (!data || typeof data !== "object") return;
          const message = data as Record<string, unknown>;
          if (message.type === "complete") {
            terminal = true;
            setConnection("complete");
            setConnectionDetail(typeof message.status === "string" ? `Task ${message.status}. Log stream is complete.` : "Log stream is complete.");
            currentSocket.close();
            return;
          }
          if (typeof message.line !== "string") return;
          if (typeof message.cursor === "string") {
            cursorRef.current = message.cursor;
            if (seenRef.current.has(message.cursor)) return;
            seenRef.current.add(message.cursor);
            if (seenRef.current.size > MAX_LINES * 2) seenRef.current.delete(seenRef.current.values().next().value!);
          }
          delay = 1000;
          const line: LogLine = { stage: typeof message.stage === "string" ? message.stage : "", timestamp: typeof message.timestamp === "string" ? message.timestamp : "", line: message.line, id: nextIdRef.current++ };
          setLines((previous) => [...previous.slice(-(MAX_LINES - 1)), line]);
          if (line.id >= MAX_LINES) setDropped((count) => count + 1);
        } catch { /* Ignore non-log frames without breaking the connection. */ }
      };
      socket.onclose = (event) => {
        if (disposed || terminal) return;
        if ([1008, 4004, 4401, 4403].includes(event.code)) {
          terminal = true;
          setConnection("unavailable");
          setConnectionDetail(event.code === 4004 ? "This task no longer exists." : "The server denied access to this log stream.");
          return;
        }
        scheduleReconnect();
      };
      socket.onerror = () => { if (!disposed) currentSocket.close(); };
    };
    connect();
    return () => {
      disposed = true;
      if (timer) clearTimeout(timer);
      if (socket) {
        socket.onopen = null;
        socket.onmessage = null;
        socket.onclose = null;
        socket.onerror = null;
        socket.close();
      }
    };
  }, [taskId, reconnectAttempt]);

  useEffect(() => {
    if (follow && containerRef.current) containerRef.current.scrollTop = containerRef.current.scrollHeight;
  }, [filteredLines, follow]);

  const download = () => {
    setExportError("");
    try {
      const content = lines.map((entry) => [entry.timestamp, entry.stage ? `[${entry.stage}]` : "", entry.line].filter(Boolean).join(" ")).join("\n");
      const url = URL.createObjectURL(new Blob([content + "\n"], { type: "text/plain;charset=utf-8" }));
      const link = document.createElement("a");
      link.href = url;
      link.download = `gsstudio-${taskId}-logs.txt`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch { setExportError("Logs could not be downloaded. Try again in a supported browser."); }
  };

  return <div className="log-stream">
    <div className="log-header"><div className="log-meta"><span className={`log-status ${connection === "live" ? "connected" : "disconnected"}`} role="status">{STATUS_LABEL[connection]}</span><span>{filteredLines.length} / {lines.length} lines</span></div><div className="log-tools"><button type="button" className={`btn btn-sm ${follow ? "active" : ""}`} aria-pressed={follow} onClick={() => setFollow(!follow)}>{follow ? "Pause scrolling" : "Follow logs"}</button><button type="button" className="btn btn-sm" disabled={!lines.length} onClick={download}>Download logs</button><button type="button" className="btn btn-sm" disabled={!lines.length} onClick={() => { setLines([]); setDropped(0); nextIdRef.current = 0; }}>Clear</button><button type="button" className="btn btn-sm" onClick={() => setReconnectAttempt((count) => count + 1)}>Reconnect</button></div></div>
    <div className="log-filter"><input className="form-input" aria-label="Filter logs" placeholder="Filter by message or stage…" value={search} onChange={(event) => setSearch(event.target.value)} /><span className="form-help">{dropped ? `Latest ${MAX_LINES.toLocaleString()} lines retained; ${dropped.toLocaleString()} older lines omitted.` : "Live output from your pipeline stages."}</span></div>
    {connectionDetail && <p className="log-connection-detail">{connectionDetail}</p>}
    {exportError && <p className="error-banner" role="alert">{exportError}</p>}
    <div className="log-container" ref={containerRef} role="log" aria-label="Pipeline log output" aria-live="off" onScroll={(event) => { const element = event.currentTarget; if (element.scrollHeight - element.scrollTop - element.clientHeight > 60) setFollow(false); }}>
      {!filteredLines.length ? <div className="log-empty">{search ? "No logs match your filter." : connection === "complete" ? "This task produced no log output." : "Waiting for pipeline output…"}</div> : filteredLines.map((entry) => <div key={entry.id} className={`log-line ${getLogLevel(entry.line)}`}>{entry.timestamp && <time className="log-time">{entry.timestamp}</time>}{entry.stage && <span className="log-stage">[{entry.stage}]</span>}<span className="log-text">{entry.line}</span></div>)}
    </div>
  </div>;
}

function getLogLevel(line: string): string {
  if (/\b(error|failed|failure|fatal|traceback)\b/i.test(line)) return "log-error";
  if (/\b(warn|warning)\b/i.test(line)) return "log-warning";
  if (/\binfo\b/i.test(line)) return "log-info";
  return "";
}
