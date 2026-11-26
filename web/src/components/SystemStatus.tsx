import {createContext, ReactNode, useCallback, useContext, useEffect, useState} from "react";
import {Link} from "react-router-dom";
import {getSystemStatus, SystemStatus} from "../api";
import Icon from "./Icon";

const Context = createContext<{system: SystemStatus | null; error: string; refresh: () => void}>({system: null, error: "", refresh: () => {}});

export function SystemStatusProvider({children}: {children: ReactNode}) {
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const refresh = useCallback(() => setRevision(value => value + 1), []);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function update() {
      try {
        const result = await getSystemStatus(controller.signal);
        if (!controller.signal.aborted) { setSystem(result); setError(""); }
      } catch (error) {
        if (!controller.signal.aborted) { setSystem(null); setError(error instanceof Error ? error.message : "API unavailable"); }
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(update, 30000);
      }
    }
    void update();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [revision]);
  return <Context.Provider value={{system, error, refresh}}>{children}</Context.Provider>;
}

export const useSystemStatus = () => useContext(Context);

export function RuntimeNotice() {
  const {system, error} = useSystemStatus();
  if (!error && (!system || (system.ready && system.mode === "real"))) return null;
  return <div className={`runtime-notice ${error ? "runtime-error" : ""}`} role="status">
    <Icon name="alert" size={26} />
    <div><strong>{error ? "API connection unavailable" : system?.mode === "demo" ? "Demo runtime" : "Compute setup required"}</strong>
      <p>{error || (system?.mode === "demo" ? "Simulated stages only. Demo tasks do not generate or validate 3D models." : "Connect a ready GPU runtime to start reconstruction.")}</p></div>
    <Link to="/system" className="btn btn-sm">Check system</Link>
  </div>;
}
