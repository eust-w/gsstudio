import {useEffect} from "react";
import {NavLink, Outlet, useLocation} from "react-router-dom";
import Icon from "./Icon";
import {SystemStatusProvider, useSystemStatus} from "./SystemStatus";

function Workspace() {
  const {system, error} = useSystemStatus();
  const {pathname} = useLocation();
  useEffect(() => { window.scrollTo(0, 0); }, [pathname]);
  const pageName = pathname === "/" ? "Scenes" : pathname === "/create" ? "Create scene" : pathname === "/system" ? "System" : "Scene details";
  return <div className="app-layout">
    <a className="skip-link" href="#workspace-main">Skip to content</a>
    <aside className="sidebar">
      <NavLink to="/" className="sidebar-logo"><Icon name="cube" size={36} /><div><h1>GSStudio</h1><span className="version">3D scene workspace</span></div></NavLink>
      <nav className="sidebar-nav" aria-label="Main navigation">
        <NavLink to="/" end className={({isActive}) => `nav-link ${isActive ? "active" : ""}`}><Icon name="scenes" /><span>Scenes</span></NavLink>
        <NavLink to="/create" className={({isActive}) => `nav-link ${isActive ? "active" : ""}`}><Icon name="plus" /><span>Create scene</span></NavLink>
        <NavLink to="/system" className={({isActive}) => `nav-link ${isActive ? "active" : ""}`}><Icon name="system" /><span>System</span></NavLink>
      </nav>
      <div className="sidebar-footer"><Icon name="cube" size={18} /><span>HY-World 2.0</span></div>
    </aside>
    <main className="main-content" id="workspace-main">
      <div className="workspace-topbar"><span>Workspace <span className="breadcrumb-divider">/</span> {pageName}</span>
        <NavLink to="/system" className="connection-status"><i className={error ? "offline" : system ? "online" : "checking"} />{error ? "API offline" : system ? "API connected" : "Connecting…"}</NavLink>
      </div>
      <Outlet />
    </main>
  </div>;
}

export default function Layout() { return <SystemStatusProvider><Workspace /></SystemStatusProvider>; }
