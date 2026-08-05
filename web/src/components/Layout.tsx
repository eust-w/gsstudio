import { NavLink, Outlet } from "react-router-dom";

export default function Layout() {
  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <h1>GSStudio</h1>
          <span className="version">v0.1.0</span>
        </div>
        <nav className="sidebar-nav">
          <NavLink to="/" end className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            <span className="nav-icon">&#9776;</span> Tasks
          </NavLink>
          <NavLink to="/create" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            <span className="nav-icon">+</span> Create Task
          </NavLink>
        </nav>
        <div className="sidebar-footer">
          <span className="gpu-indicator">HY-World 2.0</span>
        </div>
      </aside>
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}
