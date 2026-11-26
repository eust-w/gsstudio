import { Link, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import TaskList from "./pages/TaskList";
import TaskCreate from "./pages/TaskCreate";
import TaskDetail from "./pages/TaskDetail";
import ViewerPage from "./pages/ViewerPage";
import SystemPage from "./pages/SystemPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<TaskList />} />
        <Route path="create" element={<TaskCreate />} />
        <Route path="system" element={<SystemPage />} />
        <Route path="tasks/:taskId" element={<TaskDetail />} />
        <Route path="viewer/:taskId/:artifactPath" element={<ViewerPage />} />
        <Route path="*" element={<div className="empty-state"><h2>Page not found</h2><p>This workspace page does not exist.</p><Link className="btn btn-primary" to="/">Back to scenes</Link></div>} />
      </Route>
    </Routes>
  );
}
