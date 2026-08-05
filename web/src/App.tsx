import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import TaskList from "./pages/TaskList";
import TaskCreate from "./pages/TaskCreate";
import TaskDetail from "./pages/TaskDetail";
import ViewerPage from "./pages/ViewerPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<TaskList />} />
        <Route path="create" element={<TaskCreate />} />
        <Route path="tasks/:taskId" element={<TaskDetail />} />
        <Route path="viewer/:taskId/:artifactPath" element={<ViewerPage />} />
      </Route>
    </Routes>
  );
}
