import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getArtifactUrl } from "../api";
import SplatViewer from "../components/SplatViewer";

export default function ViewerPage() {
  const { taskId, artifactPath } = useParams<{ taskId: string; artifactPath: string }>();
  const [url, setUrl] = useState("");

  useEffect(() => {
    if (taskId && artifactPath) {
      setUrl(getArtifactUrl(taskId, decodeURIComponent(artifactPath)));
    }
  }, [taskId, artifactPath]);

  if (!url) {
    return <div className="viewer-page loading">Loading...</div>;
  }

  return (
    <div className="viewer-page">
      <div className="viewer-header">
        <Link to={`/tasks/${taskId}`} className="back-link">
          &larr; Back to Task
        </Link>
      </div>
      <div className="viewer-container">
        <SplatViewer url={url} />
      </div>
    </div>
  );
}
