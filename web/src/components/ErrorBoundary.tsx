import {Component, ErrorInfo, ReactNode} from "react";

export default class ErrorBoundary extends Component<{children: ReactNode}, {error: boolean}> {
  state = {error: false};
  static getDerivedStateFromError() { return {error: true}; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error("Workspace rendering failed", error, info.componentStack); }
  render() {
    if (this.state.error) return <main className="empty-state" role="alert"><h2>The workspace could not be displayed</h2><p>Reload to recover your connection. Submitted tasks remain on the server.</p><button className="btn btn-primary" onClick={() => window.location.reload()}>Reload workspace</button></main>;
    return this.props.children;
  }
}
