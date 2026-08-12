const API_BASE = "/api";

export interface Task {
  id: string;
  name: string;
  type: "worldgen" | "worldgen_from_pano" | "worldrecon";
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  current_stage?: string;
  progress: number;
  created_at: string;
  updated_at?: string;
}

export interface StageInfo {
  name: string;
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  started_at?: string;
  completed_at?: string;
  duration_seconds?: number;
  error?: string;
}

export interface Artifact {
  name: string;
  path: string;
  size: number;
  type: "image" | "video" | "3dgs" | "pointcloud" | "mesh" | "data" | "other";
}

export interface TaskDetail extends Task {
  stages: StageInfo[];
  config: Record<string, unknown>;
  artifacts: Artifact[];
  error?: string;
  error_code?: string | null;
  error_summary?: string | null;
}

export interface SourceGenerationStatus {
  mode?: "image" | "prompt";
  state?: "generating_source" | "source_ready" | "expanding_panorama" | "completed" | "failed";
  source_attempt?: number;
  seed?: number;
  artifact_path?: string;
  width?: number;
  height?: number;
  duration_seconds?: number;
  error_code?: string | null;
  error_summary?: string | null;
}

export interface TaskQualityRecovery {
  available: boolean;
  state?: string;
  phase?: string | null;
  rescue_round?: number | null;
  source_attempt?: number;
  accepted_trajectories?: number;
  required_trajectories?: number;
  attempted_trajectories?: number;
  pending_trajectories?: number;
  generated_video_trajectories?: number;
  video_seconds?: number;
  limits?: {
    maximum_video_trajectories?: number;
    maximum_video_seconds?: number;
    maximum_source_regenerations?: number;
  };
  delivery_tier?: "strict_3d" | "degraded_3d" | "panorama_only" | null;
  best_source_attempt?: number;
  quality_warning?: string | null;
  recovery_exhausted?: boolean;
  transient_retry_count?: number;
  terminal_reason?: string | null;
  updated_at?: string;
  source_generation?: SourceGenerationStatus | null;
}

export interface GpuInfo {
  index: number;
  name: string;
  used_by?: string;
  vram_used_gb: number;
  vram_total_gb: number;
}

export interface GpuStatus {
  gpus: GpuInfo[];
  active_task?: string;
}

export interface TaskListResponse {
  tasks: Task[];
  total: number;
}

export async function createTask(formData: FormData): Promise<{ task_id: string }> {
  const resp = await fetch(`${API_BASE}/tasks`, { method: "POST", body: formData });
  if (!resp.ok) throw new Error(`Create task failed: ${resp.statusText}`);
  return resp.json();
}

export async function listTasks(params?: {
  status?: string;
  type?: string;
  limit?: number;
  offset?: number;
}): Promise<TaskListResponse> {
  const searchParams = new URLSearchParams();
  if (params?.status) searchParams.set("status", params.status);
  if (params?.type) searchParams.set("type", params.type);
  if (params?.limit) searchParams.set("limit", String(params.limit));
  if (params?.offset) searchParams.set("offset", String(params.offset));
  const qs = searchParams.toString();
  const resp = await fetch(`${API_BASE}/tasks${qs ? "?" + qs : ""}`);
  if (!resp.ok) throw new Error(`List tasks failed: ${resp.statusText}`);
  return resp.json();
}

export async function getTask(id: string): Promise<TaskDetail> {
  const resp = await fetch(`${API_BASE}/tasks/${id}`);
  if (!resp.ok) throw new Error(`Get task failed: ${resp.statusText}`);
  return resp.json();
}

export async function getTaskQuality(id: string): Promise<TaskQualityRecovery> {
  const resp = await fetch(`${API_BASE}/tasks/${id}/quality`);
  if (!resp.ok) throw new Error(`Quality status failed: ${resp.statusText}`);
  return resp.json();
}

export async function cancelTask(id: string): Promise<void> {
  const resp = await fetch(`${API_BASE}/tasks/${id}/cancel`, { method: "POST" });
  if (!resp.ok) throw new Error(`Cancel failed: ${resp.statusText}`);
}

export async function retryTask(id: string): Promise<void> {
  const resp = await fetch(`${API_BASE}/tasks/${id}/retry`, { method: "POST" });
  if (!resp.ok) throw new Error(`Retry failed: ${resp.statusText}`);
}

export async function retryTaskSource(
  id: string,
  source?: File,
  taskType: Task["type"] = "worldgen_from_pano",
): Promise<void> {
  const body = source ? new FormData() : undefined;
  if (body && source) body.set(taskType === "worldgen" ? "image" : "panorama", source);
  const resp = await fetch(`${API_BASE}/tasks/${id}/retry-source`, { method: "POST", body });
  if (!resp.ok) throw new Error(`Source retry failed: ${resp.statusText}`);
}

export async function updateTask(id: string, updates: { name: string }): Promise<TaskDetail> {
  const resp = await fetch(`${API_BASE}/tasks/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });
  if (!resp.ok) throw new Error(`Update task failed: ${resp.statusText}`);
  return resp.json();
}

export async function deleteTask(id: string): Promise<void> {
  const resp = await fetch(`${API_BASE}/tasks/${id}`, { method: "DELETE" });
  if (!resp.ok) throw new Error(`Delete task failed: ${resp.statusText}`);
}

export async function getArtifacts(taskId: string): Promise<Artifact[]> {
  const resp = await fetch(`${API_BASE}/tasks/${taskId}/artifacts`);
  if (!resp.ok) throw new Error(`Get artifacts failed: ${resp.statusText}`);
  const data = await resp.json();
  return data.artifacts || [];
}

export function getArtifactUrl(taskId: string, path: string): string {
  return `${API_BASE}/tasks/${taskId}/artifacts/${path}`;
}

export function getLogWebSocketUrl(taskId: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${API_BASE}/tasks/${taskId}/logs`;
}

export async function healthCheck(): Promise<{ status: string }> {
  const resp = await fetch(`${API_BASE}/health`);
  return resp.json();
}

export async function getGpuStatus(): Promise<GpuStatus> {
  const resp = await fetch(`${API_BASE}/gpu/status`);
  if (!resp.ok) throw new Error(`GPU status failed: ${resp.statusText}`);
  return resp.json();
}
