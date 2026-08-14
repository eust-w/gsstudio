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
  original_prompt?: string;
  effective_prompt?: string;
  requested_source_width?: number;
  requested_source_height?: number;
}

export interface TaskQualityRecovery {
  available: boolean;
  contract_version?: number;
  input_lane?: "production_multiview" | "generative_worldgen";
  success_policy?: "strict_free_view";
  state?: string;
  phase?: string | null;
  rescue_round?: number | null;
  source_attempt?: number;
  accepted_trajectories?: number;
  required_trajectories?: number;
  target_trajectories?: number;
  attempted_trajectories?: number;
  pending_trajectories?: number;
  generated_video_trajectories?: number;
  video_seconds?: number;
  limits?: {
    maximum_video_trajectories?: number;
    maximum_video_seconds?: number;
    maximum_source_regenerations?: number;
    maximum_source_attempts?: number;
    maximum_trajectories_per_source?: number;
  };
  delivery_tier?: "strict_3d" | null;
  free_view_eligible?: boolean;
  product_state?: "processing" | "repair_available" | "repairing" | "source_replacement_required" | "ready";
  user_message?: string;
  next_action?: "wait" | "repair_model" | "replace_source" | "view_3d";
  repair_eligibility?: {
    eligible?: boolean;
    requires_reanalysis?: boolean;
    reason?: string;
    offending_gaussian_count?: number;
    offending_gaussian_fraction?: number;
    attempts_used?: number;
    maximum_attempts?: number;
  };
  repair_report?: {
    state?: "running" | "repaired_pending_validation" | "passed" | "failed" | "failed_validation";
    attempt?: number;
    source_gaussian_count?: number;
    removed_gaussian_count?: number;
    result_gaussian_count?: number;
    original_artifacts_preserved?: boolean;
    navigation_stress_passed?: boolean;
    failure_reason?: string;
  } | null;
  post_panorama_preflight?: {
    state?: "passed" | "failed";
    composite_score?: number;
    failure_reasons?: string[];
  } | null;
  navigation_validation?: {
    free_view_eligible?: boolean;
    accepted_trajectory_count?: number;
    unique_translated_camera_centers?: number;
    camera_obb_half_width?: number;
    camera_obb_half_height?: number;
    human_scale?: number;
    azimuth_bin_count?: number;
    validation_render_count?: number;
    metrics?: { psnr?: number | null; ssim?: number | null; lpips?: number | null };
    failure_reasons?: string[];
  } | null;
  source_pool?: {
    state?: string;
    selected_candidate_id?: string | null;
    accepted_candidate_count?: number;
    ranking?: string[];
    selected_ids?: string[];
    candidates?: Array<{
      id?: string;
      seed?: number;
      composite_score?: number;
      accepted?: boolean;
      rejection_reasons?: string[];
    }>;
  } | null;
  input_preflight?: {
    state?: string;
    input_mode?: string;
    candidate_frame_count?: number;
    selected_frame_count?: number;
    failure_reasons?: string[];
  } | null;
  pilot_validation?: {
    state?: string;
    validation_render_count?: number;
    gaussian_count?: number;
    metrics?: { psnr?: number | null; ssim?: number | null; lpips?: number | null };
    failure_reasons?: string[];
  } | null;
  navigation_stress?: {
    state?: string;
    free_view_eligible?: boolean;
    gaussian_count?: number;
    walk?: { completed_steps?: number; accepted_steps?: number };
    representative_views?: unknown[];
    micro_translation_pairs?: unknown[];
    failure_reasons?: string[];
  } | null;
  global_budget?: Record<string, number>;
  training_mode?: "accepted_trajectories" | null;
  strict_accepted_count?: number;
  anchor_view_count?: number;
  training_input_manifest?: Record<string, unknown> | null;
  artifact_validation?: Record<string, unknown> | null;
  fallback_reason?: string | null;
  best_source_attempt?: number;
  quality_warning?: string | null;
  recovery_exhausted?: boolean;
  transient_retry_count?: number;
  source_guidance_collapse?: boolean;
  guidance_diagnostics?: {
    detected?: boolean;
    diagnostic_cutoff?: number;
    candidate_count?: number;
    observed_count?: number;
    missing_trajectory_ids?: string[];
    best_minimum_guidance_mask?: number | null;
  } | null;
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

export async function repairTaskModel(id: string): Promise<void> {
  const resp = await fetch(`${API_BASE}/tasks/${id}/repair`, { method: "POST" });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => null);
    throw new Error(detail?.detail || `Model repair failed: ${resp.statusText}`);
  }
}

export async function retryTaskSource(
  id: string,
  source?: File,
  taskType: Task["type"] = "worldgen_from_pano",
  images?: File[],
  video?: File,
): Promise<void> {
  const hasUpload = Boolean(source || images?.length || video);
  const body = hasUpload ? new FormData() : undefined;
  if (body && source) body.set(taskType === "worldgen" ? "image" : "panorama", source);
  images?.forEach((file) => body?.append("images", file));
  if (body && video) body.set("video", video);
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
