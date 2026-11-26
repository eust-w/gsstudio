import { FormEvent, useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { createTask } from "../api";
import { useSystemStatus } from "../components/SystemStatus";
import UploadInput, { formatFileSize, MAX_IMAGES } from "../components/UploadInput";

type TabType = "worldgen" | "worldgen_from_pano" | "worldrecon";
const TABS: { key: TabType; label: string; tag: string; description: string; steps: string[] }[] = [
  { key: "worldrecon", label: "Multi-View Reconstruction", tag: "Production recommended", description: "Reconstruct a real place from overlapping photos or a video with camera movement.", steps: ["Validate capture quality", "Recover cameras & reconstruct", "Build viewer boundaries", "Validate free-view navigation"] },
  { key: "worldgen", label: "Text / Image to World", tag: "Generative", description: "Start with a scene description or a reference image and generate a new environment.", steps: ["Generate & rank source views", "Expand a 360° panorama", "Generate translated trajectories", "Train & validate Gaussian splats"] },
  { key: "worldgen_from_pano", label: "Panorama to World", tag: "Generative", description: "Expand an existing 360° panorama into a scene, subject to strict geometry validation.", steps: ["Validate the panorama", "Plan translated trajectories", "Reconstruct & train splats", "Validate free-view navigation"] },
];

function PipelineIcon({ type }: { type: TabType }) {
  return <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">{type === "worldrecon"
    ? <><path d="m12 3 9 5v9l-9 5-9-5V8l9-5Z" /><path d="m3 8 9 5 9-5M12 13v9M7.5 5.5l9 5" /></>
    : type === "worldgen" ? <><path d="m12 2 2.7 7.3L22 12l-7.3 2.7L12 22l-2.7-7.3L2 12l7.3-2.7L12 2Z" /><path d="M20 2v4m-2-2h4" /></>
      : <><ellipse cx="12" cy="6" rx="9" ry="3" /><path d="M3 6v12c0 4 18 4 18 0V6M3 13l4-3 5 5 4-3 5 3" /></>}</svg>;
}

export default function TaskCreate() {
  const navigate = useNavigate();
  const { system, error: systemError, refresh } = useSystemStatus();
  const [activeTab, setActiveTab] = useState<TabType>("worldrecon");
  const [inputMode, setInputMode] = useState<"images" | "video">("images");
  const [name, setName] = useState("");
  const [prompt, setPrompt] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [seed, setSeed] = useState("");
  const [reconGpus, setReconGpus] = useState("");
  const [configJson, setConfigJson] = useState("{}");
  const [submitting, setSubmitting] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState("");
  const [panoramaFiles, setPanoramaFiles] = useState<File[]>([]);
  const [referenceFiles, setReferenceFiles] = useState<File[]>([]);
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [videoFiles, setVideoFiles] = useState<File[]>([]);
  const defaultName = useRef(`Scene ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`);
  const submittingRef = useRef(false);
  const uploadRef = useRef<AbortController | null>(null);
  const attemptRef = useRef<{ fingerprint: string; key: string } | null>(null);
  const selectedPipeline = TABS.find((tab) => tab.key === activeTab)!;
  const activeFiles = activeTab === "worldrecon" ? (inputMode === "images" ? imageFiles : videoFiles) : activeTab === "worldgen" ? referenceFiles : panoramaFiles;
  const totalBytes = activeFiles.reduce((total, file) => total + file.size, 0);
  const capability = system?.capabilities.find((item) => item.task_type === activeTab);
  const runtimeBlocked = Boolean(systemError || (system?.mode === "real" && capability?.ready === false));
  const primaryBlocker = capability?.blockers.find((blocker) => /gpu|vlm|vllm/i.test(blocker.code));
  useEffect(() => () => uploadRef.current?.abort(), []);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (submittingRef.current || runtimeBlocked) return;
    setError("");
    if (name.trim().length > 120) { setError("Task name must be at most 120 characters."); return; }
    if (activeTab === "worldgen_from_pano" && !panoramaFiles.length) { setError("Please upload a panorama image"); return; }
    if (activeTab === "worldgen" && !referenceFiles.length && !prompt.trim()) { setError("Please provide a text prompt or upload an image"); return; }
    if (activeTab === "worldgen" && prompt.trim().length > 2000) { setError("Prompt must be at most 2000 characters."); return; }
    if (activeTab === "worldrecon") {
      if (!activeFiles.length) { setError("Please upload multi-view images or a video"); return; }
      if (inputMode === "images" && imageFiles.length < 12) { setError("Production preflight requires at least 12 multi-view images"); return; }
      if (imageFiles.length > MAX_IMAGES && inputMode === "images") { setError(`Choose at most ${MAX_IMAGES} multi-view images.`); return; }
    }
    let mergedConfig: Record<string, unknown>;
    try {
      const parsed = JSON.parse(configJson);
      if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") throw new Error();
      mergedConfig = { ...parsed };
    } catch { setError("Invalid JSON in config. Use a JSON object such as {\"seed\": 42}."); return; }
    const allowedKeys = new Set(["prompt", "seed", "quality_preset", "recovery_policy", "success_policy", "input_lane", ...(activeTab === "worldrecon" ? ["recon_gpus"] : [])]);
    const unsupported = Object.keys(mergedConfig).filter((key) => !allowedKeys.has(key));
    if (unsupported.length) { setError(`Unsupported config fields: ${unsupported.join(", ")}. Pipeline paths and quality gates are managed by the server.`); return; }
    if (seed.trim()) mergedConfig.seed = Number(seed);
    if (mergedConfig.seed !== undefined && (typeof mergedConfig.seed !== "number" || !Number.isInteger(mergedConfig.seed) || mergedConfig.seed < 0 || mergedConfig.seed > 4294967295)) { setError("Seed must be an integer from 0 to 4294967295."); return; }
    if (activeTab === "worldrecon" && reconGpus) mergedConfig.recon_gpus = Number(reconGpus);
    if (mergedConfig.recon_gpus !== undefined && (typeof mergedConfig.recon_gpus !== "number" || !Number.isInteger(mergedConfig.recon_gpus) || mergedConfig.recon_gpus < 1 || mergedConfig.recon_gpus > 5)) { setError("Reconstruction GPUs must be an integer from 1 to 5."); return; }
    delete mergedConfig.prompt;
    if (activeTab !== "worldrecon") {
      mergedConfig.quality_preset = "high";
      mergedConfig.recovery_policy = "auto";
      if (activeTab === "worldgen" && prompt.trim()) mergedConfig.prompt = prompt.trim();
    }
    mergedConfig.success_policy = "strict_free_view";
    mergedConfig.input_lane = activeTab === "worldrecon" ? "production_multiview" : "generative_worldgen";
    const taskName = name.trim() || defaultName.current;
    const fingerprint = JSON.stringify({ type: activeTab, name: taskName, config: mergedConfig, files: activeFiles.map((file) => [file.name, file.size, file.lastModified]) });
    if (attemptRef.current?.fingerprint !== fingerprint) attemptRef.current = { fingerprint, key: typeof crypto.randomUUID === "function" ? crypto.randomUUID() : `task-${Date.now()}-${Math.random().toString(36).slice(2)}` };
    const controller = new AbortController();
    uploadRef.current = controller;
    submittingRef.current = true;
    setSubmitting(true);
    setUploadProgress(0);
    try {
      const formData = new FormData();
      formData.set("type", activeTab);
      formData.set("name", taskName);
      if (activeTab === "worldgen" && prompt.trim()) formData.set("prompt", prompt.trim());
      formData.set("config", JSON.stringify(mergedConfig));
      if (activeTab === "worldgen_from_pano" && panoramaFiles[0]) formData.set("panorama", panoramaFiles[0]);
      if (activeTab === "worldgen" && referenceFiles[0]) formData.set("image", referenceFiles[0]);
      if (activeTab === "worldrecon" && inputMode === "video" && videoFiles[0]) formData.set("video", videoFiles[0]);
      if (activeTab === "worldrecon" && inputMode === "images") imageFiles.forEach((file) => formData.append("images", file));
      const result = await createTask(formData, { onProgress: setUploadProgress, signal: controller.signal, idempotencyKey: attemptRef.current.key });
      navigate(`/tasks/${result.task_id}`);
    } catch (cause) { setError(controller.signal.aborted ? "Upload cancelled. You can retry with the same inputs safely." : cause instanceof Error ? cause.message : "Failed to create task"); }
    finally { submittingRef.current = false; uploadRef.current = null; setSubmitting(false); }
  };

  return <div className="page create-page">
    <div className="page-header"><div><h2>Create scene</h2><p className="page-subtitle">Turn your captures and ideas into an explorable 3D scene.</p></div><Link className="btn btn-secondary" to="/">Back to workspace</Link></div>
    <form onSubmit={handleSubmit} noValidate>
      <div className="create-layout">
        <div className="create-main">
          <section className="form-panel">
            <div className="section-heading"><span className="step-number">01</span><div><h3>Choose your starting point</h3><p>Three workflows. One connected workspace.</p></div></div>
            <div className="pipeline-options">{TABS.map((tab) => <button type="button" key={tab.key} disabled={submitting} aria-pressed={activeTab === tab.key} className={`pipeline-option ${activeTab === tab.key ? "active" : ""}`} onClick={() => { setActiveTab(tab.key); setError(""); }}><span className="pipeline-icon"><PipelineIcon type={tab.key} /></span><span className="pipeline-name">{tab.label}</span><span className="pipeline-tag">{tab.tag}</span><span className="pipeline-copy">{tab.description}</span><span className="pipeline-selection" aria-hidden="true">{activeTab === tab.key ? "✓" : ""}</span></button>)}</div>
          </section>
          <section className="form-panel">
            <div className="section-heading"><span className="step-number">02</span><div><h3>Add your source</h3><p>{selectedPipeline.description}</p></div></div>
            <div className="form-section"><label className="form-label" htmlFor="task-name">Task Name</label><input id="task-name" className="form-input" value={name} maxLength={120} placeholder={defaultName.current} disabled={submitting} onChange={(event) => setName(event.target.value)} /><p className="form-help">Leave blank to use the suggested name. You can rename it later.</p></div>
            {activeTab === "worldgen" && <>
              <div className="form-section"><label className="form-label" htmlFor="task-prompt">Text Prompt</label><textarea id="task-prompt" className="form-textarea" rows={4} maxLength={2000} disabled={submitting} placeholder="Describe a place, its materials, lighting, and atmosphere…" value={prompt} onChange={(event) => setPrompt(event.target.value)} /><div className="field-counter">{prompt.length} / 2,000</div></div>
              <div className="form-section"><span className="form-label">Reference image <span className="muted">(optional)</span></span><UploadInput files={referenceFiles} onFiles={setReferenceFiles} kind="image" label="Drop a reference image here" hint="JPEG, PNG, WebP, TIFF or BMP · up to 32 MiB" disabled={submitting} /><p className="form-help">Without an image, GSStudio generates a source view first from your prompt, then expands it into a panorama.</p></div>
            </>}
            {activeTab === "worldgen_from_pano" && <div className="form-section"><span className="form-label">Panorama Image</span><UploadInput files={panoramaFiles} onFiles={setPanoramaFiles} kind="image" label="Drop a 360° panorama here" hint="Equirectangular 2:1 image recommended · up to 32 MiB" disabled={submitting} /><p className="form-help">Use a sharp panorama with clear scene structure. The pipeline evaluates whether it supports translated viewpoints.</p></div>}
            {activeTab === "worldrecon" && <>
              <div className="upload-mode-switch" aria-label="Reconstruction input"><button type="button" disabled={submitting} className={inputMode === "images" ? "active" : ""} aria-pressed={inputMode === "images"} onClick={() => { setInputMode("images"); setError(""); }}>Photo collection</button><button type="button" disabled={submitting} className={inputMode === "video" ? "active" : ""} aria-pressed={inputMode === "video"} onClick={() => { setInputMode("video"); setError(""); }}>Video capture</button></div>
              {inputMode === "images" ? <div className="form-section"><span className="form-label">Multi-View Images <span className="muted">{imageFiles.length} / 48 selected</span></span><UploadInput files={imageFiles} onFiles={setImageFiles} kind="image" multiple label="Drop your scene photos here" hint="12–48 images · JPEG, PNG, WebP, TIFF or BMP · 32 MiB each" disabled={submitting} /><p className="form-help">Capture overlapping views while moving through the scene. At least 12 usable 512px+ frames with real camera translation are required. Pure rotation is rejected.</p></div>
                : <div className="form-section"><span className="form-label">Video capture</span><UploadInput files={videoFiles} onFiles={setVideoFiles} kind="video" label="Drop a video capture here" hint="MP4, MOV, WebM, MKV, AVI or M4V · up to 2 GiB" disabled={submitting} /><p className="form-help">Walk slowly through the space with a steady camera. At least 24 decodable frames are needed; sharp frames are selected by the server.</p></div>}
            </>}
          </section>
          <section className="form-panel">
            <div className="section-heading"><span className="step-number">03</span><div><h3>Processing settings</h3><p>Quality is validated before a scene is ready to explore.</p></div></div>
            <div className="quality-setting"><span className="quality-mark" aria-hidden="true">✓</span><div><strong>{activeTab === "worldrecon" ? "Production contract" : "Quality preset"}</strong><p className="form-help">{activeTab === "worldrecon" ? "Strict free-view reconstruction · camera validation · 1,000-step navigation stress test. WASD navigation opens after every gate passes." : "Generative, strict but not guaranteed — four ranked sources, up to three source attempts, 24 WorldStereo trajectories or four hours. Six trajectories are the training minimum and ten are the recovery target. No anchored or degraded 3D is delivered."}</p></div><span className="pipeline-tag">High quality</span></div>
            <button type="button" className="btn btn-secondary advanced-toggle" aria-expanded={showAdvanced} aria-controls="advanced-settings" onClick={() => setShowAdvanced(!showAdvanced)}>{showAdvanced ? "Hide" : "Show"} Advanced Config <span aria-hidden="true">{showAdvanced ? "−" : "+"}</span></button>
            {showAdvanced && <div id="advanced-settings" className="advanced-settings">
              <div className="advanced-fields"><div><label className="form-label" htmlFor="task-seed">Random seed</label><input id="task-seed" className="form-input" type="number" min={0} max={4294967295} step={1} value={seed} onChange={(event) => setSeed(event.target.value)} disabled={submitting} placeholder="1024 (server default)" /><p className="form-help">Repeat the same seed and inputs for reproducible source generation.</p></div>{activeTab === "worldrecon" && <div><label className="form-label" htmlFor="task-gpus">Reconstruction GPUs</label><select id="task-gpus" className="form-input" value={reconGpus} onChange={(event) => setReconGpus(event.target.value)} disabled={submitting}><option value="">Server default</option>{[1, 2, 3, 4, 5].map((count) => <option key={count} value={count}>{count} GPU{count === 1 ? "" : "s"}</option>)}</select><p className="form-help">Uses available server GPUs when the task begins.</p></div>}</div>
              <label className="form-label" htmlFor="advanced-config">Advanced JSON config</label><textarea id="advanced-config" aria-label="Advanced JSON config" className="form-textarea config-editor" rows={4} disabled={submitting} value={configJson} onChange={(event) => setConfigJson(event.target.value)} placeholder='{"seed": 42}' /><p className="form-help">Supports seed{activeTab === "worldrecon" ? " and recon_gpus" : ""}. Visible fields override JSON. Quality gates and runtime paths are server-managed.</p>
            </div>}
          </section>
        </div>
        <aside className="create-summary">
          <div className="summary-heading"><h3>From source to scene</h3></div>
          <div className="summary-pipeline"><PipelineIcon type={activeTab} /><strong>{selectedPipeline.label}</strong></div>
          <ol className="summary-list">{selectedPipeline.steps.map((step, index) => <li className="summary-step" key={step}><span>{String(index + 1).padStart(2, "0")}</span>{step}</li>)}</ol>
          <div className="summary-total"><span>Source files</span><strong>{activeFiles.length}</strong><span>Upload size</span><strong>{formatFileSize(totalBytes)}</strong><span>Target</span><strong>{system?.mode === "demo" ? "Pipeline preview only" : "Validated 3DGS"}</strong></div>
          <p className="form-help">Input quality and available compute determine the result and processing time. Failed quality gates remain visible in the task.</p>
          {system?.mode === "demo" && <div className="info-banner">Demo mode exercises the workflow only. It does not generate or validate a real 3D scene.</div>}
          {runtimeBlocked && <div className="error-banner" role="alert"><strong>Runtime not ready</strong><p>This server is not ready to run this pipeline.</p><p>{systemError || primaryBlocker?.message || "Required runtime dependencies are missing."}</p><Link to="/system" className="btn btn-sm">View all runtime checks</Link><button type="button" className="btn btn-sm" onClick={refresh}>Check again</button></div>}
          {error && <div className="error-banner" role="alert">{error}</div>}
          <button type="submit" className="btn btn-primary btn-lg create-submit" disabled={submitting || runtimeBlocked}>{submitting ? <><span className="spinner" /> Creating...</> : <>Create Task <span aria-hidden="true">↗</span></>}</button>
          {submitting && <div className="upload-progress" role="status"><progress max={100} value={uploadProgress} aria-label="Upload progress" /><p className="form-help">{uploadProgress < 100 ? `Uploading source files · ${Math.round(uploadProgress)}%` : "Upload complete. Server is validating the inputs…"}</p><button className="btn btn-sm" type="button" onClick={() => uploadRef.current?.abort()}>Cancel upload</button></div>}
          <p className="summary-footnote">Files stay on your configured GSStudio server.</p>
        </aside>
      </div>
    </form>
  </div>;
}
