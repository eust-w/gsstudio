import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createTask } from "../api";

type TabType = "worldgen" | "worldgen_from_pano" | "worldrecon";
type QualityPreset = "high" | "fast";

const TABS: { key: TabType; label: string; description: string }[] = [
  {
    key: "worldgen",
    label: "Text to World",
    description: "Generate a 3D world from a text prompt or image using HY-Pano-2 + WorldStereo-2",
  },
  {
    key: "worldgen_from_pano",
    label: "Panorama to World",
    description: "Upload an existing 360 panorama and generate a navigable 3D world",
  },
  {
    key: "worldrecon",
    label: "Multi-View Reconstruction",
    description: "Reconstruct 3D from multi-view images or video using WorldMirror-2",
  },
];

export default function TaskCreate() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<TabType>("worldgen");
  const [name, setName] = useState("");
  const [prompt, setPrompt] = useState("");
  const [qualityPreset, setQualityPreset] = useState<QualityPreset>("high");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [configJson, setConfigJson] = useState("{}");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const [panoramaFile, setPanoramaFile] = useState<File | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [videoFile, setVideoFile] = useState<File | null>(null);

  const panoInputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const imagesInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async () => {
    setError("");

    if (!name.trim()) {
      setName(`task-${Date.now()}`);
    }

    // Validate required files
    if (activeTab === "worldgen_from_pano" && !panoramaFile) {
      setError("Please upload a panorama image");
      return;
    }
    if (activeTab === "worldgen" && !imageFile && !prompt.trim()) {
      setError("Please provide a text prompt or upload an image");
      return;
    }
    if (activeTab === "worldrecon" && imageFiles.length === 0 && !videoFile) {
      setError("Please upload multi-view images or a video");
      return;
    }

    // Parse and merge advanced config. Form values intentionally win so the
    // visible prompt/quality choice can never be lost behind a stale JSON blob.
    let mergedConfig: Record<string, unknown>;
    try {
      const parsed = JSON.parse(configJson);
      if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
        throw new Error("Config must be an object");
      }
      mergedConfig = { ...parsed };
    } catch {
      setError("Invalid JSON in config");
      return;
    }
    if (activeTab === "worldgen" || activeTab === "worldgen_from_pano") {
      mergedConfig.quality_preset = qualityPreset;
      mergedConfig.recovery_policy = qualityPreset === "high" ? "auto" : "off";
      if (prompt.trim()) mergedConfig.prompt = prompt.trim();
    }

    setSubmitting(true);

    try {
      const formData = new FormData();
      formData.set("type", activeTab);
      formData.set("name", name.trim() || `task-${Date.now()}`);
      if (prompt.trim()) formData.set("prompt", prompt.trim());
      formData.set("config", JSON.stringify(mergedConfig));

      if (panoramaFile) formData.set("panorama", panoramaFile);
      if (imageFile) formData.set("image", imageFile);
      if (videoFile) formData.set("video", videoFile);
      imageFiles.forEach((f) => formData.append("images", f));

      const result = await createTask(formData);
      navigate(`/tasks/${result.task_id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create task");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="page">
      <div className="page-header">
        <h2>Create Task</h2>
      </div>

      <div className="tab-bar">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            className={`tab-btn ${activeTab === tab.key ? "active" : ""}`}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <p className="tab-description">
        {TABS.find((t) => t.key === activeTab)?.description}
      </p>

      {error && <div className="error-banner">{error}</div>}

      <div className="form-section">
        <label className="form-label" htmlFor="task-name">Task Name</label>
        <input
          id="task-name"
          type="text"
          className="form-input"
          placeholder={`task-${Date.now()}`}
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
      </div>

      {activeTab === "worldgen" && (
        <>
          <div className="form-section">
            <label className="form-label" htmlFor="task-prompt">Text Prompt</label>
            <textarea
              id="task-prompt"
              className="form-textarea"
              rows={4}
              placeholder="A futuristic city with tall buildings, neon lights, and flying cars..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </div>
          <div className="form-section">
            <label className="form-label">Input Image (optional)</label>
            <FileDropZone
              file={imageFile}
              onFile={setImageFile}
              accept="image/*"
              inputRef={imageInputRef}
              label="Drop an image here or click to browse"
            />
          </div>
        </>
      )}

      {activeTab === "worldgen_from_pano" && (
        <div className="form-section">
          <label className="form-label">Panorama Image</label>
          <FileDropZone
            file={panoramaFile}
            onFile={setPanoramaFile}
            accept="image/*"
            inputRef={panoInputRef}
            label="Drop a 360 panorama here or click to browse"
          />
        </div>
      )}

      {activeTab === "worldrecon" && (
        <>
          <div className="form-section">
            <label className="form-label">Multi-View Images</label>
            <MultiFileDropZone
              files={imageFiles}
              onFiles={setImageFiles}
              accept="image/*"
              inputRef={imagesInputRef}
              label="Drop multiple images here or click to browse"
            />
          </div>
          <div className="form-section">
            <label className="form-label">Or Video</label>
            <FileDropZone
              file={videoFile}
              onFile={setVideoFile}
              accept="video/*"
              inputRef={videoInputRef}
              label="Drop a video here or click to browse"
            />
          </div>
        </>
      )}

      {(activeTab === "worldgen" || activeTab === "worldgen_from_pano") && (
        <div className="form-section">
          <label className="form-label" htmlFor="quality-preset">Quality preset</label>
          <select
            id="quality-preset"
            className="form-input"
            value={qualityPreset}
            onChange={(event) => setQualityPreset(event.target.value as QualityPreset)}
          >
            <option value="high">High quality — full WorldStereo, quality gates, longer runtime</option>
            <option value="fast">Fast — 4-step WorldStereo DMD</option>
          </select>
          <p className="form-help">High quality is the default and uses the complete model plus strict trajectory checks. A valid image always returns a usable result; the 3D tier depends on how many trajectories pass. Automatic recovery tries safe paths and deterministic source seeds without weakening the quality gates.</p>
        </div>
      )}

      <div className="form-section">
        <button
          className="btn btn-secondary"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          {showAdvanced ? "Hide" : "Show"} Advanced Config
        </button>
        {showAdvanced && (
          <textarea
            id="advanced-config"
            aria-label="Advanced JSON config"
            className="form-textarea config-editor"
            rows={6}
            value={configJson}
            onChange={(e) => setConfigJson(e.target.value)}
            placeholder='{"seed": 42, "num_inference_steps": 50}'
          />
        )}
      </div>

      <div className="form-actions">
        <button
          className="btn btn-primary btn-lg"
          onClick={handleSubmit}
          disabled={submitting}
        >
          {submitting ? "Creating..." : "Create Task"}
        </button>
      </div>
    </div>
  );
}

function FileDropZone({
  file,
  onFile,
  accept,
  inputRef,
  label,
}: {
  file: File | null;
  onFile: (f: File | null) => void;
  accept: string;
  inputRef: React.RefObject<HTMLInputElement>;
  label: string;
}) {
  const [dragOver, setDragOver] = useState(false);

  return (
    <div
      className={`drop-zone ${dragOver ? "drag-over" : ""} ${file ? "has-file" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const f = e.dataTransfer.files[0];
        if (f) onFile(f);
      }}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        style={{ display: "none" }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
        }}
      />
      {file ? (
        <div className="file-preview">
          <span className="file-name">{file.name}</span>
          <span className="file-size">({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
          <button className="btn-remove" onClick={(e) => { e.stopPropagation(); onFile(null); }}>
            x
          </button>
        </div>
      ) : (
        <span className="drop-label">{label}</span>
      )}
    </div>
  );
}

function MultiFileDropZone({
  files,
  onFiles,
  accept,
  inputRef,
  label,
}: {
  files: File[];
  onFiles: (f: File[]) => void;
  accept: string;
  inputRef: React.RefObject<HTMLInputElement>;
  label: string;
}) {
  const [dragOver, setDragOver] = useState(false);

  return (
    <div
      className={`drop-zone multi ${dragOver ? "drag-over" : ""} ${files.length > 0 ? "has-file" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const newFiles = Array.from(e.dataTransfer.files);
        onFiles([...files, ...newFiles]);
      }}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple
        style={{ display: "none" }}
        onChange={(e) => {
          const newFiles = Array.from(e.target.files || []);
          onFiles([...files, ...newFiles]);
        }}
      />
      {files.length > 0 ? (
        <div className="file-list">
          {files.map((f, i) => (
            <div key={i} className="file-preview">
              <span className="file-name">{f.name}</span>
              <span className="file-size">({(f.size / 1024 / 1024).toFixed(1)} MB)</span>
              <button
                className="btn-remove"
                onClick={(e) => {
                  e.stopPropagation();
                  onFiles(files.filter((_, idx) => idx !== i));
                }}
              >
                x
              </button>
            </div>
          ))}
          <span className="drop-label">+ Add more</span>
        </div>
      ) : (
        <span className="drop-label">{label}</span>
      )}
    </div>
  );
}
