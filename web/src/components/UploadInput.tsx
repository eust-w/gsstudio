import { useEffect, useId, useRef, useState } from "react";

export const MAX_IMAGE_BYTES = 32 * 1024 * 1024;
export const MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024;
export const MAX_IMAGES = 48;
const IMAGE_EXTENSIONS = /\.(jpe?g|png|webp|tiff?|bmp)$/i;
const VIDEO_EXTENSIONS = /\.(mp4|mov|webm|mkv|avi|m4v)$/i;

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GiB`;
}

export function validateFile(file: File, kind: "image" | "video"): string | null {
  const supported = kind === "image" ? IMAGE_EXTENSIONS : VIDEO_EXTENSIONS;
  if (!supported.test(file.name) || (file.type && !file.type.startsWith(`${kind}/`))) return `${file.name}: choose a supported ${kind} file.`;
  if (!file.size) return `${file.name}: empty files cannot be uploaded.`;
  const limit = kind === "image" ? MAX_IMAGE_BYTES : MAX_UPLOAD_BYTES;
  if (file.size > limit) return `${file.name}: exceeds the ${formatFileSize(limit)} limit.`;
  return null;
}

function UploadIcon() {
  return <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="M12 16V3m-4 4 4-4 4 4M4 14v5a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5" /></svg>;
}

function FileThumbnail({ file, kind }: { file: File; kind: "image" | "video" }) {
  const [url, setUrl] = useState("");
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    setFailed(false);
    if (kind !== "image" || typeof URL.createObjectURL !== "function") return;
    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file, kind]);
  return <div className="upload-thumbnail">{url && !failed
    ? <img src={url} alt="" loading="lazy" onError={() => setFailed(true)} />
    : <span aria-hidden="true">{kind === "video" ? "▶" : "▧"}</span>}</div>;
}

interface UploadInputProps {
  files: File[];
  onFiles: (files: File[]) => void;
  kind: "image" | "video";
  multiple?: boolean;
  label: string;
  hint: string;
  disabled?: boolean;
}

export default function UploadInput({ files, onFiles, kind, multiple = false, label, hint, disabled = false }: UploadInputProps) {
  const id = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [message, setMessage] = useState("");
  const totalSize = files.reduce((total, file) => total + file.size, 0);
  const addFiles = (incoming: File[]) => {
    if (disabled || !incoming.length) return;
    const next = multiple ? [...files] : [];
    const names = new Set(next.map((file) => file.name.toLowerCase()));
    const notices: string[] = [];
    let size = multiple ? totalSize : 0;
    for (const file of incoming) {
      const invalid = validateFile(file, kind);
      if (invalid) { notices.push(invalid); continue; }
      if (names.has(file.name.toLowerCase())) { notices.push(`${file.name}: this filename is already selected.`); continue; }
      if (next.length >= (multiple ? MAX_IMAGES : 1)) { notices.push(multiple ? `Choose at most ${MAX_IMAGES} images.` : "Only one file can be selected."); break; }
      if (size + file.size > MAX_UPLOAD_BYTES) { notices.push("Selected files exceed the 2 GiB total upload limit."); continue; }
      next.push(file);
      names.add(file.name.toLowerCase());
      size += file.size;
    }
    setMessage([...new Set(notices)].join(" "));
    if (next.length || multiple) onFiles(next);
  };
  return <div className="upload-input">
    <input id={id} aria-label={label} ref={inputRef} type="file" accept={kind === "image" ? ".jpg,.jpeg,.png,.webp,.tif,.tiff,.bmp" : ".mp4,.mov,.webm,.mkv,.avi,.m4v"} multiple={multiple} disabled={disabled} hidden onChange={(event) => { addFiles(Array.from(event.target.files || [])); event.target.value = ""; }} />
    <button type="button" className={`drop-zone ${multiple ? "multi" : ""} ${dragOver ? "drag-over" : ""} ${files.length ? "has-file" : ""}`} disabled={disabled} onClick={() => inputRef.current?.click()}
      onDragOver={(event) => { event.preventDefault(); if (!disabled) setDragOver(true); }}
      onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragOver(false); }}
      onDrop={(event) => { event.preventDefault(); setDragOver(false); addFiles(Array.from(event.dataTransfer.files)); }} aria-describedby={`${id}-hint`}>
      <span className="upload-icon"><UploadIcon /></span><span className="drop-label">{files.length ? (multiple ? "Add more images" : "Replace selected file") : label}</span><span className="upload-hint" id={`${id}-hint`}>{hint}</span><span className="upload-browse">Browse files <span aria-hidden="true">↗</span></span>
    </button>
    {message && <p className="upload-message" role="alert">{message}</p>}
    {files.length > 0 && <>
      <div className="upload-toolbar"><span>{files.length} {files.length === 1 ? "file" : "files"} selected <span className="muted">· {formatFileSize(totalSize)}</span></span><button type="button" className="btn btn-sm" onClick={() => { onFiles([]); setMessage(""); }} disabled={disabled}>Clear all</button></div>
      <ul className="upload-list">{files.map((file) => <li className="upload-file" key={`${file.name}-${file.size}-${file.lastModified}`}><FileThumbnail file={file} kind={kind} /><div className="upload-file-details"><span className="upload-file-name" title={file.name}>{file.name}</span><span className="upload-file-meta">{formatFileSize(file.size)}</span></div><button type="button" className="btn-remove" disabled={disabled} aria-label={`Remove ${file.name}`} onClick={() => onFiles(files.filter((candidate) => candidate !== file))}>×</button></li>)}</ul>
    </>}
  </div>;
}
