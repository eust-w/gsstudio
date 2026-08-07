#!/usr/bin/env bash
# Fix HF cache layout: link local_dir downloads into the standard cache structure
# so that from_pretrained('repo_id') can find them.
set -euo pipefail

HF_HOME="${HF_HOME:-/models/hf-cache}"
HUB="$HF_HOME/hub"

cd "$HUB"

link_repo() {
  local repo_id="$1"
  local local_dir="$2"
  local canonical="$HUB/models--${repo_id//\//--}"

  if [ -d "$canonical/blobs" ] && [ -s "$canonical/blobs/.populated" ] 2>/dev/null; then
    echo "Already populated: $repo_id"
    return
  fi
  if [ ! -d "$local_dir" ]; then
    echo "Missing local dir: $local_dir"
    return
  fi
  mkdir -p "$canonical/snapshots/main"
  # symlink each file into the snapshot
  find "$local_dir" -type f ! -path '*/.cache/*' | while read -r f; do
    rel="${f#$local_dir/}"
    mkdir -p "$canonical/snapshots/main/$(dirname "$rel")"
    ln -sf "$f" "$canonical/snapshots/main/$rel" 2>/dev/null || true
  done
  echo "Linked $repo_id -> $canonical"
}

link_repo "Qwen/Qwen3-VL-8B-Instruct" "$HUB/Qwen--Qwen3-VL-8B-Instruct"
link_repo "Ruicheng/moge-2-vitl-normal" "$HUB/models--Ruicheng--moge-2-vitl-normal"

echo "Layout fix complete."
du -sh "$HUB"/* 2>/dev/null | sort -h
