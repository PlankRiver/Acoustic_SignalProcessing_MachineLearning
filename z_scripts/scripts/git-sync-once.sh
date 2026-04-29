#!/usr/bin/env bash
set -euo pipefail

# Default to MachineLearning repo root (script lives in z_scripts/scripts).
# You can override with:
#   REPO_DIR=/path/to/repo bash git-sync-once.sh
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repository: $REPO_DIR"
  exit 1
fi

# Keep branch in sync before pushing.
git pull --rebase --autostash origin main || true

# Stage everything, then aggressively unstage generated/data artifacts.
git add -A
git reset -- '*.json' '*.txt' || true
git reset -- '*.pt' '*.pth' '*.ckpt' '*.onnx' '*.h5' '*.pb' '*.npy' '*.npz' || true
git reset -- '*.tar' '*.gz' '*.zip' '*.7z' '*.rar' || true
git reset -- 'DeepLearning/DAS/Object_detection/Yolov3/pred_json/**' || true
git reset -- 'DeepLearning/DAS/Object_detection/Yolov3/runs/**' || true
git reset -- 'DeepLearning/DAS/Object_detection/Yolov3/datasets/**' || true

# Optional strict mode: only commit source-like files.
# Enable with: STRICT_SOURCE_ONLY=1 bash git-sync-once.sh
if [[ "${STRICT_SOURCE_ONLY:-0}" == "1" ]]; then
  git reset
  git add '*.py' '*.md' '*.sh' '*.yaml' '*.yml' '*.toml' '*.ini' '.gitignore' || true
fi

if ! git diff --cached --quiet; then
  git commit -m "auto-sync: $(date '+%Y-%m-%d %H:%M:%S')"
  git push origin main
else
  echo "No changes to sync."
fi
