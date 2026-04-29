#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default to MachineLearning repo root.
# You can override with:
#   REPO_DIR=/path/to/repo bash git-sync-watch.sh 30
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
INTERVAL="${1:-30}"

if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [interval_seconds]"
  exit 1
fi

if ! git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repository: $REPO_DIR"
  exit 1
fi

while true; do
  REPO_DIR="$REPO_DIR" "$SCRIPT_DIR/git-sync-once.sh" || true
  sleep "$INTERVAL"
done
