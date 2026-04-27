#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${1:-30}"

if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [interval_seconds]"
  exit 1
fi

while true; do
  "$REPO_DIR/scripts/git-sync-once.sh" || true
  sleep "$INTERVAL"
done
