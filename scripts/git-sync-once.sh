#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Keep branch in sync before pushing.
git pull --rebase --autostash origin main || true

git add -A
if ! git diff --cached --quiet; then
  git commit -m "auto-sync: $(date '+%Y-%m-%d %H:%M:%S')"
  git push origin main
else
  echo "No changes to sync."
fi
