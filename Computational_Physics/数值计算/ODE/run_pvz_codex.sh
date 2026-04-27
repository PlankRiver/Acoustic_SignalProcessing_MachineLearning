#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
PY_FILE="$SCRIPT_DIR/Plants_vs_Zombies.py"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE"
  echo "Run: cp \"$SCRIPT_DIR/.env.example\" \"$ENV_FILE\""
  echo "Then edit $ENV_FILE and set OPENAI_API_KEY."
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is empty in $ENV_FILE"
  exit 1
fi

export PVZ_AGENT="${PVZ_AGENT:-codex}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5}"

if [[ "$PVZ_AGENT" == "codex" ]]; then
  echo "Auto backend: CODEX (model=$OPENAI_MODEL)"
else
  echo "Auto backend: HEURISTIC"
fi

python "$PY_FILE"
