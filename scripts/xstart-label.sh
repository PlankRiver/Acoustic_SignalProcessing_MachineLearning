#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/levono/MachineLearning"
DEFAULT_TARGET="$ROOT/DeepLearning/DAS/Object_detection/DAS_object_detection/images_pro"
TARGET="${1:-$DEFAULT_TARGET}"
source "$ROOT/.venv-xany/bin/activate"
exec xanylabeling --filename "$TARGET"
