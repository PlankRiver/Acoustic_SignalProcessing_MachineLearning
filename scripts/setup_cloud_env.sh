#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash setup_cloud_env.sh [lite|full] [env_name]
# Examples:
#   bash setup_cloud_env.sh lite linux_ai
#   bash setup_cloud_env.sh full linux_ai

MODE="${1:-lite}"
ENV_NAME="${2:-linux_ai}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
ENV_FILE="${PROJECT_DIR}/environment.yml"

if [[ "${MODE}" != "lite" && "${MODE}" != "full" ]]; then
  echo "[ERROR] MODE must be lite or full"
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda command not found. Please install Miniconda/Anaconda first."
  exit 1
fi

eval "$(conda shell.bash hook)"

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"
}

install_torch_cu124() {
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
}

install_lite_project_deps() {
  python -m pip install \
    numpy==1.26.4 \
    scipy==1.16.3 \
    h5py \
    pillow \
    matplotlib \
    scikit-learn \
    pandas \
    seaborn \
    tqdm \
    jupyter \
    ipykernel \
    opencv-python
}

if [[ "${MODE}" == "full" ]]; then
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[ERROR] Missing ${ENV_FILE}"
    exit 1
  fi

  if env_exists; then
    echo "[INFO] Updating existing env ${ENV_NAME} from environment.yml"
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
  else
    echo "[INFO] Creating env ${ENV_NAME} from environment.yml"
    conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
  fi
else
  if env_exists; then
    echo "[INFO] Reusing existing env ${ENV_NAME}"
  else
    echo "[INFO] Creating env ${ENV_NAME} with Python 3.11.15"
    conda create -n "${ENV_NAME}" python=3.11.15 pip -y
  fi

  conda activate "${ENV_NAME}"
  echo "[INFO] Installing PyTorch 2.6.0 + cu124"
  install_torch_cu124
  echo "[INFO] Installing lightweight project dependencies"
  install_lite_project_deps
fi

conda activate "${ENV_NAME}"
echo "[INFO] Verifying environment"
python - <<'PY'
import sys
import torch

print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "[DONE] Environment is ready."
echo "To activate later: conda activate ${ENV_NAME}"
