import os
import glob
import h5py
import torch
import numpy as np
import torch.nn as nn
from scipy.io import loadmat


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DAS_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
PREDICT_DATA_DIR = os.path.join(DAS_ROOT, "ROISF_test")
BEST_MODEL_PATH = os.path.join(CURRENT_DIR, "best_das_transformer.pth")
LAST_MODEL_PATH = os.path.join(CURRENT_DIR, "last_das_transformer.pth")
BEST_MODEL_GLOB = os.path.join(CURRENT_DIR, "best_das_transformer_e*of*.pth")
LAST_MODEL_GLOB = os.path.join(CURRENT_DIR, "last_das_transformer_e*of*.pth")

SEQ_LEN = 300
INPUT_DIM = 100
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_roi_sf(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            signal = np.array(f['ROI_SF'])
    except OSError:
        mat_data = loadmat(file_path)
        if 'ROI_SF' not in mat_data:
            raise KeyError(f"文件 {file_path} 中未找到 ROI_SF 字段")
        signal = np.array(mat_data['ROI_SF'])

    signal = np.squeeze(signal)
    if signal.shape == (5, 3001):
        signal = signal.T
    elif signal.shape != (3001, 5):
        raise ValueError(f"文件 {file_path} 的 ROI_SF 形状异常: {signal.shape}")

    return signal.astype(np.float32)


def preprocess_signal(file_path):
    signal = _load_roi_sf(file_path)
    flat_signal = signal.flatten()
    target_size = SEQ_LEN * INPUT_DIM
    if len(flat_signal) < target_size:
        flat_signal = np.pad(flat_signal, (0, target_size - len(flat_signal)), mode='constant')
    else:
        flat_signal = flat_signal[:target_size]

    return torch.tensor(flat_signal.reshape(1, SEQ_LEN, INPUT_DIM), dtype=torch.float32)


class DasTransformerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DasTransformerClassifier, self).__init__()
        self.input_proj = nn.Linear(INPUT_DIM, D_MODEL)
        self.pos_embed = nn.Parameter(torch.zeros(1, SEQ_LEN, D_MODEL))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.classifier = nn.Linear(D_MODEL, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.input_proj(x)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        metadata = {
            'best_acc': checkpoint.get('best_acc'),
            'best_epoch': checkpoint.get('best_epoch'),
            'last_epoch': checkpoint.get('last_epoch'),
            'total_epochs': checkpoint.get('total_epochs'),
        }
    else:
        state_dict = checkpoint
        metadata = {
            'best_acc': None,
            'best_epoch': None,
            'last_epoch': None,
            'total_epochs': None,
        }

    num_classes = state_dict['classifier.weight'].shape[0]
    model = DasTransformerClassifier(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


def _resolve_model_path(model_glob, fallback_path):
    candidates = glob.glob(model_glob)
    if candidates:
        return max(candidates, key=os.path.getmtime)
    if os.path.isfile(fallback_path):
        return fallback_path
    return None


def evaluate_model(model_path, model_tag, class_names):
    model, metadata = load_model(model_path)
    total = 0
    correct = 0
    print(f"\n[{model_tag}] 开始验证")
    print(f"使用设备: {DEVICE}")
    print(f"模型文件: {model_path}")
    if metadata['total_epochs'] is not None:
        if metadata['best_epoch'] is not None:
            print(f"最佳轮次: {metadata['best_epoch']}/{metadata['total_epochs']}")
        if metadata['last_epoch'] is not None:
            print(f"最后轮次: {metadata['last_epoch']}/{metadata['total_epochs']}")
    print(f"推理目录: {PREDICT_DATA_DIR}\n")

    with torch.no_grad():
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(PREDICT_DATA_DIR, class_name)
            mat_files = sorted(glob.glob(os.path.join(class_dir, '*.mat')))
            if not mat_files:
                print(f"[跳过] {class_name}: 没有 .mat 文件")
                continue

            class_total = 0
            class_correct = 0
            for file_path in mat_files:
                inputs = preprocess_signal(file_path).unsqueeze(0).to(DEVICE)
                outputs = model(inputs)
                predicted = outputs.argmax(dim=1).item()

                total += 1
                class_total += 1
                if predicted == class_idx:
                    correct += 1
                    class_correct += 1

            print(f"[{class_name}] 准确: {class_correct}/{class_total}")

    overall_acc = 100.0 * correct / total if total else 0.0
    print(f"\n[{model_tag}] 总体准确率: {overall_acc:.2f}% ({correct}/{total})")


def main():
    if not os.path.isdir(PREDICT_DATA_DIR):
        raise FileNotFoundError(f"未找到推理数据目录: {PREDICT_DATA_DIR}")

    class_names = sorted(
        d for d in os.listdir(PREDICT_DATA_DIR)
        if os.path.isdir(os.path.join(PREDICT_DATA_DIR, d))
    )

    model_items = [
        (_resolve_model_path(BEST_MODEL_GLOB, BEST_MODEL_PATH), "Best 模型"),
        (_resolve_model_path(LAST_MODEL_GLOB, LAST_MODEL_PATH), "Last 模型"),
    ]

    found_any = False
    for model_path, model_tag in model_items:
        if model_path is None:
            print(f"[跳过] 未找到 {model_tag}")
            continue
        found_any = True
        evaluate_model(model_path, model_tag, class_names)

    if not found_any:
        raise FileNotFoundError("未找到可验证的模型文件（best/last 都不存在）。")


if __name__ == "__main__":
    main()