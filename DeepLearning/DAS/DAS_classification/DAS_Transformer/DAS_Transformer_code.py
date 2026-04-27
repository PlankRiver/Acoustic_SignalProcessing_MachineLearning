import os
import glob
import copy
import h5py
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat


# ==========================================
# 1. 全局配置与路径设置
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DAS_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))

TRAIN_DATA_DIR = os.path.join(DAS_ROOT, "ROISF")
VALIDATION_DATA_DIR = os.path.join(DAS_ROOT, "ROISF_test")

TRAIN_PROCESSED_FILE = os.path.join(CURRENT_DIR, "preprocessed_roisf_train.pt")
VALIDATION_PROCESSED_FILE = os.path.join(CURRENT_DIR, "preprocessed_roisf_validation.pt")

MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "best_das_transformer.pth")
FINAL_MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "last_das_transformer.pth")

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

# 序列长度和每步特征维度来自 reshape(1, 300, 100)
SEQ_LEN = 300
INPUT_DIM = 100

# 针对当前数据规模，使用中等容量 Transformer，兼顾表达能力和稳定性
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_epoch_suffix_path(base_path, epoch, total_epochs):
	base, ext = os.path.splitext(base_path)
	return f"{base}_e{epoch:02d}of{total_epochs:02d}{ext}"


# ==========================================
# 2. 离线预处理模块
# ==========================================
def _load_roi_sf(file_path):
	"""兼容读取 v7.3(HDF5) 和 v5 MATLAB mat 文件中的 ROI_SF。"""
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


def preprocess_and_save_from_dir(data_dir, processed_file):
	"""遍历指定目录下所有 mat 文件，处理成张量并打包保存。"""
	print(f"[预处理] 开始扫描目录: {data_dir} ...")

	classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
	classes.sort()
	num_classes = len(classes)
	class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

	all_signals = []
	all_labels = []
	class_counts = np.zeros(num_classes)

	target_size = SEQ_LEN * INPUT_DIM
	total_files = 0

	for cls_name in classes:
		cls_dir = os.path.join(data_dir, cls_name)
		mat_files = glob.glob(os.path.join(cls_dir, '*.mat'))
		total_files += len(mat_files)

		for file_path in mat_files:
			signal = _load_roi_sf(file_path)

			flat_signal = signal.flatten()
			if len(flat_signal) < target_size:
				pad_width = target_size - len(flat_signal)
				padded_signal = np.pad(flat_signal, (0, pad_width), mode='constant')
			else:
				padded_signal = flat_signal[:target_size]

			reshaped_signal = padded_signal.reshape(1, SEQ_LEN, INPUT_DIM)
			all_signals.append(reshaped_signal)
			all_labels.append(class_to_idx[cls_name])

		class_counts[class_to_idx[cls_name]] = len(mat_files)
		print(f"  - [{cls_name}] 处理完成, 数量: {len(mat_files)}")

	print(f"[预处理] 所有文件处理完毕，共计 {total_files} 个样本。")

	tensor_signals = torch.tensor(np.array(all_signals), dtype=torch.float32)
	tensor_labels = torch.tensor(np.array(all_labels), dtype=torch.long)

	class_weights = total_files / (num_classes * class_counts)
	tensor_weights = torch.tensor(class_weights, dtype=torch.float32)

	save_dict = {
		'signals': tensor_signals,
		'labels': tensor_labels,
		'weights': tensor_weights,
		'num_classes': num_classes,
		'class_names': classes,
	}
	torch.save(save_dict, processed_file)
	print(f"[预处理] 数据包已保存至: {processed_file}\n")


# ==========================================
# 3. 数据加载模块
# ==========================================
def load_prepared_data():
	if not os.path.isdir(VALIDATION_DATA_DIR):
		raise FileNotFoundError(
			f"未找到固定验证集目录: {VALIDATION_DATA_DIR}。请先运行 DAS 目录下的抽样脚本。"
		)

	if not os.path.exists(TRAIN_PROCESSED_FILE):
		print("未检测到训练集预处理数据包，正在首次生成...")
		preprocess_and_save_from_dir(TRAIN_DATA_DIR, TRAIN_PROCESSED_FILE)

	if not os.path.exists(VALIDATION_PROCESSED_FILE):
		print("未检测到验证集预处理数据包，正在首次生成...")
		preprocess_and_save_from_dir(VALIDATION_DATA_DIR, VALIDATION_PROCESSED_FILE)

	print(f"正在从 {TRAIN_PROCESSED_FILE} 和 {VALIDATION_PROCESSED_FILE} 加载数据包到内存...")
	train_dict = torch.load(TRAIN_PROCESSED_FILE)
	validation_dict = torch.load(VALIDATION_PROCESSED_FILE)

	train_signals = train_dict['signals']
	train_labels = train_dict['labels']
	class_weights = train_dict['weights'].to(DEVICE)
	num_classes = train_dict['num_classes']

	validation_signals = validation_dict['signals']
	validation_labels = validation_dict['labels']

	if validation_dict['num_classes'] != num_classes:
		raise ValueError("训练集和验证集的类别数量不一致，请检查抽样脚本输出。")

	train_dataset = TensorDataset(train_signals, train_labels)
	validation_dataset = TensorDataset(validation_signals, validation_labels)

	print(f"加载成功！训练集: {len(train_dataset)}，验证集: {len(validation_dataset)}")

	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

	return train_loader, validation_loader, num_classes, class_weights


# ==========================================
# 4. Transformer 模型定义
# ==========================================
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

		nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

	def forward(self, x):
		# x: [B, 1, 300, 100]
		x = x.squeeze(1)
		x = self.input_proj(x)
		x = x + self.pos_embed
		x = self.encoder(x)
		x = self.norm(x)

		# 对时间维做均值池化，得到全局表示
		x = x.mean(dim=1)
		x = self.classifier(x)
		return x


# ==========================================
# 5. 训练与验证主流程
# ==========================================
def main():
	print("=" * 50)
	print(f"设备环境: {DEVICE}")
	if DEVICE.type == 'cuda':
		print(f"显卡型号: {torch.cuda.get_device_name(0)}")
	print("=" * 50)

	train_loader, validation_loader, num_classes, class_weights = load_prepared_data()

	model = DasTransformerClassifier(num_classes=num_classes).to(DEVICE)
	criterion = nn.CrossEntropyLoss(weight=class_weights)
	optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

	best_acc = -1.0
	best_epoch = -1
	best_state_dict = None

	print("\n开始 Transformer 训练...")
	for epoch in range(EPOCHS):
		model.train()
		train_loss = 0.0

		for inputs, targets in train_loader:
			inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()

			train_loss += loss.item()

		model.eval()
		validation_loss = 0.0
		correct = 0
		total = 0

		with torch.no_grad():
			for inputs, targets in validation_loader:
				inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
				outputs = model(inputs)
				loss = criterion(outputs, targets)
				validation_loss += loss.item()

				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()

		train_loss_avg = train_loss / len(train_loader)
		validation_loss_avg = validation_loss / len(validation_loader)
		validation_acc = 100.0 * correct / total

		print(
			f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
			f"Train Loss: {train_loss_avg:.4f} | "
			f"Validation Loss: {validation_loss_avg:.4f} | "
			f"Validation Acc: {validation_acc:.2f}%"
		)

		if validation_acc > best_acc:
			best_acc = validation_acc
			best_epoch = epoch + 1
			best_state_dict = copy.deepcopy(model.state_dict())
			best_checkpoint = {
				'state_dict': best_state_dict,
				'best_acc': best_acc,
				'best_epoch': best_epoch,
				'total_epochs': EPOCHS,
			}
			torch.save(best_checkpoint, MODEL_SAVE_PATH)
			print(f"  --> 已保存新模型 (Acc: {best_acc:.2f}%)")

	print("\n" + "=" * 50)
	last_epoch = EPOCHS
	last_checkpoint = {
		'state_dict': model.state_dict(),
		'best_acc': best_acc,
		'best_epoch': best_epoch,
		'last_epoch': last_epoch,
		'total_epochs': EPOCHS,
	}
	torch.save(last_checkpoint, FINAL_MODEL_SAVE_PATH)
	final_suffix_path = _build_epoch_suffix_path(FINAL_MODEL_SAVE_PATH, last_epoch, EPOCHS)
	torch.save(last_checkpoint, final_suffix_path)

	if best_state_dict is not None:
		best_suffix_path = _build_epoch_suffix_path(MODEL_SAVE_PATH, best_epoch, EPOCHS)
		best_checkpoint = {
			'state_dict': best_state_dict,
			'best_acc': best_acc,
			'best_epoch': best_epoch,
			'last_epoch': last_epoch,
			'total_epochs': EPOCHS,
		}
		torch.save(best_checkpoint, best_suffix_path)
	else:
		best_suffix_path = MODEL_SAVE_PATH

	print(f"训练完成！最高验证准确率: {best_acc:.2f}% (Epoch {best_epoch}/{EPOCHS})")
	print(f"最佳验证模型(稳定名): {MODEL_SAVE_PATH}")
	print(f"最佳验证模型(轮次后缀): {best_suffix_path}")
	print(f"最后一轮模型(稳定名): {FINAL_MODEL_SAVE_PATH}")
	print(f"最后一轮模型(轮次后缀): {final_suffix_path}")


if __name__ == "__main__":
	main()
