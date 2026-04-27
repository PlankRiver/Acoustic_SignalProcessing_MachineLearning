import os
import glob
import copy
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat

# ==========================================
# 1. 全局配置与路径设置
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# DAS 根目录
DAS_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
# 训练集与验证集路径
TRAIN_DATA_DIR = os.path.join(DAS_ROOT, "ROISF")
# 这里的验证集由独立脚本移动生成，不在训练时随机切分
VALIDATION_DATA_DIR = os.path.join(DAS_ROOT, "ROISF_test")
# 预处理后产生的数据包路径 (存储在当前代码目录下)
TRAIN_PROCESSED_FILE = os.path.join(CURRENT_DIR, "preprocessed_roisf_train.pt")
VALIDATION_PROCESSED_FILE = os.path.join(CURRENT_DIR, "preprocessed_roisf_validation.pt")
# 模型保存路径
MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "best_das_resnet18.pth")
FINAL_MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "last_das_resnet18.pth")

BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_epoch_suffix_path(base_path, epoch, total_epochs):
    base, ext = os.path.splitext(base_path)
    return f"{base}_e{epoch:02d}of{total_epochs:02d}{ext}"


# ==========================================
# 2. 离线预处理模块 (CPU密集型)
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

    # 统一成 (3001, 5)，不同版本 mat 文件的存储方向不同
    if signal.shape == (5, 3001):
        signal = signal.T
    elif signal.shape != (3001, 5):
        raise ValueError(f"文件 {file_path} 的 ROI_SF 形状异常: {signal.shape}")

    return signal.astype(np.float32)


def preprocess_and_save():
    """遍历所有 mat 文件，处理成张量并打包保存"""
    preprocess_and_save_from_dir(TRAIN_DATA_DIR, TRAIN_PROCESSED_FILE)


def preprocess_and_save_from_dir(data_dir, processed_file):
    """遍历指定目录下所有 mat 文件，处理成张量并打包保存"""
    print(f"[{'预处理'}] 开始扫描目录: {data_dir} ...")
    
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    classes.sort()
    num_classes = len(classes)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    all_signals = []
    all_labels = []
    class_counts = np.zeros(num_classes)
    
    target_size = 300 * 100
    total_files = 0

    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        mat_files = glob.glob(os.path.join(cls_dir, '*.mat'))
        total_files += len(mat_files)
        
        for file_path in mat_files:
            # 1. 兼容读取 MATLAB v7.3 和 v5 格式文件
            signal = _load_roi_sf(file_path)
            
            # 2. 展平与对齐
            flat_signal = signal.flatten()
            if len(flat_signal) < target_size:
                pad_width = target_size - len(flat_signal)
                padded_signal = np.pad(flat_signal, (0, pad_width), mode='constant')
            else:
                padded_signal = flat_signal[:target_size]
                
            # 3. Reshape 为 (1, 300, 100)
            reshaped_signal = padded_signal.reshape(1, 300, 100)
            
            all_signals.append(reshaped_signal)
            all_labels.append(class_to_idx[cls_name])
            
        class_counts[class_to_idx[cls_name]] = len(mat_files)
        print(f"  - [{cls_name}] 处理完成, 数量: {len(mat_files)}")
        
    print(f"[{'预处理'}] 所有文件处理完毕，共计 {total_files} 个样本。")
    
    # 转换为 PyTorch 张量
    tensor_signals = torch.tensor(np.array(all_signals))
    tensor_labels = torch.tensor(np.array(all_labels), dtype=torch.long)
    
    # 计算权重缓解样本不平衡
    class_weights = total_files / (num_classes * class_counts)
    tensor_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # 打包保存到磁盘
    save_dict = {
        'signals': tensor_signals,
        'labels': tensor_labels,
        'weights': tensor_weights,
        'num_classes': num_classes,
        'class_names': classes
    }
    torch.save(save_dict, processed_file)
    print(f"[{'预处理'}] 数据包已保存至: {processed_file}\n")


# ==========================================
# 3. 数据加载模块 (直接从内存加载打包好的数据)
# ==========================================
def load_prepared_data():
    # 训练集和验证集分别来自两个目录，避免随机切分导致的数据泄漏
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
    
    # 注意：因为数据已经在内存里了，不需要 num_workers 多进程读取了
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, validation_loader, num_classes, class_weights


# ==========================================
# 4. 手工构建 ResNet18
# ==========================================
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18Custom(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18Custom, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
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
    
    # 加载预处理好的数据
    train_loader, validation_loader, num_classes, class_weights = load_prepared_data()
    
    model = ResNet18Custom(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = -1.0
    best_epoch = -1
    best_state_dict = None
    
    print("\n🚀 开始高能训练...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
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
        validation_acc = 100. * correct / total
        
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Train Loss: {train_loss_avg:.4f} | Validation Loss: {validation_loss_avg:.4f} | Validation Acc: {validation_acc:.2f}%")
        
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

    print(f"✅ 训练大功告成！最高准确率: {best_acc:.2f}% (Epoch {best_epoch}/{EPOCHS})")
    print(f"📂 最佳验证模型(稳定名): {MODEL_SAVE_PATH}")
    print(f"📂 最佳验证模型(轮次后缀): {best_suffix_path}")
    print(f"📂 最后一轮模型(稳定名): {FINAL_MODEL_SAVE_PATH}")
    print(f"📂 最后一轮模型(轮次后缀): {final_suffix_path}")

if __name__ == "__main__":
    main()