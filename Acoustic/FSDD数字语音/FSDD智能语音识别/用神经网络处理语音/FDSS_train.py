import os
import torch
import torch.nn as nn
import torch.optim as optim  # 引入优化器
import torchaudio
from torch.utils.data import Dataset, DataLoader
# 移除 from d2l import torch as d2l
from FDSS_model_def import AudioCNN  # 确保这个文件里的 AudioCNN 也是去 d2l 之后的版本

BASE_DIR = r'/Acoustic/FSDD数字语音/dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
MODEL_SAVE_PATH = 'fsdd_model.pth'


# --- 1. 数据集定义 (保持不变，原本就是纯 PyTorch) ---
class FSDDDataset(Dataset):
    def __init__(self, data_dir, target_sample_rate=8000, max_length=8000):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        self.target_sample_rate = target_sample_rate
        self.max_length = max_length
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=target_sample_rate, n_mfcc=13,
            melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 23, 'center': False}
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        try:
            label = int(self.files[idx].split('_')[0])
        except ValueError:
            label = 0  # 容错
        waveform, sample_rate = torchaudio.load(file_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)(waveform)

        if waveform.shape[1] < self.max_length:
            padding = torch.zeros(1, self.max_length - waveform.shape[1])
            waveform = torch.cat((waveform, padding), dim=1)
        elif waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]

        mfcc = self.mfcc_transform(waveform)
        return mfcc, label


# 移除 class FSDDDataModule...

# --- 2. 辅助函数：评估准确率 (替代 d2l 的可视化部分) ---
def evaluate_accuracy(model, data_loader, device):
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


# --- 3. 执行训练 (主逻辑) ---
if __name__ == '__main__':
    if not os.path.exists(TRAIN_DIR):
        print(f"错误：找不到路径 {TRAIN_DIR}")
    else:
        # ---------------- 配置超参数 ----------------
        BATCH_SIZE = 32
        LEARNING_RATE = 0.005
        EPOCHS = 85

        # 自动选择设备 (GPU 优先)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"正在使用设备: {device}")

        # ---------------- 准备数据 (替代 DataModule) ----------------
        # 直接实例化 DataLoader
        train_dataset = FSDDDataset(TRAIN_DIR)
        test_dataset = FSDDDataset(TEST_DIR)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # ---------------- 准备模型 ----------------
        # 实例化模型并移动到 GPU
        model = AudioCNN(lr=LEARNING_RATE).to(device)

        # 定义损失函数 (分类任务通常用交叉熵)
        criterion = nn.CrossEntropyLoss()

        # 定义优化器 (Adam 通常比 SGD 收敛快，这里用 Adam)
        # 注意：这里我们使用模型参数和上面定义的 LEARNING_RATE
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # ---------------- 训练循环 (替代 trainer.fit) ----------------
        print("开始训练...")

        for epoch in range(EPOCHS):
            model.train()  # 切换到训练模式
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(train_loader):
                # 1. 数据搬家到 GPU
                inputs, labels = inputs.to(device), labels.to(device)

                # 2. 梯度清零
                optimizer.zero_grad()

                # 3. 前向传播 (LazyLinear 会在这里第一次初始化)
                outputs = model(inputs)

                # 4. 计算损失
                loss = criterion(outputs, labels)

                # 5. 反向传播
                loss.backward()

                # 6. 更新参数
                optimizer.step()

                running_loss += loss.item()

            # 每 5 个 epoch 打印一次状态
            if (epoch + 1) % 5 == 0:
                # 计算测试集准确率
                acc = evaluate_accuracy(model, test_loader, device)
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}, Test Acc: {acc:.2%}")

        # ---------------- 保存模型 ----------------
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"训练完成！模型参数已保存至: {os.path.abspath(MODEL_SAVE_PATH)}")