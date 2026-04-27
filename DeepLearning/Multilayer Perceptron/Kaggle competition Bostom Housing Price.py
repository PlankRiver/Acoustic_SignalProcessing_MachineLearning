import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ==========================================
# 1. 工具函数：下载数据
# ==========================================
def download_url(url, folder='data', sha1_hash=None):
    if not os.path.exists(folder):
        os.makedirs(folder)
    fname = os.path.join(folder, url.split('/')[-1])

    if os.path.exists(fname):
        return fname

    print(f'Downloading {fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


# 数据集 URL (源自 D2L)
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
TRAIN_URL = DATA_URL + 'kaggle_house_pred_train.csv'
TEST_URL = DATA_URL + 'kaggle_house_pred_test.csv'

train_file = download_url(TRAIN_URL)
test_file = download_url(TEST_URL)

# ==========================================
# 2. 数据预处理 (Pandas)
# ==========================================
# 读取数据
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

print(f"Train shape: {train_data.shape}")
print(f"Test shape: {test_data.shape}")

# 移除 ID 列，并将训练集和测试集的特征合并处理
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 标准化数值特征
# 筛选出数值类型的列
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# (x - mean) / std
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 将数值特征的 NaN 填充为 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 处理离散特征：One-Hot 编码 (dummy_na=True 会将 NaN 也视为一个类别)
all_features = pd.get_dummies(all_features, dummy_na=True)

print(f"Preprocessed features shape: {all_features.shape}")

# ==========================================
# 3. 转换为 PyTorch Tensor
# ==========================================
n_train = train_data.shape[0]

# 训练集特征
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
# 测试集特征 (用于最后生成提交文件)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
# 训练集标签 (取对数，因为比赛评估的是 Log 误差)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32
)
# 对标签取 Log
train_labels = torch.log(train_labels)

# ==========================================
# 4. 定义模型和损失函数
# ==========================================
# 这里的 Loss 使用 MSE，因为 label 已经是 Log 后的价格了
# 所以 MSE(log_pred, log_true) 等同于 RMSLE
loss_fn = nn.MSELoss()


def get_net(num_inputs):
    # 简单的线性回归模型
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    return net


def log_rmse(net, features, labels):
    # 为了在打印日志时看懂误差，我们计算 RMSE
    # 此时 features 输入模型得到的是 log_price_hat
    with torch.no_grad():
        clipped_preds = torch.clamp(net(features), 1, float('inf'))
        rmse = torch.sqrt(loss_fn(clipped_preds, labels))
    return rmse.item()


# ==========================================
# 5. 训练函数
# ==========================================
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = TensorDataset(train_features, train_labels)
    train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 这里使用 Adam 优化器，它对于房价预测这种任务通常比 SGD 更稳健
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)

    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            output = net(X)
            l = loss_fn(output, y)
            l.backward()
            optimizer.step()

        # 记录每个 Epoch 结束后的 Log RMSE
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))

    return train_ls, test_ls


# ==========================================
# 6. K-Fold 交叉验证
# ==========================================
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    trained_nets = []  # 保存每一折训练好的模型

    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])

        # 训练
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)

        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        trained_nets.append(net)

        if i == 0:
            print(f'Fold {i + 1}, Train log rmse: {float(train_ls[-1]):f}, '
                  f'Valid log rmse: {float(valid_ls[-1]):f}')

    print(f'{k}-fold validation: avg train log rmse: {train_l_sum / k:f}, '
          f'avg valid log rmse: {valid_l_sum / k:f}')

    return trained_nets


# ==========================================
# 7. 开始训练与预测
# ==========================================
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64

# 执行 K-Fold 训练，并获取所有折的模型
print("Start training...")
nets = k_fold(k, train_features, train_labels, num_epochs, lr,
              weight_decay, batch_size)

# ==========================================
# 8. 集成预测 (Ensemble) 与生成提交文件
# ==========================================
print("Generating submission...")

# 对每个模型的预测结果取平均
preds = []
for net in nets:
    net.eval()  # 评估模式
    with torch.no_grad():
        preds.append(net(test_features))

# 1. 拼接所有模型的预测结果 (shape: [n_test, k])
# 2. 取平均 (mean)
# 3. 指数还原 (exp)，因为之前我们是在 Log 空间训练的
ensemble_preds = torch.exp(torch.cat(preds, dim=1)).mean(dim=1)

# 保存为 CSV
submission = pd.DataFrame({
    'Id': test_data.Id,
    'SalePrice': ensemble_preds.detach().numpy()
})
submission.to_csv('submission.csv', index=False)
print("submission.csv saved!")