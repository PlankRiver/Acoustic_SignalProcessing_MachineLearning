import torch
from torch import nn
import torch.utils.data as Data
import time
import math
import numpy as np
import os
import re
import random
import collections
import urllib.request

# ==========================================
# 0. 设备配置
# ==========================================
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# ==========================================
# 1. 数据预处理 (复用之前的标准流程)
# ==========================================
def load_data_time_machine(batch_size, num_steps):
    # A. 下载与读取
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
    filename = 'timemachine.txt'
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

    # B. 分词 (字符级)
    tokens = [list(line) for line in lines]
    flat_tokens = [token for line in tokens for token in line]

    # C. 构建词表
    counter = collections.Counter(flat_tokens)
    token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    idx_to_token = ['<unk>'] + [token for token, freq in token_freqs]
    token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}
    vocab_size = len(idx_to_token)

    # D. 数字化 (Corpus)
    corpus = [token_to_idx.get(token, 0) for token in flat_tokens]
    corpus = torch.tensor(corpus, dtype=torch.long)

    return corpus, vocab_size, idx_to_token, token_to_idx


class SeqDataLoader:
    """顺序分区迭代器 (保留上下文记忆)"""

    def __init__(self, corpus, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps
        # 丢弃不能整除的部分
        offset = random.randint(0, num_steps)
        num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
        Xs = corpus[offset: offset + num_tokens]
        Ys = corpus[offset + 1: offset + 1 + num_tokens]  # 标签是下一个词

        # 变换形状为 (batch_size, -1)
        self.Xs = Xs.reshape(batch_size, -1)
        self.Ys = Ys.reshape(batch_size, -1)
        self.num_batches = self.Xs.shape[1] // num_steps

    def __iter__(self):
        # 每次产出 (X, Y)
        for i in range(0, self.num_steps * self.num_batches, self.num_steps):
            X = self.Xs[:, i: i + self.num_steps]
            Y = self.Ys[:, i: i + self.num_steps]
            yield X, Y

    def __len__(self):
        return self.num_batches


# ==========================================
# 2. 定义 RNN 模型
# ==========================================
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1. 嵌入层: 将数字索引转为稠密向量 (Batch, Seq, Embed)
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 2. RNN 层 (也可以换成 nn.LSTM 或 nn.GRU)
        # batch_first=True 表示输入形状为 (Batch, Seq, Feature)
        self.rnn = nn.RNN(input_size=embed_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        # 3. 输出层: 将隐状态映射回词表大小
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state):
        # x: (Batch, Seq_len)
        # state: (Num_layers, Batch, Hidden_size)

        # 1. Embedding
        X = self.embedding(x)  # -> (Batch, Seq_len, Embed_size)

        # 2. RNN Forward
        # Y: 所有时间步的输出 (Batch, Seq_len, Hidden_size)
        # state: 最后一个时间步的隐状态 (用于传给下一个Batch)
        Y, state = self.rnn(X, state)

        # 3. Reshape and Linear
        # 我们要把 (Batch, Seq) 展平放到 Linear 里去算
        # output -> (Batch * Seq_len, Vocab_size)
        output = self.linear(Y.reshape(-1, Y.shape[-1]))

        return output, state

    def begin_state(self, batch_size, device):
        """初始化隐状态"""
        # RNN/GRU 的状态形状: (Num_layers, Batch, Hidden_size)
        # 如果是 LSTM，则需要返回 (h_0, c_0) 元组
        return torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device)


# ==========================================
# 3. 辅助函数 (预测与裁剪)
# ==========================================
def predict(prefix, num_preds, model, vocab_idx, idx_vocab, device):
    """给定开头(prefix)，让模型接着写(num_preds个字)"""
    state = model.begin_state(batch_size=1, device=device)
    outputs = [vocab_idx[prefix[0]]]

    # 1. 预热 (Warm-up): 把 prefix 喂进去更新 state，但不记录结果
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape(1, 1)
    for y in prefix[1:]:
        _, state = model(get_input(), state)
        outputs.append(vocab_idx[y])

    # 2. 生成: 拿着更新好的 state 开始预测
    for _ in range(num_preds):
        y, state = model(get_input(), state)
        # y 是 logits，取 argmax 得到最可能的下一个词索引
        outputs.append(int(y.argmax(dim=1).reshape(1)))

    return ''.join([idx_vocab[i] for i in outputs])


def grad_clipping(net, theta):
    """梯度裁剪：防止梯度爆炸"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

# ==========================================
# 4. 参数设置与数据准备
# ==========================================
BATCH_SIZE = 32
NUM_STEPS = 35  # 时间步长 (序列长度)
EMBED_SIZE = 128  # 嵌入维度
HIDDEN_SIZE = 256  # 隐状态维度
LR = 0.001
EPOCHS = 200  # RNN 需要多训练一会

# 加载数据
corpus, vocab_size, idx_to_token, token_to_idx = load_data_time_machine(BATCH_SIZE, NUM_STEPS)
train_loader = SeqDataLoader(corpus, BATCH_SIZE, NUM_STEPS)

# 初始化模型
model = RNNModel(vocab_size, EMBED_SIZE, HIDDEN_SIZE).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ==========================================
# 5. 训练循环
# ==========================================
print(f"Start Training RNN on Time Machine Dataset...")
start_time = time.time()

for epoch in range(EPOCHS):
    # 1. 每个 Epoch 开始时，初始化隐状态
    state = model.begin_state(batch_size=BATCH_SIZE, device=device)
    total_loss = 0
    num_batches = 0

    model.train()
    for X, Y in train_loader:
        # X: (Batch, Seq), Y: (Batch, Seq)
        num_batches += 1
        X = X.to(device)
        Y = Y.to(device)
        y = Y.T.reshape(-1)  # 展平标签以匹配 output: (Batch * Seq)

        # 【重点】截断梯度流
        # 如果不 detach，PyTorch 会试图反向传播到上一个 Batch 甚至上上个...
        # 导致计算图无限增大最终内存溢出
        if isinstance(state, tuple):  # LSTM case
            state = (state[0].detach(), state[1].detach())
        else:
            state = state.detach()

        # 前向
        # output shape: (Batch * Seq, Vocab_size)
        output, state = model(X, state)

        # 计算损失
        loss = loss_fn(output, y)

        # 反向
        optimizer.zero_grad()
        loss.backward()

        # 【重点】梯度裁剪 (在 step 之前)
        grad_clipping(model, 1)

        optimizer.step()
        total_loss += loss.item()

    # 计算困惑度 Perplexity (exp(loss))，这是 NLP 的标准指标
    perplexity = math.exp(total_loss / num_batches)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS} | Ppl: {perplexity:.1f} | Loss: {total_loss / num_batches:.4f}")
        # 现场生成一段文本来看看效果
        with torch.no_grad():
            generated = predict('time traveller', 50, model, token_to_idx, idx_to_token, device)
            print(f" > Generated: {generated}")
            print("-" * 50)

total_time = time.time() - start_time
print(f"\nTraining Done! Total Time: {total_time:.2f}s")

# ==========================================
# 6. 保存模型 (包含权重 + 词表 + 超参数)
# ==========================================
model_path = "rnn_time_machine.pth"

checkpoint = {
    # 1. 模型权重
    'model_state_dict': model.state_dict(),

    # 2. 核心：词表映射 (没有这个，预测时就是瞎猜)
    'vocab': {
        'token_to_idx': token_to_idx,
        'idx_to_token': idx_to_token
    },

    # 3. 模型架构超参数 (用于重建模型实例)
    'hyper_params': {
        'vocab_size': vocab_size,
        'embed_size': EMBED_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': 1  # 我们的模型默认是1层
    }
}

torch.save(checkpoint, model_path)
print(f"模型已完整保存至: {model_path}")