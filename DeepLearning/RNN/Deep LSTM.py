import torch
from torch import nn
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
# 1. 数据预处理 (通用)
# ==========================================
def load_data_time_machine(batch_size, num_steps):
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
    filename = 'timemachine.txt'
    if not os.path.exists(filename):
        try:
            urllib.request.urlretrieve(url, filename)
        except:
            print("下载失败，请手动下载 timemachine.txt")
            return None, 0, [], {}

    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    tokens = [list(line) for line in lines]
    flat_tokens = [token for line in tokens for token in line]
    counter = collections.Counter(flat_tokens)
    token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    idx_to_token = ['<unk>'] + [token for token, freq in token_freqs]
    token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}
    vocab_size = len(idx_to_token)
    corpus = [token_to_idx.get(token, 0) for token in flat_tokens]
    corpus = torch.tensor(corpus, dtype=torch.long)
    return corpus, vocab_size, idx_to_token, token_to_idx


class SeqDataLoader:
    def __init__(self, corpus, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps
        offset = random.randint(0, num_steps)
        num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
        Xs = corpus[offset: offset + num_tokens]
        Ys = corpus[offset + 1: offset + 1 + num_tokens]
        self.Xs = Xs.reshape(batch_size, -1)
        self.Ys = Ys.reshape(batch_size, -1)
        self.num_batches = self.Xs.shape[1] // num_steps

    def __iter__(self):
        for i in range(0, self.num_steps * self.num_batches, self.num_steps):
            X = self.Xs[:, i: i + self.num_steps]
            Y = self.Ys[:, i: i + self.num_steps]
            yield X, Y

    def __len__(self):
        return self.num_batches


# ==========================================
# 2. 定义 Deep LSTM 模型 (深度循环网络)
# ==========================================
class DeepRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers  # 【重点】这里不再是 1 了

        # 1. 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 2. LSTM 层
        # 【修改点 A】: 设置 num_layers 参数
        # 【修改点 B】: 当层数变深时，为了防止过拟合，通常会加 dropout
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,  # 堆叠 LSTM 层
                            dropout=0.2 if num_layers > 1 else 0,  # 仅在多层时启用dropout
                            batch_first=True)

        # 3. 输出层
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state):
        # x: (Batch, Seq_len)
        # state: ((Num_layers, Batch, Hidden), (Num_layers, Batch, Hidden))
        # 注意: state 的第一维现在变成了 num_layers，不再是 1

        X = self.embedding(x)

        # PyTorch 自动处理了层与层之间的传递
        Y, state = self.lstm(X, state)

        output = self.linear(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, batch_size, device):
        """初始化隐状态"""
        # 【修改点 C】: 形状的第一维必须是 self.num_layers
        return (
            torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device),  # H
            torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device)  # C
        )


# ==========================================
# 3. 辅助函数
# ==========================================
def predict(prefix, num_preds, model, vocab_idx, idx_vocab, device):
    state = model.begin_state(batch_size=1, device=device)
    outputs = [vocab_idx[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape(1, 1)

    for y in prefix[1:]:
        _, state = model(get_input(), state)
        outputs.append(vocab_idx[y])

    for _ in range(num_preds):
        y, state = model(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([idx_vocab[i] for i in outputs])


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# ==========================================
# 4. 参数设置 (增加层数)
# ==========================================
BATCH_SIZE = 32
NUM_STEPS = 35
EMBED_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2  # 【核心修改】: 我们使用 2 层 LSTM
LR = 0.002  # 深层网络可能需要稍微大一点的学习率或更多 Epoch
EPOCHS = 200

corpus, vocab_size, idx_to_token, token_to_idx = load_data_time_machine(BATCH_SIZE, NUM_STEPS)
train_loader = SeqDataLoader(corpus, BATCH_SIZE, NUM_STEPS)

# 初始化模型
model = DeepRNNModel(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ==========================================
# 5. 训练循环 (无需修改逻辑)
# ==========================================
print(f"Start Training Deep LSTM ({NUM_LAYERS} Layers)...")
start_time = time.time()

for epoch in range(EPOCHS):
    state = model.begin_state(batch_size=BATCH_SIZE, device=device)
    total_loss = 0
    num_batches = 0

    model.train()
    for X, Y in train_loader:
        num_batches += 1
        X = X.to(device)
        Y = Y.to(device)
        y = Y.T.reshape(-1)

        # 截断梯度流
        if isinstance(state, tuple):
            state = (state[0].detach(), state[1].detach())
        else:
            state = state.detach()

        output, state = model(X, state)
        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        grad_clipping(model, 1)
        optimizer.step()
        total_loss += loss.item()

    perplexity = math.exp(total_loss / num_batches)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS} | Ppl: {perplexity:.1f} | Loss: {total_loss / num_batches:.4f}")
        with torch.no_grad():
            generated = predict('time traveller', 50, model, token_to_idx, idx_to_token, device)
            print(f" > Generated: {generated}")
            print("-" * 50)

total_time = time.time() - start_time
print(f"\nTraining Done! Total Time: {total_time:.2f}s")

# ==========================================
# 6. 保存模型
# ==========================================
model_path = "deep_lstm_time_machine.pth"
checkpoint = {
    'model_state_dict': model.state_dict(),
    'vocab': {'token_to_idx': token_to_idx, 'idx_to_token': idx_to_token},
    'hyper_params': {
        'vocab_size': vocab_size,
        'embed_size': EMBED_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS  # 保存层数信息
    }
}
torch.save(checkpoint, model_path)
print(f"Deep LSTM 模型已保存至: {model_path}")