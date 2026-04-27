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
# 1. 数据预处理 (与 RNN 完全一致)
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
# 2. 定义 GRU 模型 (关键修改处)
# ==========================================
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1. 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 2. GRU 层
        # 【修改点】: 这里将 nn.RNN 换成了 nn.GRU
        # GRU 会自动处理内部的重置门(Reset Gate)和更新门(Update Gate)
        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        # 3. 输出层
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state):
        # x: (Batch, Seq_len)
        # state: (Num_layers, Batch, Hidden_size)

        # 1. Embedding
        X = self.embedding(x)  # -> (Batch, Seq_len, Embed_size)

        # 2. GRU Forward
        # Y: 所有时间步的输出 (Batch, Seq_len, Hidden_size)
        # state: 最后一个时间步的隐状态 H_t
        # 注意：GRU 的 state 也是单个张量，不像 LSTM 是 (h, c) 元组
        Y, state = self.gru(X, state)

        # 3. Reshape and Linear
        output = self.linear(Y.reshape(-1, Y.shape[-1]))

        return output, state

    def begin_state(self, batch_size, device):
        """初始化隐状态"""
        # GRU 的状态形状与 RNN 一样: (Num_layers, Batch, Hidden_size)
        return torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device)


# ==========================================
# 3. 辅助函数
# ==========================================
def predict(prefix, num_preds, model, vocab_idx, idx_vocab, device):
    state = model.begin_state(batch_size=1, device=device)
    outputs = [vocab_idx[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape(1, 1)

    # 预热
    for y in prefix[1:]:
        _, state = model(get_input(), state)
        outputs.append(vocab_idx[y])

    # 生成
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
# 4. 参数设置与数据准备
# ==========================================
BATCH_SIZE = 32
NUM_STEPS = 35
EMBED_SIZE = 128
HIDDEN_SIZE = 256
LR = 0.001  # GRU 收敛通常比 RNN 快且稳，可以尝试稍微大一点的 LR，但这里保持一致
EPOCHS = 200

corpus, vocab_size, idx_to_token, token_to_idx = load_data_time_machine(BATCH_SIZE, NUM_STEPS)
train_loader = SeqDataLoader(corpus, BATCH_SIZE, NUM_STEPS)

# 初始化模型 (这里使用的是 GRUModel)
model = GRUModel(vocab_size, EMBED_SIZE, HIDDEN_SIZE).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ==========================================
# 5. 训练循环
# ==========================================
print(f"Start Training GRU on Time Machine Dataset...")
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
        # GRU 的 state 是一个 tensor，直接 detach 即可
        # 为了兼容性保留 tuple 判断（如果是 LSTM 就需要）
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
            # 测试生成效果
            generated = predict('time traveller', 50, model, token_to_idx, idx_to_token, device)
            print(f" > Generated: {generated}")
            print("-" * 50)

total_time = time.time() - start_time
print(f"\nTraining Done! Total Time: {total_time:.2f}s")

# ==========================================
# 6. 保存模型
# ==========================================
model_path = "gru_time_machine.pth"
checkpoint = {
    'model_state_dict': model.state_dict(),
    'vocab': {'token_to_idx': token_to_idx, 'idx_to_token': idx_to_token},
    'hyper_params': {
        'vocab_size': vocab_size,
        'embed_size': EMBED_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': 1
    }
}
torch.save(checkpoint, model_path)
print(f"GRU 模型已保存至: {model_path}")