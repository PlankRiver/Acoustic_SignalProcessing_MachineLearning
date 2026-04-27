import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

# ====================== 1. 配置参数 ======================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SOS_TOKEN = 0  # 句子开始标记
EOS_TOKEN = 1  # 句子结束标记
MAX_LENGTH = 10  # 最大序列长度


# ====================== 2. 注意力层（Bahdanau Attention） ======================
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        """
        Args:
            query: 解码器当前隐藏状态 (batch_size, hidden_size)
            keys: 编码器所有隐藏状态 (batch_size, seq_len, hidden_size)
        Returns:
            attention_weights: 注意力权重 (batch_size, seq_len)
            context_vector: 上下文向量 (batch_size, hidden_size)
        """
        # 扩展维度: query (batch_size, 1, hidden_size)
        query = query.unsqueeze(1)

        # 计算注意力分数: (batch_size, seq_len, 1)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))

        # 归一化得到注意力权重: (batch_size, seq_len)
        attention_weights = F.softmax(scores, dim=1).squeeze(2)

        # 计算上下文向量: (batch_size, hidden_size)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)

        return context_vector, attention_weights


# ====================== 3. 编码器 ======================
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 嵌入层
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 双向GRU
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        # 双向输出拼接后映射回hidden_size
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        """
        Args:
            x: 输入序列 (batch_size, seq_len)
        Returns:
            outputs: 编码器所有时间步输出 (batch_size, seq_len, hidden_size)
            hidden: 编码器最终隐藏状态 (num_layers, batch_size, hidden_size)
        """
        # 嵌入: (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        embedded = self.embedding(x)

        # GRU前向传播
        outputs, hidden = self.gru(embedded)

        # 处理双向输出: (batch_size, seq_len, 2*hidden_size) -> (batch_size, seq_len, hidden_size)
        outputs = self.fc(outputs)

        # 处理双向隐藏状态: (2*num_layers, batch_size, hidden_size) -> (num_layers, batch_size, hidden_size)
        hidden = torch.cat((hidden[0::2], hidden[1::2]), dim=2)  # 拼接双向隐藏状态
        hidden = self.fc(hidden)

        return outputs, hidden


# ====================== 4. 解码器（带注意力） ======================
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 嵌入层
        self.embedding = nn.Embedding(output_size, hidden_size)
        # 注意力层
        self.attention = BahdanauAttention(hidden_size)
        # GRU（输入包含：嵌入向量 + 上下文向量）
        self.gru = nn.GRU(
            hidden_size * 2, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        # 输出层（预测下一个token）
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, encoder_outputs):
        """
        Args:
            x: 解码器当前输入 (batch_size, 1)
            hidden: 解码器上一步隐藏状态 (num_layers, batch_size, hidden_size)
            encoder_outputs: 编码器所有输出 (batch_size, seq_len, hidden_size)
        Returns:
            output: 预测概率 (batch_size, output_size)
            hidden: 解码器当前隐藏状态 (num_layers, batch_size, hidden_size)
            attention_weights: 注意力权重 (batch_size, seq_len)
        """
        # 嵌入 + Dropout: (batch_size, 1) -> (batch_size, 1, hidden_size)
        embedded = self.dropout(self.embedding(x))

        # 计算注意力上下文向量
        context_vector, attention_weights = self.attention(hidden[-1], encoder_outputs)

        # 拼接嵌入向量和上下文向量: (batch_size, 1, 2*hidden_size)
        rnn_input = torch.cat((embedded, context_vector.unsqueeze(1)), dim=2)

        # GRU前向传播
        output, hidden = self.gru(rnn_input, hidden)

        # 输出层: (batch_size, 1, hidden_size) -> (batch_size, output_size)
        output = self.fc_out(output.squeeze(1))

        return output, hidden, attention_weights


# ====================== 5. Seq2Seq模型（整合编码和解码） ======================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: 源序列 (batch_size, src_len)
            trg: 目标序列 (batch_size, trg_len)
            teacher_forcing_ratio: 是否使用Teacher Forcing
        Returns:
            outputs: 解码器所有时间步预测 (batch_size, trg_len-1, output_size)
            attention_weights: 所有时间步注意力权重 (batch_size, trg_len-1, src_len)
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        # 初始化输出存储
        outputs = torch.zeros(batch_size, trg_len - 1, trg_vocab_size).to(self.device)
        attention_weights = torch.zeros(batch_size, trg_len - 1, src.shape[1]).to(self.device)

        # 编码器前向传播
        encoder_outputs, hidden = self.encoder(src)

        # 解码器初始输入：SOS_TOKEN
        input_token = trg[:, 0].unsqueeze(1)  # (batch_size, 1)

        # 逐步解码
        for t in range(1, trg_len):
            # 解码器前向传播
            output, hidden, attn_weights = self.decoder(input_token, hidden, encoder_outputs)

            # 存储输出和注意力权重
            outputs[:, t - 1, :] = output
            attention_weights[:, t - 1, :] = attn_weights

            # 决定是否使用Teacher Forcing
            teacher_force = random.random() < teacher_forcing_ratio
            # 取预测概率最大的token
            top1 = output.argmax(1).unsqueeze(1)
            # 更新输入（Teacher Forcing用真实值，否则用预测值）
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs, attention_weights


# ====================== 6. 测试用例（模拟数据训练） ======================
def generate_synthetic_data(batch_size=32, vocab_size=10):
    """生成模拟的源序列和目标序列（用于测试）"""
    src = torch.randint(2, vocab_size, (batch_size, MAX_LENGTH)).to(DEVICE)
    trg = torch.randint(2, vocab_size, (batch_size, MAX_LENGTH)).to(DEVICE)
    # 给目标序列添加SOS和EOS标记
    trg = torch.cat([
        torch.full((batch_size, 1), SOS_TOKEN).to(DEVICE),
        trg,
        torch.full((batch_size, 1), EOS_TOKEN).to(DEVICE)
    ], dim=1)
    return src, trg


# 初始化模型参数
INPUT_VOCAB_SIZE = 10  # 输入词汇表大小
OUTPUT_VOCAB_SIZE = 10  # 输出词汇表大小
HIDDEN_SIZE = 128
NUM_LAYERS = 2

# 构建模型
encoder = Encoder(INPUT_VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
decoder = Decoder(OUTPUT_VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略SOS_TOKEN的损失


# 训练函数
def train_step(model, src, trg, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    outputs, _ = model(src, trg)

    # 调整维度计算损失（trg去掉SOS，outputs去掉最后一步）
    output_dim = outputs.shape[-1]
    outputs = outputs.reshape(-1, output_dim)
    trg = trg[:, 1:].reshape(-1)  # 目标序列去掉SOS标记

    # 计算损失
    loss = criterion(outputs, trg)

    # 反向传播
    loss.backward()
    # 梯度裁剪（防止梯度爆炸）
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()

    return loss.item()


# 训练过程
num_epochs = 10
for epoch in range(num_epochs):
    src, trg = generate_synthetic_data()
    loss = train_step(model, src, trg, optimizer, criterion)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')


# ====================== 7. 预测函数（贪心搜索） ======================
def predict(model, src, max_len=MAX_LENGTH):
    """
    预测函数（贪心搜索）
    Args:
        model: 训练好的Seq2Seq模型
        src: 源序列 (1, src_len)
        max_len: 最大预测长度
    Returns:
        predicted_seq: 预测的目标序列
        attention_weights: 注意力权重矩阵
    """
    model.eval()
    with torch.no_grad():
        # 编码器前向传播
        encoder_outputs, hidden = model.encoder(src)

        # 初始化输入（SOS_TOKEN）
        input_token = torch.tensor([[SOS_TOKEN]]).to(DEVICE)
        predicted_seq = []
        attention_weights = []

        # 逐步预测
        for _ in range(max_len):
            # 解码器前向传播
            output, hidden, attn_weights = model.decoder(input_token, hidden, encoder_outputs)

            # 存储注意力权重
            attention_weights.append(attn_weights.cpu().numpy())

            # 取概率最大的token
            top1 = output.argmax(1).item()
            predicted_seq.append(top1)

            # 如果预测到EOS，停止
            if top1 == EOS_TOKEN:
                break

            # 更新输入
            input_token = torch.tensor([[top1]]).to(DEVICE)

        return predicted_seq, np.concatenate(attention_weights, axis=0)


# 测试预测
test_src = torch.randint(2, INPUT_VOCAB_SIZE, (1, MAX_LENGTH)).to(DEVICE)
pred_seq, attn_weights = predict(model, test_src)
print("源序列:", test_src.cpu().numpy())
print("预测序列:", pred_seq)