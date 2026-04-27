import torch
import numpy as np
from collections import defaultdict
from torch import nn
import collections
import math
import os
import urllib.request
import zipfile
from torch.utils.data import DataLoader, Dataset


def beam_search(
        net, src_sentence, src_vocab, tgt_vocab, num_steps, device,
        beam_size=3, alpha=0.7  # alpha: 长度惩罚系数
):
    """
    束搜索解码
    Args:
        net: Seq2Seq模型
        src_sentence: 源语言输入句子（字符串）
        src_vocab/tgt_vocab: 源/目标语言词汇表
        num_steps: 最大解码长度
        device: 计算设备
        beam_size: 束宽度（越大效果越好但速度越慢）
        alpha: 长度惩罚系数（0.6-1.0，越小越鼓励短序列）
    Returns:
        最优翻译结果（字符串）
    """
    net.eval()
    with torch.no_grad():
        # 1. 编码器处理输入
        src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
        enc_X = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(0)
        enc_outputs = net.encoder(enc_X)
        dec_state = net.decoder.init_state(enc_outputs)  # (num_layers, 1, hidden_size)

        # 2. 初始化束搜索状态
        # 每个候选存储：(序列token列表, 累计对数概率, 解码器状态, 是否结束)
        beams = [
            ([tgt_vocab['<bos>']], 0.0, dec_state, False)
        ]

        # 3. 逐步解码
        for _ in range(num_steps):
            candidates = []  # 存储所有候选（包括扩展后的）

            # 遍历当前所有存活的束
            for seq, score, state, finished in beams:
                if finished:  # 已结束的序列直接加入候选
                    candidates.append((seq, score, state, finished))
                    continue

                # 3.1 解码器前向传播
                dec_X = torch.tensor([seq[-1]], dtype=torch.long, device=device).unsqueeze(0)
                Y, new_state = net.decoder(dec_X, state)  # Y: (1,1,vocab_size)

                # 3.2 计算对数概率（避免数值下溢）
                log_probs = torch.log_softmax(Y[0, 0], dim=0)  # (vocab_size,)

                # 3.3 扩展候选（取top-k）
                top_k_probs, top_k_indices = torch.topk(log_probs, beam_size)
                for prob, idx in zip(top_k_probs.cpu().numpy(), top_k_indices.cpu().numpy()):
                    new_seq = seq + [int(idx)]
                    new_score = score + prob.item()
                    new_finished = (idx == tgt_vocab['<eos>'])
                    candidates.append((new_seq, new_score, new_state, new_finished))

            # 4. 筛选最优束（应用长度惩罚）
            def score_with_penalty(candidate):
                seq, score, _, finished = candidate
                # 长度惩罚：score / (len(seq)^alpha)，避免过短序列
                penalty = (len(seq) ** alpha) if not finished else 1.0
                return score / penalty

            # 按惩罚后的分数排序，保留beam_size个
            candidates.sort(key=score_with_penalty, reverse=True)
            beams = candidates[:beam_size]

            # 提前终止：所有束都已结束
            if all([b[3] for b in beams]):
                break

        # 5. 选择最优序列（无惩罚的原始分数）
        best_seq = max(beams, key=lambda x: x[1])[0]

        # 6. 转换为文本（去掉<bos>和<eos>）
        best_seq = best_seq[1:]  # 去掉<bos>
        if tgt_vocab['<eos>'] in best_seq:
            best_seq = best_seq[:best_seq.index(tgt_vocab['<eos>'])]

        return ' '.join(tgt_vocab.to_tokens(best_seq))

# ==========================================
# PART 1: 数据准备 (修复路径问题 + 增强鲁棒性)
# ==========================================
def download_and_read_fra_eng():
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip'
    filename = 'fra-eng.zip'
    data_dir = 'data_nmt'

    # 1. 下载文件（增加异常处理）
    if not os.path.exists(filename):
        try:
            print(f"正在下载数据集: {url}")
            urllib.request.urlretrieve(url, filename)
            print("下载完成！")
        except Exception as e:
            raise RuntimeError(f"下载失败: {e}")

    # 2. 解压文件
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            print("解压完成！")

    # 3. 修复文件路径（关键修复点）
    # 检查实际的文件路径
    fra_file_path = os.path.join(data_dir, 'fra-eng', 'fra.txt')
    if not os.path.exists(fra_file_path):
        # 兼容不同的解压结构
        fra_file_path = os.path.join(data_dir, 'fra.txt')

    if not os.path.exists(fra_file_path):
        raise FileNotFoundError(f"找不到fra.txt文件，请检查解压后的目录结构")

    # 读取文件
    with open(fra_file_path, 'r', encoding='utf-8') as f:
        return f.read()


def preprocess_nmt(text):
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i >= num_examples:  # 修复边界问题（原代码是i>num_examples）
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 统计词频
        counter = collections.Counter([token for line in tokens for token in line])
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 初始化词汇表
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        # 添加高频词
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, 0)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


class NMTDataset(Dataset):
    def __init__(self, source, target, src_vocab, tgt_vocab):
        self.src = [src_vocab[line] for line in source]
        self.tgt = [tgt_vocab[line] for line in target]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


def load_data_nmt_pytorch(batch_size, num_steps, num_examples=600):
    # 加载并预处理数据
    text = preprocess_nmt(download_and_read_fra_eng())
    source, target = tokenize_nmt(text, num_examples)

    # 创建词汇表
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        src_valid, tgt_valid = [], []
        for src, tgt in batch:
            # 添加结束符
            src.append(src_vocab['<eos>'])
            tgt.append(tgt_vocab['<eos>'])
            # 计算有效长度
            src_valid.append(min(len(src), num_steps))
            tgt_valid.append(min(len(tgt), num_steps))
            # 截断或填充到固定长度
            src_batch.append((src + [src_vocab['<pad>']] * num_steps)[:num_steps])
            tgt_batch.append((tgt + [tgt_vocab['<pad>']] * num_steps)[:num_steps])

        # 转换为tensor
        return (torch.tensor(src_batch, dtype=torch.long),
                torch.tensor(src_valid, dtype=torch.long),
                torch.tensor(tgt_batch, dtype=torch.long),
                torch.tensor(tgt_valid, dtype=torch.long))

    # 创建数据集和数据加载器
    dataset = NMTDataset(source, target, src_vocab, tgt_vocab)
    data_iter = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    return data_iter, src_vocab, tgt_vocab


# ==========================================
# PART 2: Seq2Seq 模型定义
# ==========================================
class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, X):
        # X: (batch_size, seq_len)
        embeddings = self.embedding(X)  # (batch_size, seq_len, embed_size)
        outputs, state = self.rnn(embeddings)  # outputs: (batch_size, seq_len, hidden_size)
        return outputs, state


class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dense = nn.Linear(hidden_size, vocab_size)

    def init_state(self, enc_outputs):
        # 仅使用编码器的最终隐藏状态
        return enc_outputs[1]

    def forward(self, X, state):
        # X: (batch_size, seq_len)
        # state: (num_layers, batch_size, hidden_size)

        # 嵌入层
        embeddings = self.embedding(X)  # (batch_size, seq_len, embed_size)

        # 上下文向量（使用最后一层的隐藏状态）
        context = state[-1].unsqueeze(1).repeat(1, X.shape[1], 1)  # (batch_size, seq_len, hidden_size)

        # 拼接输入和上下文
        input_concat = torch.cat((embeddings, context), dim=2)  # (batch_size, seq_len, embed_size+hidden_size)

        # RNN前向传播
        outputs, state = self.rnn(input_concat, state)  # outputs: (batch_size, seq_len, hidden_size)

        # 输出层
        outputs = self.dense(outputs)  # (batch_size, seq_len, vocab_size)
        return outputs, state


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X):
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(dec_X, dec_state)


# ==========================================
# PART 3: BLEU 计算函数
# ==========================================
def bleu(pred_seq, label_seq, k):
    """
    计算 BLEU 分数
    pred_seq: 预测出的单词列表 (字符串)
    label_seq: 真实的标签单词列表 (字符串)
    k: 最大的 n-gram 长度 (通常取4)
    """
    # 处理空预测的情况
    if not pred_seq.strip():
        return 0.0

    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)

    # 1. 简短惩罚 (Brevity Penalty): 如果预测太短，得分会很低
    if len_pred == 0:
        bp = 0.0
    else:
        bp = math.exp(min(0, 1 - len_label / len_pred))

    # 2. 计算 n-gram 精度
    score = bp
    for n in range(1, k + 1):
        # 处理n大于序列长度的情况
        if len_pred < n or len_label < n:
            p_n = 0.0
        else:
            num_matches, label_subs = 0, collections.defaultdict(int)

            # 统计标签中所有 n-gram 的出现次数
            for i in range(len_label - n + 1):
                label_subs[' '.join(label_tokens[i: i + n])] += 1

            # 统计预测中匹配到的 n-gram
            for i in range(len_pred - n + 1):
                sub = ' '.join(pred_tokens[i: i + n])
                if label_subs[sub] > 0:
                    num_matches += 1
                    label_subs[sub] -= 1  # 用掉一个匹配额度

            # 计算该 n-gram 的精度
            p_n = num_matches / (len_pred - n + 1) if (len_pred - n + 1) > 0 else 0

        # 乘到总分上 (p_n^(1/2^n))
        score *= math.pow(p_n, math.pow(0.5, n))

    return score


# ==========================================
# PART 4: 训练与评估
# ==========================================
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    net.eval()
    with torch.no_grad():
        # 预处理输入句子
        src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
        enc_X = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(0)

        # 编码器前向传播
        enc_outputs = net.encoder(enc_X)
        dec_state = net.decoder.init_state(enc_outputs)

        # 解码器初始输入（<bos>）
        dec_X = torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device).unsqueeze(0)

        output_seq = []
        for _ in range(num_steps):
            Y, dec_state = net.decoder(dec_X, dec_state)
            # 取概率最大的token
            dec_X = Y.argmax(dim=2)
            pred = dec_X.item()
            # 遇到<eos>停止
            if pred == tgt_vocab['<eos>']:
                break
            output_seq.append(pred)

        # 转换为文本
        return ' '.join(tgt_vocab.to_tokens(output_seq))


def train(net, data_iter, lr, num_epochs, tgt_vocab, device):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        num_batches = 0

        for batch in data_iter:
            X, X_valid, Y, Y_valid = [x.to(device) for x in batch]

            # Teacher Forcing 输入: <bos> + 句子（去掉最后一个token）
            dec_input = torch.cat(
                [torch.tensor([tgt_vocab['<bos>']] * len(X), device=device).reshape(-1, 1), Y[:, :-1]], dim=1)

            # 模型前向传播
            Y_hat, _ = net(X, dec_input)

            # 损失计算 (带Mask，只计算有效长度内的损失)
            loss = loss_fn(Y_hat.permute(0, 2, 1), Y)
            mask = torch.arange(Y.shape[1], device=device)[None, :] < Y_valid[:, None]
            loss = (loss * mask).sum() / mask.sum()  # 使用sum/有效数 替代mean，更准确

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # 每50轮打印一次损失
        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / num_batches
            print(f'Epoch {epoch + 1}, Average Loss {avg_loss:.3f}')

# ========== 测试束搜索 ==========
# 替换原有predict_seq2seq调用即可
if __name__ == "__main__":
    # （复用你之前的设备/模型/词汇表初始化代码）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size, num_steps = 64, 10
    train_iter, src_vocab, tgt_vocab = load_data_nmt_pytorch(batch_size, num_steps)

    # 初始化模型
    encoder = Seq2SeqEncoder(len(src_vocab), 32, 32, 2, 0.1)
    decoder = Seq2SeqDecoder(len(tgt_vocab), 32, 32, 2, 0.1)
    net = EncoderDecoder(encoder, decoder).to(device)

    # 用束搜索翻译
    eng = 'go .'
    translation = beam_search(net, eng, src_vocab, tgt_vocab, num_steps, device, beam_size=3)
    print(f"输入: {eng}")
    print(f"束搜索翻译: {translation}")