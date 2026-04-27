import os
import torch
import collections
import urllib.request
import zipfile
import re
from torch.utils.data import DataLoader, Dataset


# ==========================================
# 1. 下载与读取原始数据 (替代 d2l.download)
# ==========================================
def download_and_read_fra_eng():
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip'
    filename = 'fra-eng.zip'
    data_dir = 'data_nmt'

    # 下载
    if not os.path.exists(filename):
        print(f"正在下载 {filename}...")
        urllib.request.urlretrieve(url, filename)

    # 解压
    if not os.path.exists(data_dir):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    # 读取
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()


# ==========================================
# 2. 文本预处理 (清洗与分词)
# ==========================================
def preprocess_nmt(text):
    """
    1. 不间断空格转空格
    2. 大写转小写
    3. 标点符号前加空格
    """

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 替换特殊字符并转小写
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()

    # 在标点符号和单词之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """将文本分割成 英语(source) 和 法语(target) 的单词列表"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            # 简单按空格分词
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


# ==========================================
# 3. 构建词表 (Standard Vocab Class)
# ==========================================
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        # 统计词频
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # 初始化索引映射
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        # 添加满足频率的词
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        # 如果输入是单个token (str)
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        # 如果输入是列表 (list of str)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0  # <unk> index


def count_corpus(tokens):
    """统计词频"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


# ==========================================
# 4. 自定义 Dataset
# ==========================================
class NMTDataset(Dataset):
    def __init__(self, source_tokens, target_tokens, src_vocab, tgt_vocab):
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        # 预先定义好特殊token的索引
        self.eos_idx = self.tgt_vocab['<eos>']
        self.pad_idx = self.tgt_vocab['<pad>']

    def __len__(self):
        return len(self.source_tokens)

    def __getitem__(self, idx):
        # 将单词转换为数字索引
        src_indices = self.src_vocab[self.source_tokens[idx]]
        tgt_indices = self.tgt_vocab[self.target_tokens[idx]]
        return src_indices, tgt_indices


# ==========================================
# 5. Collate Function (核心：截断与填充)
# ==========================================
def truncate_pad(line, num_steps, padding_token):
    """截断或填充序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def batch_process_fn(batch, src_vocab, tgt_vocab, num_steps):
    """
    DataLoader的collate_fn逻辑：
    1. 给句子加上 <eos>
    2. 截断或填充到固定长度 num_steps
    3. 记录有效长度
    """
    src_batch, tgt_batch = [], []
    src_valid_lens, tgt_valid_lens = [], []

    pad_token_src = src_vocab['<pad>']
    eos_token_src = src_vocab['<eos>']
    pad_token_tgt = tgt_vocab['<pad>']
    eos_token_tgt = tgt_vocab['<eos>']

    for src_indices, tgt_indices in batch:
        # 处理 Source (英语)
        # 一般做法：Source 结尾加 <eos>
        src_indices = src_indices + [eos_token_src]
        src_valid_len = len(src_indices) if len(src_indices) < num_steps else num_steps
        src_processed = truncate_pad(src_indices, num_steps, pad_token_src)

        # 处理 Target (法语)
        # 一般做法：Target 结尾加 <eos>
        tgt_indices = tgt_indices + [eos_token_tgt]
        tgt_valid_len = len(tgt_indices) if len(tgt_indices) < num_steps else num_steps
        tgt_processed = truncate_pad(tgt_indices, num_steps, pad_token_tgt)

        src_batch.append(src_processed)
        tgt_batch.append(tgt_processed)
        src_valid_lens.append(src_valid_len)
        tgt_valid_lens.append(tgt_valid_len)

    # 转换为 Tensor
    return (torch.tensor(src_batch, dtype=torch.long),
            torch.tensor(src_valid_lens, dtype=torch.int32),
            torch.tensor(tgt_batch, dtype=torch.long),
            torch.tensor(tgt_valid_lens, dtype=torch.int32))


# ==========================================
# 6. 主加载函数
# ==========================================
def load_data_nmt_pytorch(batch_size, num_steps, num_examples=600):
    # 1. 读取原始数据
    raw_text = download_and_read_fra_eng()
    text = preprocess_nmt(raw_text)

    # 2. 分词
    source, target = tokenize_nmt(text, num_examples)

    # 3. 构建词表
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])

    # 4. 创建 Dataset
    dataset = NMTDataset(source, target, src_vocab, tgt_vocab)

    # 5. 创建 DataLoader (使用 lambda 传入额外的参数给 collate_fn)
    data_iter = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: batch_process_fn(batch, src_vocab, tgt_vocab, num_steps)
    )

    return data_iter, src_vocab, tgt_vocab


# ==========================================
# 测试代码
# ==========================================
if __name__ == '__main__':
    BATCH_SIZE = 2
    NUM_STEPS = 8

    # 加载数据
    train_iter, src_vocab, tgt_vocab = load_data_nmt_pytorch(BATCH_SIZE, NUM_STEPS)

    print(f"源语言词表大小: {len(src_vocab)}")
    print(f"目标语言词表大小: {len(tgt_vocab)}")

    # 打印一个 Batch 查看结果
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('-' * 40)
        print('X (Source, padded):', X.type(torch.int32))
        print('X Valid Len:', X_valid_len)
        print('Y (Target, padded):', Y.type(torch.int32))
        print('Y Valid Len:', Y_valid_len)
        print('-' * 40)

        # 验证一下还原回单词
        print("X[0] 还原:", [src_vocab.to_tokens(idx.item()) for idx in X[0]])
        break