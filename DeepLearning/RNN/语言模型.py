import torch
import random
import collections
import re
import matplotlib.pyplot as plt
import os
import urllib.request


# ==========================================
# 1. 基础工具类 (读取、清洗、词表)
# ==========================================
def read_time_machine():
    """下载并读取文本"""
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
    filename = 'timemachine.txt'
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token='word'):
    """分词"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]


class Vocab:
    """词表构建"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None: tokens = []
        if reserved_tokens is None: reserved_tokens = []
        counter = collections.Counter(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq: break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)): return self.token_to_idx.get(tokens, 0)
        return [self.__getitem__(token) for token in tokens]

    @property
    def token_freqs(self):
        return self._token_freqs


# ==========================================
# 2. 核心：两种采样迭代器
# ==========================================

def seq_data_iter_random(corpus, batch_size, num_steps):
    """
    方法一：随机采样
    特点：相邻的两个 Batch 在原始文本中是不相邻的。
    """
    # 随机丢弃开头的一点点数据，增加随机性
    corpus = corpus[random.randint(0, num_steps - 1):]

    # 能够切分出的子序列数量 (减1是因为要有对应的 Target)
    num_subseqs = (len(corpus) - 1) // num_steps

    # 生成所有子序列的起始索引: [0, num_steps, 2*num_steps, ...]
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))

    # 【关键】打乱这些索引，这就叫"随机采样"
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    # 每次取 batch_size 个索引出来
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]

        # X 是输入，Y 是预测目标（向后偏移一位）
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """
    方法二：顺序分区
    特点：Batch 2 的内容紧接着 Batch 1 的内容。
    """
    # 随机偏移起始位置
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size

    # 截取可用数据，转换为 Tensor
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])

    # 【关键】reshape 成 (batch_size, -1)
    # 这相当于把长文本切成 batch_size 条平行的长带子
    Xs = Xs.reshape(batch_size, -1)
    Ys = Ys.reshape(batch_size, -1)

    num_batches = Xs.shape[1] // num_steps

    # 按列切片，依次向后移动
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


# ==========================================
# 3. 封装类 (DataLoader)
# ==========================================
class TimeMachineDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter=False, max_tokens=10000):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.use_random_iter = use_random_iter

        # 加载数据
        lines = read_time_machine()
        tokens = tokenize(lines, 'char')
        self.vocab = Vocab([token for line in tokens for token in line])
        # 转数字 ID
        self.corpus = [self.vocab[token] for line in tokens for token in line]
        if max_tokens > 0:
            self.corpus = self.corpus[:max_tokens]

    def __iter__(self):
        if self.use_random_iter:
            return seq_data_iter_random(self.corpus, self.batch_size, self.num_steps)
        else:
            return seq_data_iter_sequential(self.corpus, self.batch_size, self.num_steps)


# ==========================================
# 4. 测试与演示
# ==========================================
# 制造一个简单的序列 0~34 来演示
my_seq = list(range(35))

print("=== 1. 随机采样演示 ===")
# 你会发现 X 的行与行之间没关系，Batch与Batch之间也没关系
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X:', X, '\nY:', Y)
    print('-' * 20)

print("\n=== 2. 顺序分区演示 ===")
# 你会发现 Batch 2 的第一行 紧接着 Batch 1 的第一行
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X:', X, '\nY:', Y)
    print('-' * 20)