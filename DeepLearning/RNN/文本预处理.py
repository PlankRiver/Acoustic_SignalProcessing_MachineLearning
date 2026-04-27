import collections
import re
import os
import urllib.request
import torch


# ==========================================
# 1. 数据下载与读取 (替代 d2l.DATA_HUB)
# ==========================================
def download_time_machine():
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
    filename = 'timemachine.txt'
    if not os.path.exists(filename):
        print(f"正在下载 {filename} ...")
        urllib.request.urlretrieve(url, filename)
    return filename


def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    filename = download_time_machine()
    with open(filename, 'r') as f:
        lines = f.readlines()
    # 正则清洗：将非字母字符替换为空格，去两端空格，转小写
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


# ==========================================
# 2. 分词函数
# ==========================================
def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


# ==========================================
# 3. 词表类 (Vocab) - 核心工具
# ==========================================
class Vocab:
    """文本词表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        # 统计词频
        counter = count_corpus(tokens)
        # 按频率从高到低排序
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # 初始化列表和字典
        # <unk> 是 unknown 的缩写，索引通常设为 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        # 将满足频率要求的词加入词表
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """支持传入单个token或token列表，返回对应的索引"""
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """支持传入单个索引或索引列表，返回对应的token"""
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0  # <unk> 的索引

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的 tokens 可能是 1D 列表或 2D 列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


# ==========================================
# 4. 打包函数：加载数据并转换为 Corpus (Tensor)
# ==========================================
def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表(corpus)和词表(vocab)"""
    lines = read_time_machine()
    # 这里通常使用 'char' (字符级) 来做简单的序列模型演示
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)

    # Corpus: 将所有 token 转换为对应的数字索引，并展平成一个长列表
    corpus = [vocab[token] for line in tokens for token in line]

    if max_tokens > 0:
        corpus = corpus[:max_tokens]

    # 【重点】转换为 PyTorch Tensor，这是深度学习的标准格式
    return torch.tensor(corpus), vocab


# ==========================================
# 5. 测试运行
# ==========================================
# 获取 corpus (数据) 和 vocab (字典)
corpus, vocab = load_corpus_time_machine()

print(f"Corpus 类型: {type(corpus)}")  # 应该是 torch.Tensor
print(f"Corpus 形状: {corpus.shape}")  # 这是一个很长的一维向量
print(f"Vocab 长度: {len(vocab)}")  # 只有28个 (26个字母 + 空格 + <unk>)

# 打印前 10 个数据看看
print("\n前10个数据 (Indices):", corpus[:10])
print("前10个数据 (Tokens):", vocab.to_tokens(corpus[:10].tolist()))