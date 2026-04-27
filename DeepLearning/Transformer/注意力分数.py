import math
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


# ====================== 1. 掩码Softmax（替代d2l.sequence_mask + 原masked_softmax） ======================
def sequence_mask(X, valid_len, value=-1e6):
    """
    对序列进行掩码操作：超过有效长度的位置填充value
    X: 2D张量 (batch_size * seq_len, feature_dim) 或 (batch_size, seq_len)
    valid_len: 1D张量 (batch_size * seq_len,) 或 (batch_size,)
    """
    max_len = X.size(1)
    # 创建长度索引矩阵: [0,1,2,...,max_len-1]
    mask = torch.arange(max_len, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    # 对mask取反的位置填充value
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X: 3D张量 (batch_size, num_queries/keys, feature_dim)
    # valid_lens: 1D张量 (batch_size,) 或 2D张量 (batch_size, num_queries)
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        # 适配valid_lens维度：转为1D（长度=batch_size * shape[1]）
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)

        # 重塑为2D进行掩码，再恢复原形状
        X_reshape = X.reshape(-1, shape[-1])  # (batch_size*shape[1], shape[-1])
        X_masked = sequence_mask(X_reshape, valid_lens, value=-1e6)
        return nn.functional.softmax(X_masked.reshape(shape), dim=-1)


# 测试掩码Softmax
print("=== 测试masked_softmax ===")
# 测试用例1：valid_lens为1D
test_X1 = torch.rand(2, 2, 4)
test_valid1 = torch.tensor([2, 3])
output1 = masked_softmax(test_X1, test_valid1)
print("1D valid_lens输出:\n", output1)

# 测试用例2：valid_lens为2D
test_X2 = torch.rand(2, 2, 4)
test_valid2 = torch.tensor([[1, 3], [2, 4]])
output2 = masked_softmax(test_X2, test_valid2)
print("2D valid_lens输出:\n", output2)


# ====================== 2. 加性注意力（Additive Attention） ======================
class AdditiveAttention(nn.Module):
    """加性注意力（适用于查询和键维度不同的场景）"""

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # queries: (batch_size, num_queries, query_size)
        # keys: (batch_size, num_kv_pairs, key_size)
        # values: (batch_size, num_kv_pairs, value_size)
        # valid_lens: (batch_size,) 或 (batch_size, num_queries)

        # 线性变换到隐藏空间
        queries = self.W_q(queries)  # (batch_size, num_queries, num_hiddens)
        keys = self.W_k(keys)  # (batch_size, num_kv_pairs, num_hiddens)

        # 广播求和：扩展维度后相加
        # queries扩展: (batch_size, num_queries, 1, num_hiddens)
        # keys扩展: (batch_size, 1, num_kv_pairs, num_hiddens)
        # features: (batch_size, num_queries, num_kv_pairs, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)

        # 计算注意力分数：去掉最后一维
        # scores: (batch_size, num_queries, num_kv_pairs)
        scores = self.w_v(features).squeeze(-1)

        # 掩码Softmax得到注意力权重
        self.attention_weights = masked_softmax(scores, valid_lens)

        # 加权求和：dropout后与values做批量矩阵乘法
        # output: (batch_size, num_queries, value_size)
        return torch.bmm(self.dropout(self.attention_weights), values)


# 测试加性注意力
print("\n=== 测试加性注意力 ===")
queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# values: 两个样本的value矩阵相同
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
valid_lens = torch.tensor([2, 6])

# 初始化模型
add_attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
add_attention.eval()  # 评估模式，关闭dropout
with torch.no_grad():
    add_output = add_attention(queries, keys, values, valid_lens)
print("加性注意力输出形状:", add_output.shape)
print("加性注意力输出:\n", add_output)


# ====================== 3. 缩放点积注意力（Dot Product Attention） ======================
class DotProductAttention(nn.Module):
    """缩放点积注意力（适用于查询和键维度相同的场景，Transformer核心）"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # queries: (batch_size, num_queries, d)
        # keys: (batch_size, num_kv_pairs, d)
        # values: (batch_size, num_kv_pairs, value_size)
        # valid_lens: (batch_size,) 或 (batch_size, num_queries)
        d = queries.shape[-1]

        # 计算点积并缩放：scores = (Q @ K^T) / sqrt(d)
        # scores: (batch_size, num_queries, num_kv_pairs)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)

        # 掩码Softmax得到注意力权重
        self.attention_weights = masked_softmax(scores, valid_lens)

        # 加权求和：dropout后与values做批量矩阵乘法
        return torch.bmm(self.dropout(self.attention_weights), values)


# 测试缩放点积注意力
print("\n=== 测试缩放点积注意力 ===")
queries = torch.normal(0, 1, (2, 1, 2))  # 查询维度与键维度一致（2）
dot_attention = DotProductAttention(dropout=0.5)
dot_attention.eval()
with torch.no_grad():
    dot_output = dot_attention(queries, keys, values, valid_lens)
print("缩放点积注意力输出形状:", dot_output.shape)
print("缩放点积注意力输出:\n", dot_output)


# ====================== 4. 注意力权重热力图可视化（替代d2l.show_heatmaps） ======================
def show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries', title='Attention Weights'):
    """
    可视化注意力权重热力图
    attention_weights: 4D张量 (num_rows, num_cols, num_queries, num_keys)
    """
    # 转换为numpy数组
    attention_weights = attention_weights.detach().cpu().numpy()
    # 获取子图数量
    num_rows, num_cols = attention_weights.shape[0], attention_weights.shape[1]

    # 创建子图
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    axes = axes.flatten() if num_rows * num_cols > 1 else [axes]

    # 绘制每个热力图
    for i, (ax, weight) in enumerate(zip(axes, attention_weights.reshape(-1, *attention_weights.shape[2:]))):
        im = ax.imshow(weight, cmap='Reds')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} ({i + 1}/{num_rows * num_cols})')
        # 添加颜色条
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


# 可视化加性注意力权重
print("\n=== 可视化加性注意力权重 ===")
add_attn_weights = add_attention.attention_weights.reshape((1, 1, 2, 10))
show_heatmaps(add_attn_weights, xlabel='Keys', ylabel='Queries')

# 可视化缩放点积注意力权重
print("\n=== 可视化缩放点积注意力权重 ===")
dot_attn_weights = dot_attention.attention_weights.reshape((1, 1, 2, 10))
show_heatmaps(dot_attn_weights, xlabel='Keys', ylabel='Queries')