import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class MarkovPredictor:
    def __init__(self):
        self.states = None  # 状态列表
        self.state_dict = {}  # 状态到索引的映射
        self.P = None  # 转移概率矩阵

    def fit(self, state_sequence):
        """
        训练模型：根据历史状态序列计算转移矩阵
        :param state_sequence: list or array, 历史状态序列 (如 ['A', 'A', 'B', 'C', 'A'])
        """
        # 1. 识别所有唯一状态并排序
        self.states = sorted(list(set(state_sequence)))
        n_states = len(self.states)
        self.state_dict = {state: i for i, state in enumerate(self.states)}

        print(f"识别到 {n_states} 个状态: {self.states}")

        # 2. 初始化计数矩阵
        count_mat = np.zeros((n_states, n_states))

        # 3. 遍历序列统计转移次数 (从 i 到 j)
        for k in range(len(state_sequence) - 1):
            current_s = state_sequence[k]
            next_s = state_sequence[k + 1]

            i = self.state_dict[current_s]
            j = self.state_dict[next_s]
            count_mat[i, j] += 1

        # 4. 归一化得到概率矩阵 P
        # axis=1 按行求和, keepdims=True 保持二维以便广播除法
        row_sums = count_mat.sum(axis=1, keepdims=True)

        # 处理某状态从未转移出去的情况 (避免除以0)
        # 如果某行和为0，说明该状态是终点，通常设为保持自身 (概率1)
        row_sums[row_sums == 0] = 1

        self.P = count_mat / row_sums

        print("-" * 30)
        print("转移概率矩阵构建完成。")

    def predict(self, current_status_dist, steps=1):
        """
        预测未来状态
        :param current_status_dist: 当前状态分布向量 (如 [0.3, 0.4, 0.3]) 或 单个状态名称
        :param steps: 预测未来几步
        :return: 预测后的状态分布向量
        """
        if self.P is None:
            raise ValueError("请先调用 fit() 训练模型")

        # 处理输入：如果是单个状态名称（如 'A'），转为 one-hot 向量
        n = len(self.states)
        S0 = np.zeros(n)

        if isinstance(current_status_dist, str):
            idx = self.state_dict[current_status_dist]
            S0[idx] = 1.0
        else:
            S0 = np.array(current_status_dist)
            if abs(S0.sum() - 1) > 0.01:
                print("警告: 初始状态概率和不为1，已自动归一化")
                S0 = S0 / S0.sum()

        # 矩阵乘法预测: S_k = S_0 * P^k
        # 也就是连续乘 k 次 P
        Sk = S0
        for _ in range(steps):
            Sk = np.dot(Sk, self.P)

        return Sk

    def plot_matrix(self):
        """绘制转移矩阵热力图"""
        if self.P is None: return

        plt.figure(figsize=(8, 6))
        sns.heatmap(self.P, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.states, yticklabels=self.states)
        plt.title('马尔可夫状态转移矩阵')
        plt.xlabel('下一时刻状态')
        plt.ylabel('当前时刻状态')
        plt.show()


# ================= 使用示例 =================

if __name__ == "__main__":
    # 场景：某城市每年最流行的交通方式变化
    # 数据：过去15年的主要交通方式记录
    # A=公交, B=地铁, C=私家车
    history_data = [
        'A', 'A', 'B', 'B', 'B', 'C', 'C', 'B',
        'C', 'C', 'C', 'B', 'A', 'B', 'C'
    ]

    # 1. 初始化并训练
    mk = MarkovPredictor()
    mk.fit(history_data)

    # 2. 展示转移矩阵
    mk.plot_matrix()

    # 3. 预测：假设今年是 'C' (私家车)，预测 3 年后的交通格局
    future_steps = 3
    current_state = 'C'

    pred_dist = mk.predict(current_state, steps=future_steps)

    print("-" * 30)
    print(f"当前状态: {current_state}")
    print(f"{future_steps} 年后的状态分布预测:")
    for state, prob in zip(mk.states, pred_dist):
        print(f"状态 {state}: {prob * 100:.2f}%")

    # 4. 稳态分析 (Long-term)
    # 预测很多步之后，分布会趋于稳定
    steady_dist = mk.predict(current_state, steps=100)
    print("-" * 30)
    print("长期稳态分布 (最终市场份额):")
    print(dict(zip(mk.states, np.round(steady_dist, 4))))