import numpy as np
import pandas as pd


class FuzzyEvaluation:
    def __init__(self, weights, criteria, ratings):
        """
        初始化模糊综合评价模型
        :param weights: list or np.array, 指标权重向量 (1 x n)
        :param criteria: list, 指标名称列表 (n)
        :param ratings: list, 评价等级名称列表 (m)
        """
        self.weights = np.array(weights)
        self.criteria = criteria
        self.ratings = ratings

        # 检查维度
        if len(self.weights) != len(self.criteria):
            raise ValueError("权重数量与指标数量不一致")

    def evaluate(self, membership_matrix, score_vector=None):
        """
        进行模糊运算
        :param membership_matrix: np.array, 隶属度矩阵 (n x m)
                                  每一行代表一个指标在各个等级上的得票率
        :param score_vector: list, 等级对应的分数值 (1 x m), 用于计算最终得分
                             例如: [100, 80, 60, 40]
        :return: result_vector (综合评价向量), final_score (综合得分)
        """
        R = np.array(membership_matrix)

        # 检查矩阵维度 (行数必须等于指标数)
        if R.shape[0] != len(self.weights):
            raise ValueError(f"隶属度矩阵行数 ({R.shape[0]}) 与权重数 ({len(self.weights)}) 不匹配")

        # --- 核心运算: 矩阵乘法 (B = A * R) ---
        # 这种算法叫做 "加权平均型" (M(·, +))，最常用，考虑了所有因素的影响
        B = np.dot(self.weights, R)

        # 归一化 (理论上如果R的行和为1，权重和为1，B的和也为1，但为了保险起见再归一化一次)
        B_normalized = B / np.sum(B)

        # --- 计算具体得分 (可选) ---
        final_score = 0
        if score_vector is not None:
            scores = np.array(score_vector)
            if len(scores) != len(self.ratings):
                raise ValueError("分数值数量与评价等级数量不一致")
            final_score = np.dot(B_normalized, scores)

        return B_normalized, final_score


# ================= 使用示例 =================

if __name__ == "__main__":
    print("=== 模糊综合评价示例：员工绩效考核 ===")

    # 1. 定义基本信息
    # 指标集 U
    criteria_list = ['工作业绩', '团队合作', '工作态度', '创新能力']
    # 评价集 V
    ratings_list = ['优秀', '良好', '一般', '较差']
    # 评价集对应的分数 (用于最后算总分)
    rating_scores = [95, 80, 65, 50]

    # 2. 确定权重 A (假设通过AHP或熵权法已算出)
    # 业绩最重要(0.4)，其次是态度(0.3)...
    weights = [0.4, 0.2, 0.3, 0.1]

    # 3. 构建隶属度矩阵 R
    # 假设我们发放了 10 份问卷，或者由 10 位专家打分。
    # 我们统计每个人在每个指标上的得票数，然后除以总人数得到比例。

    # 行1(业绩): 5人投优秀，3人投良好，2人一般，0人差 -> [0.5, 0.3, 0.2, 0.0]
    # 行2(合作): 2人投优秀，5人投良好，3人一般，0人差 -> [0.2, 0.5, 0.3, 0.0]
    # 行3(态度): 9人投优秀，1人投良好，0人一般，0人差 -> [0.9, 0.1, 0.0, 0.0]
    # 行4(创新): 0人投优秀，2人投良好，5人一般，3人差 -> [0.0, 0.2, 0.5, 0.3]

    membership_mat = np.array([
        [0.5, 0.3, 0.2, 0.0],  # 工作业绩
        [0.2, 0.5, 0.3, 0.0],  # 团队合作
        [0.9, 0.1, 0.0, 0.0],  # 工作态度
        [0.0, 0.2, 0.5, 0.3]  # 创新能力
    ])

    print("1. 权重向量:", weights)
    print("2. 隶属度矩阵 R:\n", membership_mat)

    # 4. 运行模型
    fce = FuzzyEvaluation(weights, criteria_list, ratings_list)
    result_vec, score = fce.evaluate(membership_mat, rating_scores)

    print("-" * 30)
    print("3. 综合评价结果向量 B:", np.round(result_vec, 4))

    # 5. 结果分析
    max_idx = np.argmax(result_vec)
    print(f">> 根据最大隶属度原则，该员工评价等级为: 【{ratings_list[max_idx]}】 (概率: {result_vec[max_idx] * 100:.2f}%)")

    print("-" * 30)
    print(f">> 综合百分制得分: {score:.2f} 分")