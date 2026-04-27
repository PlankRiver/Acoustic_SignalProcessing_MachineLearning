import numpy as np


class AHP:
    def __init__(self, criteria_matrix):
        """
        criteria_matrix: 用户输入的判断矩阵 (numpy array)
        """
        self.matrix = np.array(criteria_matrix)
        self.n = self.matrix.shape[0]  # 指标个数
        self.RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
                        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}  # RI查表标准值

    def calculate_weights(self):
        """
        使用算术平均法计算权重
        """
        # 1. 列向量归一化
        column_sum = np.sum(self.matrix, axis=0)
        standardized_matrix = self.matrix / column_sum

        # 2. 求行和
        row_sum = np.sum(standardized_matrix, axis=1)

        # 3. 归一化得到权重
        self.weights = row_sum / self.n
        return self.weights

    def consistency_check(self):
        """
        进行一致性检验
        """
        # 计算最大特征值 lambda_max
        # lambda_max = sum((Aw)_i / (n * w_i))
        Aw = np.dot(self.matrix, self.weights)
        lambda_max = np.sum(Aw / (self.n * self.weights)) / self.n

        # 计算 CI
        CI = (lambda_max - self.n) / (self.n - 1)

        # 查找 RI
        if self.n in self.RI_dict:
            RI = self.RI_dict[self.n]
        else:
            print(f"警告: 矩阵维度 n={self.n} 超出常见RI表范围，请手动查表补充。")
            RI = 1.45  # 默认给个大值防止报错

        # 计算 CR
        # 只有当 n > 2 时才需要计算CR，n=1或2时CI为0，必定通过
        if self.n <= 2:
            CR = 0.0
        else:
            CR = CI / RI

        print("-" * 30)
        print(f"最大特征值 lambda_max: {lambda_max:.4f}")
        print(f"一致性指标 CI: {CI:.4f}")
        print(f"一致性比例 CR: {CR:.4f}")

        if CR < 0.1:
            print(">> 结果: 通过一致性检验！权重有效。")
            return True
        else:
            print(">> 结果: 未通过一致性检验 (CR >= 0.1)。请调整判断矩阵！")
            return False


# ================= 使用示例 =================

if __name__ == "__main__":
    # 场景模拟：我们要评价“景色、花费、居住、饮食”4个指标的权重
    # 用户(专家)给出的判断矩阵：
    #      景  花  居  食
    # 景 [[1,  2,  5,  7],  -> 景色比花费稍重要(2)，比居住明显重要(5)...
    # 花  [1/2,1,  3,  5],
    # 居  [1/5,1/3,1,  2],
    # 食  [1/7,1/5,1/2,1]]

    # 注意：矩阵必须是正互反矩阵 (对角线为1，a_ij = 1 / a_ji)
    input_matrix = np.array([
        [1, 2, 5, 7],
        [1 / 2, 1, 3, 5],
        [1 / 5, 1 / 3, 1, 2],
        [1 / 7, 1 / 5, 1 / 2, 1]
    ])

    # 1. 实例化模型
    ahp_model = AHP(input_matrix)

    # 2. 计算权重
    weights = ahp_model.calculate_weights()

    # 3. 进行检验
    is_valid = ahp_model.consistency_check()

    # 4. 输出最终结果
    if is_valid:
        print("-" * 30)
        print("最终计算的指标权重:")
        indicators = ['景色', '花费', '居住', '饮食']  # 修改成你的指标名
        for i, w in enumerate(weights):
            print(f"{indicators[i]}: {w:.4f} ({w * 100:.2f}%)")