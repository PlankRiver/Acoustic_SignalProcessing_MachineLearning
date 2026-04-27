import numpy as np
import pandas as pd
from scipy.optimize import linprog


class DEA:
    def __init__(self, inputs, outputs):
        """
        初始化 DEA 模型
        :param inputs: pandas DataFrame or np.array, 投入数据 (n_samples x m_inputs)
        :param outputs: pandas DataFrame or np.array, 产出数据 (n_samples x s_outputs)
        """
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n = self.inputs.shape[0]  # DMU数量
        self.m = self.inputs.shape[1]  # 投入指标数量
        self.s = self.outputs.shape[1]  # 产出指标数量

        # 结果容器
        self.efficiency = []

    def fit(self):
        """
        使用 CCR 模型 (投入导向) 计算每个 DMU 的效率
        """
        # 对每一个 DMU (k) 进行循环求解
        for k in range(self.n):
            # === 线性规划配置 ===
            # 目标函数: min theta
            # 决策变量向量构造: [theta, lambda_1, lambda_2, ..., lambda_n]
            # 所以总变量个数 = 1 + n

            # 1. 目标函数系数 c
            # 我们只优化 theta (第一个变量), lambda 的系数都是 0
            c = np.zeros(1 + self.n)
            c[0] = 1

            # 2. 不等式约束 A_ub * x <= b_ub
            # 我们有 m 个投入约束 和 s 个产出约束

            # (1) 投入约束: sum(lambda * x_j) <= theta * x_k
            # 移项得: sum(lambda * x_j) - theta * x_k <= 0
            # 对应系数: [-x_ik, x_i1, x_i2, ..., x_in]
            A_inputs = []
            for i in range(self.m):
                row = [-self.inputs[k, i]] + list(self.inputs[:, i])
                A_inputs.append(row)

            # (2) 产出约束: sum(lambda * y_j) >= y_k
            # scipy 只支持 <=, 所以两边取反: -sum(lambda * y_j) <= -y_k
            # 对应系数: [0, -y_r1, -y_r2, ..., -y_rn] (theta的系数是0)
            A_outputs = []
            for r in range(self.s):
                row = [0] + list(-self.outputs[:, r])
                A_outputs.append(row)

            # 合并约束矩阵
            A_ub = np.array(A_inputs + A_outputs)

            # 构造 b_ub
            # 投入约束右边是 0
            # 产出约束右边是 -y_rk
            b_ub = np.concatenate([np.zeros(self.m), -self.outputs[k, :]])

            # 3. 变量边界
            # theta >= 0 (其实理论上没有上限，但效率一般<=1，不过不设限让模型自己跑)
            # lambda >= 0
            bounds = [(0, None)] + [(0, None)] * self.n

            # === 求解 ===
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

            if res.success:
                self.efficiency.append(res.x[0])  # res.x[0] 就是 theta
            else:
                self.efficiency.append(np.nan)
                print(f"DMU {k} 求解失败")

        return np.array(self.efficiency)

    def analysis(self, index_names=None):
        """
        输出分析结果表格
        """
        eff = np.array(self.efficiency)

        # 简单的评价逻辑
        results = []
        for e in eff:
            if np.isclose(e, 1.0):
                results.append("DEA有效")
            else:
                results.append("非DEA有效")

        df_res = pd.DataFrame({
            '效率值(Theta)': np.round(eff, 4),
            '评价结果': results
        })

        if index_names is not None:
            df_res.index = index_names

        return df_res


# ================= 使用示例 =================

if __name__ == "__main__":
    # 场景：评价 5 家医院的运营效率
    # 投入 (越少越好)：医生人数、护士人数、运营成本(万)
    inputs_data = pd.DataFrame({
        '医生': [20, 50, 40, 60, 25],
        '护士': [50, 100, 90, 120, 60],
        '成本': [100, 300, 250, 400, 120]
    })

    # 产出 (越多越好)：门诊量(人次)、治愈人数、病床周转率
    outputs_data = pd.DataFrame({
        '门诊': [2000, 6000, 4500, 6500, 2800],
        '治愈': [180, 500, 400, 550, 220],
        '周转': [80, 90, 85, 95, 88]
    })

    print("投入数据:\n", inputs_data)
    print("产出数据:\n", outputs_data)

    # 1. 初始化模型
    dea = DEA(inputs_data, outputs_data)

    # 2. 计算效率
    dea.fit()

    # 3. 输出结果
    hospital_names = ['医院A', '医院B', '医院C', '医院D', '医院E']
    result_df = dea.analysis(index_names=hospital_names)

    print("-" * 30)
    print("DEA 评价结果:")
    print(result_df)

    # 结果解读：
    # 效率值 = 1.0000 -> 它是标杆，投入产出比非常棒。
    # 效率值 = 0.8500 -> 它是非有效的，说明它当前的产出，理论上只需要投入现在的 85% 就能做到（或者在当前投入下，产出还能更高）。