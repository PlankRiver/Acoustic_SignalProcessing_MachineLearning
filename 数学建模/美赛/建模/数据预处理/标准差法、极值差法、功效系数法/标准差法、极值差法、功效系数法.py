import numpy as np

def range_scaling(data, direction=1):
    """
    极值差法
    direction: 1为正向, 0为负向
    """
    x_min = np.min(data)
    x_max = np.max(data)
    if direction == 1:
        return (data - x_min) / (x_max - x_min)
    else:
        return (x_max - data) / (x_max - x_min)

def z_score_scaling(data):
    return (data - np.mean(data)) / np.std(data)

def efficacy_coefficient(data, c=60, d=40):
    """
    c: 基础分, d: 档次分
    映射结果区间为 [c, c+d]
    """
    x_min = np.min(data)
    x_max = np.max(data)
    # 极值差部分
    r = (data - x_min) / (x_max - x_min)
    # 仿射变换
    return c + r * d