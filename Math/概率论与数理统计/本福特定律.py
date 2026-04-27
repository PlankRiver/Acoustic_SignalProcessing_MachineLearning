import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

def first_num(x):
    """获取数字的首位数字"""
    while x >= 10:
        x //= 10
    return x

if __name__ == '__main__':
    # 计算0!到1000!的数据
    n = 1
    frequency = [0] * 9
    for i in range(1, 1001):
        n *= i
        m = first_num(n)
        frequency[m - 1] += 1
    
    # 转换为概率
    frequency = [i / sum(frequency) for i in frequency]
    
    # 计算本福特定律的理论值
    benford_theory = [np.log10(i + 1) - np.log10(i) for i in range(1, 10)]
    
    print("实际频率:", frequency)
    print("理论频率:", benford_theory)
    
    # 绘制对比图
    x_pos = np.arange(9) + 1
    plt.figure(figsize=(10, 6))
    plt.plot(x_pos, frequency, 'ro-', linewidth=2, label='实际频率', markersize=8)
    plt.plot(x_pos, benford_theory, 'b^--', linewidth=2, label='理论频率', markersize=8)
    plt.xlabel('首位数字', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.title('本福特定律：0!~1000!的首位数字分布', fontsize=14)
    plt.xticks(x_pos)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()