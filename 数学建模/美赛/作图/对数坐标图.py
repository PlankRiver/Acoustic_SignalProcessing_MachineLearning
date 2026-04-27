import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatterSciNotation

# 1. 准备数据
x = np.linspace(1, 30, 100)
y1 = x**3           # 模拟蓝线 (幂增长)
y2 = 10 * x**2      # 模拟红线

# 2. 设置全局绘图风格 (美赛推荐清爽风格)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 创建一个 2x2 的画布
fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=150)

# --- (1) 左上: 原直角坐标 (Linear Plot) ---
ax = axs[0, 0]
ax.plot(x, y1, color='#0072BD', linewidth=2, label='Model A')
ax.plot(x, y2, color='#D95319', linewidth=2, label='Model B')
ax.set_title('Original Linear Scale', fontsize=14, fontweight='bold')
ax.grid(True, which='major', linestyle='-', alpha=0.6)
# 设置 y 轴使用科学计数法 (10^4)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# --- (2) 右上: X对数坐标 (Semilog-X) ---
ax = axs[0, 1]
ax.semilogx(x, y1, color='#0072BD', linewidth=2)
ax.semilogx(x, y2, color='#D95319', linewidth=2)
ax.set_title('Semilog-X Scale', fontsize=14, fontweight='bold')
# 开启主次网格线
ax.grid(True, which='major', linestyle='-', alpha=0.6)
ax.grid(True, which='minor', linestyle=':', alpha=0.3)
ax.set_xlim(1, 100)

# --- (3) 左下: Y对数坐标 (Semilog-Y) ---
ax = axs[1, 0]
ax.semilogy(x, y1, color='#0072BD', linewidth=2)
ax.semilogy(x, y2, color='#D95319', linewidth=2)
ax.set_title('Semilog-Y Scale', fontsize=14, fontweight='bold')
ax.grid(True, which='major', linestyle='-', alpha=0.6)
ax.grid(True, which='minor', linestyle=':', alpha=0.3)
ax.set_ylim(1, 10**5)

# --- (4) 右下: X-Y双对数坐标 (Log-Log) ---
ax = axs[1, 1]
ax.loglog(x, y1, color='#0072BD', linewidth=2)
ax.loglog(x, y2, color='#D95319', linewidth=2)
ax.set_title('Log-Log Scale', fontsize=14, fontweight='bold')
ax.grid(True, which='major', linestyle='-', alpha=0.6)
ax.grid(True, which='minor', linestyle=':', alpha=0.3)
ax.set_xlim(1, 100)
ax.set_ylim(1, 10**5)

# 统一细节美化
for ax in axs.flat:
    ax.tick_params(direction='in', top=True, right=True)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
# plt.savefig('log_comparison.pdf', bbox_inches='tight') # 建议保存为PDF
plt.show()