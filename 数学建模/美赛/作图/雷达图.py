import numpy as np
import matplotlib.pyplot as plt

# 1. 设置全局字体（美赛建议使用 Serif 字体，如 Times New Roman）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'bold'

# 2. 准备数据
labels = np.array([
    'Yield strength', 'Ultimate tensile strength', 'Elongation to fracture',
    'UTS×EF', 'Saturation induction', 'Coercivity', 'Electrical resistivity'
])
n_attr = len(labels)

# 各组方案的数据（需根据实际量程归一化或直接填入）
# 注意：雷达图需要“闭合”，所以要在末尾重复第一个值
data_m_mca = np.array([9.5, 9.2, 8.8, 9.6, 7.8, 8.2, 8.5])
data_m_mca = np.concatenate((data_m_mca, [data_m_mca[0]]))

data_fe_49co = np.array([4.2, 4.8, 3.5, 4.0, 7.5, 5.0, 4.5])
data_fe_49co = np.concatenate((data_fe_49co, [data_fe_49co[0]]))

data_fe_78 = np.array([6.5, 5.8, 5.2, 5.5, 5.8, 6.2, 3.2])
data_fe_78 = np.concatenate((data_fe_78, [data_fe_78[0]]))

data_fe_4si = np.array([1.5, 4.2, 6.5, 2.5, 4.2, 9.0, 2.8])
data_fe_4si = np.concatenate((data_fe_4si, [data_fe_4si[0]]))

# 计算角度并闭合
angles = np.linspace(0, 2*np.pi, n_attr, endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

# 3. 创建画布
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'), dpi=150)

# 4. 绘图（设置颜色、线宽、标志位、填充透明度）
# M-MCA (蓝色)
ax.plot(angles, data_m_mca, color='#4169E1', linewidth=2.5, marker='v', markersize=10, label='M-MCA')
ax.fill(angles, data_m_mca, color='#4169E1', alpha=0.1)

# Fe-49Co-2V (深红色)
ax.plot(angles, data_fe_49co, color='#F08080', linewidth=2.5, marker='o', markersize=10, label='Fe-49Co-2V')
ax.fill(angles, data_fe_49co, color='#F08080', alpha=0.1)

# Fe-78.5Ni (淡红色)
ax.plot(angles, data_fe_78, color='#E9967A', linewidth=2.5, marker='^', markersize=10, label='Fe-78.5Ni')
ax.fill(angles, data_fe_78, color='#E9967A', alpha=0.1)

# Fe-4Si (浅蓝色)
ax.plot(angles, data_fe_4si, color='#87CEEB', linewidth=2.5, marker='s', markersize=10, label='Fe-4Si')
ax.fill(angles, data_fe_4si, color='#87CEEB', alpha=0.1)

# 5. 美化坐标轴
# 设置起始角度（让第一个指标在正上方）
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1) # 顺时针排列

# 设置指标标签
plt.xticks(angles[:-1], labels, fontsize=12, fontweight='bold')

# 设置径向刻度（原图刻度为 1, 3, 6, 8, 10）
ax.set_rlabel_position(0) # 刻度数字的位置
plt.yticks([1, 3, 6, 8, 10], ["1", "3", "6", "8", "10"], color="black", size=12, fontweight='bold')
plt.ylim(0, 10)

# 设置网格线（虚线风格）
ax.grid(True, linestyle='--', alpha=0.6)

# 隐藏最外圈的圆形边框（可选，还原学术风）
ax.spines['polar'].set_visible(False)

# 6. 添加图例
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 0.8), frameon=False, fontsize=12)

# 7. 布局优化
plt.tight_layout()
# plt.savefig('radar_chart.pdf', bbox_inches='tight') # 建议保存为PDF
plt.show()