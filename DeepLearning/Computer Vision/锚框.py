import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os


# ==========================================
# 1. 定义锚框生成函数
# ==========================================
def generate_anchors(center_x, center_y, scales, ratios):
    """
    在指定中心点生成不同尺度和比例的锚框
    :param center_x: 中心点 x
    :param center_y: 中心点 y
    :param scales: 尺寸列表 (比如 [128, 256, 512])
    :param ratios: 长宽比列表 (比如 [0.5, 1, 2])
    :return: 锚框列表 [(x, y, w, h), ...]
    """
    anchors = []

    for scale in scales:
        for ratio in ratios:
            # 计算宽和高
            # 如果面积 = scale^2，且 w/h = ratio
            # 那么 w = sqrt(scale^2 * ratio)
            #      h = scale^2 / w
            h = int(np.sqrt(scale ** 2 / ratio))
            w = int(h * ratio)

            # 计算左上角坐标 (x, y)
            x = center_x - w / 2
            y = center_y - h / 2

            anchors.append((x, y, w, h, scale, ratio))

    return anchors


# ==========================================
# 2. 主绘图逻辑
# ==========================================
def plot_anchor_boxes(image_path):
    if not os.path.exists(image_path):
        print(f"找不到图片: {image_path}")
        return

    # 读取图片
    img = Image.open(image_path)
    img_w, img_h = img.size

    # 设定中心点 (就选图片的正中心)
    cx, cy = img_w // 2, img_h // 2

    # --- 核心配置：定义锚框的形态 ---
    # Scales: 框的大小 (像素)
    # Ratios: 框的形状 (宽/高) -> 0.5是高瘦, 1是正方, 2是扁平
    scales = [100, 200, 350]
    ratios = [0.5, 1.0, 2.0]

    # 生成锚框数据
    anchors = generate_anchors(cx, cy, scales, ratios)

    # 开始绘图
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    # 画出中心点
    ax.scatter([cx], [cy], c='red', s=100, marker='+', linewidth=3, label="Center Point")

    # 遍历并画出每一个锚框
    # 为了好看，我们用不同的颜色代表不同的比例
    colors = {0.5: 'yellow', 1.0: 'cyan', 2.0: 'lime'}

    for (x, y, w, h, s, r) in anchors:
        color = colors[r]

        # 创建矩形
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor='none',
            linestyle='--'  # 虚线，表示这是虚拟的框
        )
        ax.add_patch(rect)

        # (可选) 在框上写文字
        if s == max(scales):  # 只在大框上写字，防止太乱
            ax.text(x, y, f'Ratio {r}', color=color, fontsize=10, backgroundcolor='black')

    # 图例设置
    # 创建虚拟的图例句柄
    handles = [
        patches.Patch(color='yellow', label='Ratio 1:2 (Tall)'),
        patches.Patch(color='cyan', label='Ratio 1:1 (Square)'),
        patches.Patch(color='lime', label='Ratio 2:1 (Wide)')
    ]
    ax.legend(handles=handles, loc='upper right')

    plt.title(f"Anchor Boxes Visualization (Center: {cx}, {cy})")
    plt.show()


if __name__ == "__main__":
    # 使用你的图片文件
    img_file = "cat_dog.png"
    plot_anchor_boxes(img_file)
