import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # 专门用来画框的库
import os

# ====================================================
# 1. 设置颜色规则 (复刻你的图片风格)
# ====================================================
# COCO数据集里：17是猫，18是狗
COLOR_MAP = {
    17: 'red',  # Cat -> 红色
    18: 'blue'  # Dog -> 蓝色
}


# ====================================================
# 2. 加载模型
# ====================================================
def get_model():
    # 加载预训练的 Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    model.eval()
    return model


# ====================================================
# 3. 核心：使用 Matplotlib 画框
# ====================================================
def detect_and_plot(image_path, model, threshold=0.8):
    if not os.path.exists(image_path):
        print(f"错误: 找不到文件 {image_path}")
        return

    # --- A. 图像处理 ---
    # 使用 PIL 读取 (不做任何 OpenCV 转换)
    img_pil = Image.open(image_path).convert("RGB")

    # 转为 Tensor
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img_pil).unsqueeze(0)

    # --- B. 推理 ---
    print("正在检测...")
    with torch.no_grad():
        prediction = model(img_tensor)[0]

    # --- C. Matplotlib 绘图 ---
    # 创建画布
    fig, ax = plt.subplots(1, figsize=(10, 8))

    # 显示原图
    ax.imshow(img_pil)

    # 遍历检测结果
    for i in range(len(prediction['boxes'])):
        score = prediction['scores'][i].item()
        label = prediction['labels'][i].item()

        # 1. 过滤置信度
        if score > threshold:
            # 2. 过滤类别 (我们只关心猫和狗)
            if label in COLOR_MAP:
                box = prediction['boxes'][i].detach().cpu().numpy()
                x1, y1, x2, y2 = box

                # Matplotlib 的 Rectangle 需要 (左上角x, 左上角y, 宽度, 高度)
                width = x2 - x1
                height = y2 - y1

                # 获取对应颜色
                box_color = COLOR_MAP[label]

                # 3. 创建矩形补丁 (Patch)
                # linewidth=2: 线宽
                # edgecolor: 边框颜色
                # facecolor='none':以此保证框里面是透明的
                rect = patches.Rectangle(
                    (x1, y1),
                    width, height,
                    linewidth=2,
                    edgecolor=box_color,
                    facecolor='none'
                )

                # 将矩形添加到图里
                ax.add_patch(rect)

                print(f"-> 发现目标 (类别ID: {label}), 颜色: {box_color}, 置信度: {score:.2f}")

    # 显示结果
    plt.title("Object Detection (Red: Cat, Blue: Dog)")
    plt.show()


# ====================================================
# 4. 运行
# ====================================================
if __name__ == "__main__":
    # 请确保你的图片名字是这个，或者修改这里
    image_file = "cat_dog.png"

    model = get_model()
    detect_and_plot(image_file, model, threshold=0.8)
