import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ============================================================
# 1. 配置区域
# ============================================================

# 【⚠️重要】这个列表必须和你【训练时】的类别顺序完全一致（字母顺序）
# 如果你的训练集文件夹名字和这里不一样，请修改这里！
class_names = [
    'Audi',
    'Hyundai Creta',
    'Mahindra Scorpio',
    'Rolls Royce',
    'Swift',
    'Tata Safari',
    'Toyota Innova'
]

# 文件夹和模型路径 (根据你的截图设置)
# 假设脚本放在 "Hardware and Training" 目录下
test_dir = 'Cars Prediction'  # 存放测试图片的文件夹
model_path = 'my_car_model.pth'  # 训练好的模型


# ============================================================
# 2. 定义模型结构 (必须与训练时一致)
# ============================================================
def get_model(num_classes):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


# ============================================================
# 3. 核心逻辑：批量预测
# ============================================================
def predict_batch():
    # --- A. 准备环境 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # --- B. 加载模型 (只加载一次) ---
    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到模型文件 {model_path}")
        return

    print("正在加载模型...")
    model = get_model(len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # 开启评估模式
    print("模型加载完成！开始批量推理...\n")

    # --- C. 定义预处理 ---
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inference_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # --- D. 读取文件夹中的所有图片 ---
    if not os.path.exists(test_dir):
        print(f"❌ 错误：找不到文件夹 {test_dir}")
        return

    # 获取所有 jpg/png 图片
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print("文件夹里没有找到图片！")
        return

    print(f"{'文件名':<25} | {'预测结果':<20} | {'置信度':<10}")
    print("-" * 65)

    # --- E. 循环推理 ---
    for img_name in image_files:
        img_full_path = os.path.join(test_dir, img_name)

        try:
            # 1. 读取图片
            img_raw = Image.open(img_full_path).convert('RGB')

            # 2. 转换格式并增加维度 [1, 3, 224, 224]
            img_tensor = inference_transform(img_raw).unsqueeze(0).to(device)

            # 3. 推理
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

                pred_class = class_names[predicted_idx.item()]
                conf_score = confidence.item()

            # 4. 打印结果
            # 为了美观，如果文件名太长截断一下
            display_name = (img_name[:22] + '..') if len(img_name) > 22 else img_name
            print(f"{display_name:<25} | {pred_class:<20} | {conf_score * 100:.1f}%")

        except Exception as e:
            print(f"{img_name:<25} | ❌ 读取或预测出错: {e}")

    print("-" * 65)
    print("全部完成。")


if __name__ == '__main__':
    predict_batch()
