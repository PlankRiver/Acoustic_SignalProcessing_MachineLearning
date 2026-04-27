import torch
from torch import nn
from torchvision import transforms
from PIL import Image  # 需要安装 pillow 库: pip install pillow
import matplotlib.pyplot as plt


# ==========================================
# 1. 必须重新定义模型结构
# (必须与训练时的类定义完全一致)
# ==========================================
class SoftmaxRegression(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)


# ==========================================
# 2. 设置环境与加载模型
# ==========================================
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# 实例化一个“空”模型
model = SoftmaxRegression(num_inputs=784, num_outputs=10).to(device)

# 加载保存的参数
model_path = "softmax_fashion_mnist.pth"
try:
    # weights_only=True 是为了安全，只加载权重数据
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"成功加载模型: {model_path}")
except FileNotFoundError:
    print(f"错误: 找不到文件 {model_path}，请确保路径正确。")
    exit()

# 设置为评估模式 (非常重要！虽然这个简单模型没 Dropout/BN，但要养成好习惯)
model.eval()

# FashionMNIST 的标签映射
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ==========================================
# 3. 预测一张真实的图片 (推理函数)
# ==========================================
def predict_image(image_path):
    """
    读取一张图片文件并进行预测
    """
    # A. 定义预处理流程
    # 1. 转为灰度图 (FashionMNIST 是灰度的)
    # 2. 缩放到 28x28
    # 3. 转为 Tensor 并归一化到 [0, 1]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    try:
        # B. 读取图片
        img = Image.open(image_path)

        # C. 预处理
        img_tensor = transform(img)  # 现在的形状是 [1, 28, 28]

        # D. 增加 Batch 维度 [1, 28, 28] -> [1, 1, 28, 28]
        # PyTorch 模型一次是处理一批数据的，哪怕只有一张图，也要伪装成一批
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # E. 推理
        with torch.no_grad():
            output = model(img_tensor)
            # output 是一个长度为 10 的向量，包含每个类别的分数

            # 使用 softmax 算出概率 (可选，方便看置信度)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

            # 找到最大值的索引
            predicted_idx = output.argmax(1).item()
            predicted_label = classes[predicted_idx]
            confidence = probabilities[predicted_idx].item()

        print(f"图片: {image_path}")
        print(f"预测结果: {predicted_label} (置信度: {confidence * 100:.2f}%)")

        # (可选) 显示图片
        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {predicted_label}")
        plt.show()

    except Exception as e:
        print(f"预测出错: {e}")


# ==========================================
# 4. 执行预测
# ==========================================
# 你可以在这里找一张本地的图片路径，比如 'my_shirt.jpg'
# 如果你没有图片，我们可以临时生成一张随机噪声图来演示流程
import numpy as np
from PIL import Image

# --- 模拟生成一张图片用于测试 ---
random_array = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
dummy_img = Image.fromarray(random_array)
dummy_img.save("test_input.png")
# -----------------------------

print("\n开始预测...")
predict_image("test_input.png")