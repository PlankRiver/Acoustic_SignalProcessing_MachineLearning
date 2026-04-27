import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os
import time


# ============================================================
# 1. 辅助函数
# ============================================================

def get_model(num_classes):
    print("正在加载模型并配置分层微调策略...")
    # 下载预训练模型
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 1. 先冻结所有参数 (Lock everything)
    for param in model.parameters():
        param.requires_grad = False

    # 2. 【核心修改】解冻 Layer4
    # 让 ResNet 的最高级卷积层参与学习，适应汽车的特征
    print("-> 解冻 Layer4 (用于学习更高级的汽车特征)")
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 3. 替换全连接层 (New Head)
    # 新创建的层默认就是 requires_grad=True
    print(f"-> 重置分类头 (输出类别: {num_classes})")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def train_model(model, loss_fn, optimizer, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 记录最佳准确率
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    return model


# ============================================================
# 2. 主程序
# ============================================================

if __name__ == '__main__':
    # 配置
    data_dir = os.path.join(os.path.dirname(__file__), 'Cars Dataset')
    batch_size = 32
    num_workers = 4
    num_epochs = 25

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ============================================================
    # 数据增强保持不变 (这部分你之前写得很好)
    # ============================================================
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    # 加载数据
    if not os.path.exists(os.path.join(data_dir, 'train')):
        print(f"报错：找不到路径 {os.path.join(data_dir, 'train')}")
        exit()

    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
        'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    print(f"检测到的类别: {class_names}")

    # 初始化模型 (这里会调用修改后的 get_model)
    model = get_model(len(class_names))
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    # ============================================================
    # 【重要修改】优化器配置
    # ============================================================
    # 1. 学习率改为 0.0001 (比之前小10倍)，因为 layer4 已经有基础了，不能大改
    # 2. filter(...) 会自动找出 requires_grad=True 的参数 (即 layer4 + fc)
    print("配置优化器：学习率 0.0001，优化 Layer4 + FC")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    # 开始训练
    trained_model = train_model(
        model,
        loss_fn,
        optimizer,
        dataloaders,
        dataset_sizes,
        device,
        num_epochs
    )

    # 保存模型
    save_path = 'my_car_model.pth'
    torch.save(trained_model.state_dict(), save_path)
    print(f"新模型已保存为 {save_path}")
