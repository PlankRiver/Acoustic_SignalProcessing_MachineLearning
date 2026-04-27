import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# 1. 辅助模块：L2 Normalization
# SSD 在 Conv4_3 层使用了 L2 正规化，因为该层特征数值较大
# ==========================================================
class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        # 这是一个可学习的参数
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


# ==========================================================
# 2. SSD 主体网络类
# 包含：VGG骨干 + Extra Layers + Prediction Heads
# ==========================================================
class SSD(nn.Module):
    def __init__(self, num_classes, backbone, extras, head):
        super(SSD, self).__init__()
        self.num_classes = num_classes

        # 1. 骨干网络 (VGG)
        self.vgg = nn.ModuleList(backbone)

        # 2. L2 正规化层 (专门用于 conv4_3)
        self.L2Norm = L2Norm(512, 20)

        # 3. 额外的卷积层 (用于获取更小尺度的特征图)
        self.extras = nn.ModuleList(extras)

        # 4. 回归头 (Loc) 和 分类头 (Conf)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

    def forward(self, x):
        """
        x: 输入图像 (Batch, 3, 300, 300)
        """
        sources = []  # 存放用于预测的特征层 (一共6个)
        loc = []  # 存放位置预测结果
        conf = []  # 存放类别预测结果

        # ------------------------------------------------
        # 第一阶段：运行 VGG 骨干，直到 conv4_3
        # ------------------------------------------------
        for k in range(23):
            x = self.vgg[k](x)

        # 保存 Conv4_3 的输出 (第1个特征图)，需要做 L2 Norm
        s = self.L2Norm(x)
        sources.append(s)

        # ------------------------------------------------
        # 第二阶段：运行剩下的 VGG，直到 FC7 (转为卷积层)
        # ------------------------------------------------
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        # 保存 Conv7 的输出 (第2个特征图)
        sources.append(x)

        # ------------------------------------------------
        # 第三阶段：运行 Extra Layers，获取剩下4个特征图
        # ------------------------------------------------
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            # 每隔一层卷积做一次 stride=2 的下采样，作为特征源
            if k % 2 == 1:
                sources.append(x)

        # sources 列表现在包含了6个不同尺度的特征图
        # Shape 示例: [38x38, 19x19, 10x10, 5x5, 3x3, 1x1]

        # ------------------------------------------------
        # 第四阶段：在每个特征图上应用 Loc 和 Conf 卷积头
        # ------------------------------------------------
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # l(x) -> (Batch, 4 * num_anchors, H, W)
            # c(x) -> (Batch, num_classes * num_anchors, H, W)

            # 变形为 (Batch, H*W*num_anchors, ...) 以便最后拼接
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # 拼接所有特征图的预测结果
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # 最终输出 reshape
        # loc: (Batch, 总锚框数, 4)
        # conf: (Batch, 总锚框数, 类别数)
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
        )
        return output


# ==========================================================
# 3. 构建辅助函数：生成具体的层结构
# ==========================================================

# VGG 配置 (数字是通道数，'M'是最大池化，'C'是Ceil模式的池化)
vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]


def make_vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    # 这一块是 SSD 特有的：把 VGG 的全连接层变成了卷积层 (Conv6, Conv7)
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def make_extras(cfg, i, batch_norm=False):
    # 为 VGG 后面添加额外的下采样层
    layers = []
    in_channels = i
    # 这里的配置略微简化，实际上是 1x1卷积 -> 3x3卷积(stride=2) 的组合
    # 256 -> 512, 128 -> 256, 128 -> 256, 128 -> 256
    cfg = [256, 512, 128, 256, 128, 256, 128, 256]

    for k, v in enumerate(cfg):
        if k % 2 == 0:
            layers += [nn.Conv2d(in_channels, v, kernel_size=1)]
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, stride=2, padding=1)]
            in_channels = v
    return layers


def make_multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    # 获取需要做预测的特征图通道数
    # vgg[21]是conv4_3, vgg[-2]是conv7
    vgg_source = [21, -2]

    # 1. 处理 VGG 内的特征层
    for k, v in enumerate(vgg_source):
        # cfg[k] 是该层的锚框数量 (比如 4 或 6)
        loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]

    # 2. 处理 Extra Layers 内的特征层
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]

    return vgg, extra_layers, (loc_layers, conf_layers)


# ==========================================================
# 4. 实例化模型的接口
# ==========================================================
def build_ssd(num_classes=21):
    # 这里定义每个特征图上，每个像素点生成几个锚框
    # [Conv4_3, Conv7, Extra1, Extra2, Extra3, Extra4]
    mbox = [4, 6, 6, 6, 4, 4]

    base = make_vgg(vgg_config, 3)
    extras = make_extras(None, 1024)

    # 组装所有部分
    base, extras, head = make_multibox(base, extras, mbox, num_classes)

    return SSD(num_classes, base, extras, head)


# ==========================================================
# 测试代码
# ==========================================================
if __name__ == "__main__":
    # 创建一个 SSD 模型，假设有 20 个物体类别 + 1 个背景 = 21 类
    ssd_net = build_ssd(num_classes=21)

    # 创建一个随机的输入图像 (Batch=1, RGB, 300x300)
    dummy_input = torch.randn(1, 3, 300, 300)

    # 前向传播
    locs, confs = ssd_net(dummy_input)

    print("Layer Architecture Constructed!")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Loc Output Shape: {locs.shape}")  # 应该是 (1, 8732, 4)
    print(f"Conf Output Shape: {confs.shape}")  # 应该是 (1, 8732, 21)
    print("\n解释: 8732 是 SSD300 在所有特征图上生成的锚框总数。")
