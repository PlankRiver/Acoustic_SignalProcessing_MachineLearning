import torch
import torch.nn as nn


class ConvBNLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = channels // 2
        self.conv1 = ConvBNLeaky(channels, hidden, 1)
        self.conv2 = ConvBNLeaky(hidden, channels, 3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


def make_residual_stage(in_channels, out_channels, num_blocks):
    layers = [ConvBNLeaky(in_channels, out_channels, 3, stride=2)]
    for _ in range(num_blocks):
        layers.append(ResidualBlock(out_channels))
    return nn.Sequential(*layers)


class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBNLeaky(3, 32, 3)
        self.stage1 = make_residual_stage(32, 64, 1)
        self.stage2 = make_residual_stage(64, 128, 2)
        self.stage3 = make_residual_stage(128, 256, 8)   # 52x52
        self.stage4 = make_residual_stage(256, 512, 8)   # 26x26
        self.stage5 = make_residual_stage(512, 1024, 4)  # 13x13

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        feat_small = self.stage3(x)
        feat_medium = self.stage4(feat_small)
        feat_large = self.stage5(feat_medium)
        return feat_small, feat_medium, feat_large


class DetectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        half = out_channels // 2
        self.layers = nn.Sequential(
            ConvBNLeaky(in_channels, half, 1),
            ConvBNLeaky(half, out_channels, 3),
            ConvBNLeaky(out_channels, half, 1),
            ConvBNLeaky(half, out_channels, 3),
            ConvBNLeaky(out_channels, half, 1),
        )

    def forward(self, x):
        return self.layers(x)


class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        out_channels = num_anchors * (num_classes + 5)
        self.pred = nn.Sequential(
            ConvBNLeaky(in_channels, in_channels * 2, 3),
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.pred(x)


class YOLOv3(nn.Module):
    """
    YOLOv3 model (backbone + FPN + 3 detection heads).
    Forward returns raw predictions at 3 scales:
      - p_large:  [B, A*(5+C), 13, 13]
      - p_medium: [B, A*(5+C), 26, 26]
      - p_small:  [B, A*(5+C), 52, 52]
    """

    def __init__(self, num_classes=80, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.backbone = Darknet53()

        self.det_large = DetectionBlock(1024, 1024)
        self.head_large = YOLOHead(512, num_anchors, num_classes)

        self.reduce1 = ConvBNLeaky(512, 256, 1)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")

        self.det_medium = DetectionBlock(256 + 512, 512)
        self.head_medium = YOLOHead(256, num_anchors, num_classes)

        self.reduce2 = ConvBNLeaky(256, 128, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")

        self.det_small = DetectionBlock(128 + 256, 256)
        self.head_small = YOLOHead(128, num_anchors, num_classes)

    def forward(self, x):
        feat_small, feat_medium, feat_large = self.backbone(x)

        x_large = self.det_large(feat_large)
        p_large = self.head_large(x_large)

        x_medium_in = self.up1(self.reduce1(x_large))
        x_medium_in = torch.cat([x_medium_in, feat_medium], dim=1)
        x_medium = self.det_medium(x_medium_in)
        p_medium = self.head_medium(x_medium)

        x_small_in = self.up2(self.reduce2(x_medium))
        x_small_in = torch.cat([x_small_in, feat_small], dim=1)
        x_small = self.det_small(x_small_in)
        p_small = self.head_small(x_small)

        return p_large, p_medium, p_small


if __name__ == "__main__":
    model = YOLOv3(num_classes=80, num_anchors=3)
    x = torch.randn(2, 3, 416, 416)
    outputs = model(x)
    for i, out in enumerate(outputs, start=1):
        print(f"Output {i} shape: {tuple(out.shape)}")
