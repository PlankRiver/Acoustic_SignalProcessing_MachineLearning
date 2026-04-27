from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SSDBackbone(nn.Module):
    """轻量 backbone，输出 SSD 的多尺度特征。"""

    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBNReLU(3, 32, stride=2),
            ConvBNReLU(32, 64),
            ConvBNReLU(64, 64),
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(64, 128, stride=2),
            ConvBNReLU(128, 128),
        )
        self.stage3 = nn.Sequential(
            ConvBNReLU(128, 256, stride=2),
            ConvBNReLU(256, 256),
        )
        self.stage4 = nn.Sequential(
            ConvBNReLU(256, 512, stride=2),
            ConvBNReLU(512, 512),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stage1(x)   # 150x150
        x = self.stage2(x)   # 75x75
        feat1 = self.stage3(x)  # 38x38
        feat2 = self.stage4(feat1)  # 19x19
        return feat1, feat2


class ExtraLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNReLU(512, 256, kernel_size=1, padding=0),
                    ConvBNReLU(256, 512, stride=2),
                ),  # 10x10
                nn.Sequential(
                    ConvBNReLU(512, 128, kernel_size=1, padding=0),
                    ConvBNReLU(128, 256, stride=2),
                ),  # 5x5
                nn.Sequential(
                    ConvBNReLU(256, 128, kernel_size=1, padding=0),
                    ConvBNReLU(128, 256, kernel_size=3, stride=1, padding=0),
                ),  # 3x3
                nn.Sequential(
                    ConvBNReLU(256, 128, kernel_size=1, padding=0),
                    ConvBNReLU(128, 256, kernel_size=3, stride=1, padding=0),
                ),  # 1x1
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats


class MultiBoxHead(nn.Module):
    def __init__(self, in_channels: List[int], num_defaults: List[int], num_classes: int):
        super().__init__()
        self.loc_heads = nn.ModuleList()
        self.cls_heads = nn.ModuleList()

        for c, d in zip(in_channels, num_defaults):
            self.loc_heads.append(nn.Conv2d(c, d * 4, kernel_size=3, padding=1))
            self.cls_heads.append(nn.Conv2d(c, d * num_classes, kernel_size=3, padding=1))

        self.num_classes = num_classes

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        locs, confs = [], []

        for feat, loc_head, cls_head in zip(features, self.loc_heads, self.cls_heads):
            loc = loc_head(feat).permute(0, 2, 3, 1).contiguous()
            conf = cls_head(feat).permute(0, 2, 3, 1).contiguous()

            b = feat.shape[0]
            locs.append(loc.view(b, -1, 4))
            confs.append(conf.view(b, -1, self.num_classes))

        return torch.cat(locs, dim=1), torch.cat(confs, dim=1)


class PriorBoxGenerator:
    """生成 SSD300 默认框，格式为 cx, cy, w, h（归一化到 0~1）。"""

    def __init__(self):
        self.feature_sizes = [38, 19, 10, 5, 3, 1]
        self.strides = [8, 16, 30, 60, 100, 300]
        self.scales = [0.10, 0.20, 0.37, 0.54, 0.71, 0.88, 1.05]
        self.aspect_ratios = [
            [2],
            [2, 3],
            [2, 3],
            [2, 3],
            [2],
            [2],
        ]

    def __call__(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        priors = []

        for k, f in enumerate(self.feature_sizes):
            scale = self.scales[k]
            next_scale = self.scales[k + 1]

            for i in range(f):
                for j in range(f):
                    cx = (j + 0.5) / f
                    cy = (i + 0.5) / f

                    priors.append([cx, cy, scale, scale])
                    s_prime = (scale * next_scale) ** 0.5
                    priors.append([cx, cy, s_prime, s_prime])

                    for ar in self.aspect_ratios[k]:
                        r = ar ** 0.5
                        priors.append([cx, cy, scale * r, scale / r])
                        priors.append([cx, cy, scale / r, scale * r])

        priors = torch.tensor(priors, device=device, dtype=dtype)
        return priors.clamp(min=0.0, max=1.0)


class SSD300(nn.Module):
    """
    输入默认 300x300，输出:
    - loc: [B, N, 4]
    - conf: [B, N, num_classes]
    - priors: [N, 4]
    """

    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = SSDBackbone()
        self.extras = ExtraLayers()

        self.head = MultiBoxHead(
            in_channels=[256, 512, 512, 256, 256, 256],
            num_defaults=[4, 6, 6, 6, 4, 4],
            num_classes=num_classes,
        )

        self.prior_generator = PriorBoxGenerator()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat1, feat2 = self.backbone(x)
        extra_feats = self.extras(feat2)

        features = [feat1, feat2] + extra_feats
        loc, conf = self.head(features)
        priors = self.prior_generator(device=x.device, dtype=x.dtype)

        return {
            "loc": loc,
            "conf": conf,
            "priors": priors,
        }


if __name__ == "__main__":
    model = SSD300(num_classes=8)
    x = torch.randn(2, 3, 300, 300)
    out = model(x)

    print("loc:", tuple(out["loc"].shape))
    print("conf:", tuple(out["conf"].shape))
    print("priors:", tuple(out["priors"].shape))
