import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.7) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break

        ious = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        inds = torch.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    return torch.stack(keep)


def apply_deltas_to_anchors(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    widths = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
    heights = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx, dy, dw, dh = deltas.unbind(dim=1)
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw.clamp(max=4.0)) * widths
    pred_h = torch.exp(dh.clamp(max=4.0)) * heights

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h
    return torch.stack([x1, y1, x2, y2], dim=1)


def clip_boxes_to_image(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    h, w = image_size
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=w - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=h - 1)
    return boxes


def roi_pool_fallback(
    feature_map: torch.Tensor,
    proposals: torch.Tensor,
    output_size: int,
    spatial_scale: float,
) -> torch.Tensor:
    pooled = []
    _, _, h, w = feature_map.shape

    for box in proposals:
        x1, y1, x2, y2 = box * spatial_scale
        x1 = int(torch.floor(x1).item())
        y1 = int(torch.floor(y1).item())
        x2 = int(torch.ceil(x2).item())
        y2 = int(torch.ceil(y2).item())

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        region = feature_map[:, :, y1:y2, x1:x2]
        pooled.append(F.adaptive_max_pool2d(region, (output_size, output_size)))

    if not pooled:
        return torch.zeros(
            (0, feature_map.shape[1], output_size, output_size),
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
    return torch.cat(pooled, dim=0)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyBackbone(nn.Module):
    """输出 stride=16 的特征图。"""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, stride=2),
            ConvBNReLU(32, 64, stride=2),
            ConvBNReLU(64, 128, stride=2),
            ConvBNReLU(128, 256, stride=2),
            ConvBNReLU(256, 256, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class AnchorGenerator:
    def __init__(self, sizes=(64, 128, 256), aspect_ratios=(0.5, 1.0, 2.0), stride=16):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.stride = stride

    @property
    def num_anchors_per_location(self) -> int:
        return len(self.sizes) * len(self.aspect_ratios)

    def _base_anchors(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        anchors = []
        for size in self.sizes:
            area = float(size * size)
            for ratio in self.aspect_ratios:
                w = math.sqrt(area / ratio)
                h = ratio * w
                anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.tensor(anchors, device=device, dtype=dtype)

    def __call__(self, feature: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        _, _, fh, fw = feature.shape
        base = self._base_anchors(feature.device, feature.dtype)

        shifts_x = (torch.arange(fw, device=feature.device, dtype=feature.dtype) + 0.5) * self.stride
        shifts_y = (torch.arange(fh, device=feature.device, dtype=feature.dtype) + 0.5) * self.stride
        yy, xx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        centers = torch.stack([xx, yy, xx, yy], dim=-1).reshape(-1, 4)

        anchors = (centers[:, None, :] + base[None, :, :]).reshape(-1, 4)
        return clip_boxes_to_image(anchors, image_size)


class RPNHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.objectness = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.box_delta = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = F.relu(self.conv(x), inplace=True)
        obj = self.objectness(t)
        box = self.box_delta(t)
        return obj, box


class RoIHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, roi_size: int = 7, hidden_dim: int = 1024):
        super().__init__()
        self.roi_size = roi_size
        self.fc1 = nn.Linear(in_channels * roi_size * roi_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.cls_score = nn.Linear(hidden_dim, num_classes + 1)  # +1 背景
        self.bbox_pred = nn.Linear(hidden_dim, (num_classes + 1) * 4)

    def forward(self, roi_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = roi_feats.flatten(1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        cls_logits = self.cls_score(x)
        box_deltas = self.bbox_pred(x)
        return cls_logits, box_deltas


class FasterRCNN(nn.Module):
    def __init__(self, num_classes: int = 7, pre_nms_topk: int = 1200, post_nms_topk: int = 300):
        super().__init__()
        self.backbone = TinyBackbone()
        self.anchor_generator = AnchorGenerator()
        self.rpn = RPNHead(256, self.anchor_generator.num_anchors_per_location)
        self.roi_head = RoIHead(256, num_classes)

        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk

    def _generate_proposals(
        self,
        objectness: torch.Tensor,
        bbox_deltas: torch.Tensor,
        anchors: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        scores = objectness.sigmoid()
        num_topk = min(self.pre_nms_topk, scores.numel())
        topk_scores, topk_idx = torch.topk(scores, k=num_topk, largest=True)

        topk_deltas = bbox_deltas[topk_idx]
        topk_anchors = anchors[topk_idx]

        proposals = apply_deltas_to_anchors(topk_deltas, topk_anchors)
        proposals = clip_boxes_to_image(proposals, image_size)

        ws = proposals[:, 2] - proposals[:, 0]
        hs = proposals[:, 3] - proposals[:, 1]
        valid = (ws >= 2.0) & (hs >= 2.0)
        proposals = proposals[valid]
        topk_scores = topk_scores[valid]

        keep = nms(proposals, topk_scores, iou_threshold=0.7)
        keep = keep[: self.post_nms_topk]
        return proposals[keep]

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        b, _, h, w = images.shape
        image_size = (h, w)

        feature = self.backbone(images)

        obj_map, box_map = self.rpn(feature)
        obj_flat = obj_map.permute(0, 2, 3, 1).reshape(b, -1)
        box_flat = box_map.permute(0, 2, 3, 1).reshape(b, -1, 4)

        anchors = self.anchor_generator(feature, image_size)
        proposals_per_image: List[torch.Tensor] = []
        cls_logits_per_image: List[torch.Tensor] = []
        bbox_deltas_per_image: List[torch.Tensor] = []

        for i in range(b):
            proposals = self._generate_proposals(obj_flat[i], box_flat[i], anchors, image_size)
            proposals_per_image.append(proposals)

            feat_i = feature[i : i + 1]
            roi_feats = roi_pool_fallback(feat_i, proposals, output_size=7, spatial_scale=1.0 / 16.0)
            cls_logits, bbox_deltas = self.roi_head(roi_feats)
            cls_logits_per_image.append(cls_logits)
            bbox_deltas_per_image.append(bbox_deltas)

        return {
            "rpn_objectness": obj_flat,
            "rpn_bbox_deltas": box_flat,
            "proposals": proposals_per_image,
            "class_logits": cls_logits_per_image,
            "box_deltas": bbox_deltas_per_image,
        }


if __name__ == "__main__":
    model = FasterRCNN(num_classes=7)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)

    print("rpn_objectness:", tuple(out["rpn_objectness"].shape))
    print("rpn_bbox_deltas:", tuple(out["rpn_bbox_deltas"].shape))
    for i, (props, logits) in enumerate(zip(out["proposals"], out["class_logits"]), start=1):
        print(f"image {i}: proposals={tuple(props.shape)}, class_logits={tuple(logits.shape)}")
