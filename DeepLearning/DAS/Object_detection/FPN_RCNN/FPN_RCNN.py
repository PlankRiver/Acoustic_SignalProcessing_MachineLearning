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


def roi_pool_fallback(feature_map: torch.Tensor, proposals: torch.Tensor, output_size: int, spatial_scale: float) -> torch.Tensor:
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
        return torch.zeros((0, feature_map.shape[1], output_size, output_size), device=feature_map.device, dtype=feature_map.dtype)
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


class TinyCBackbone(nn.Module):
    """输出 C3/C4/C5: stride 8/16/32。"""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, stride=2),
            ConvBNReLU(32, 64, stride=2),
        )
        self.layer3 = nn.Sequential(
            ConvBNReLU(64, 128, stride=2),
            ConvBNReLU(128, 128),
        )
        self.layer4 = nn.Sequential(
            ConvBNReLU(128, 256, stride=2),
            ConvBNReLU(256, 256),
        )
        self.layer5 = nn.Sequential(
            ConvBNReLU(256, 512, stride=2),
            ConvBNReLU(512, 512),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c3 = self.layer3(x)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        return c3, c4, c5


class FPN(nn.Module):
    def __init__(self, in_channels=(128, 256, 512), out_channels=256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels])
        self.smooth = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels])

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p5 = self.lateral[2](c5)
        p4 = self.lateral[1](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral[0](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        p5 = self.smooth[2](p5)
        p4 = self.smooth[1](p4)
        p3 = self.smooth[0](p3)
        return p3, p4, p5


class PyramidAnchorGenerator:
    def __init__(self, sizes=(32, 64, 128), strides=(8, 16, 32), aspect_ratios=(0.5, 1.0, 2.0)):
        self.sizes = sizes
        self.strides = strides
        self.aspect_ratios = aspect_ratios

    @property
    def num_anchors_per_location(self) -> int:
        return len(self.aspect_ratios)

    def _anchors_for_level(self, feature: torch.Tensor, size: int, stride: int, image_size: Tuple[int, int]) -> torch.Tensor:
        _, _, fh, fw = feature.shape
        device, dtype = feature.device, feature.dtype

        base = []
        area = float(size * size)
        for ratio in self.aspect_ratios:
            w = math.sqrt(area / ratio)
            h = ratio * w
            base.append([-w / 2, -h / 2, w / 2, h / 2])
        base = torch.tensor(base, device=device, dtype=dtype)

        shifts_x = (torch.arange(fw, device=device, dtype=dtype) + 0.5) * stride
        shifts_y = (torch.arange(fh, device=device, dtype=dtype) + 0.5) * stride
        yy, xx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        centers = torch.stack([xx, yy, xx, yy], dim=-1).reshape(-1, 4)

        anchors = (centers[:, None, :] + base[None, :, :]).reshape(-1, 4)
        return clip_boxes_to_image(anchors, image_size)

    def __call__(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], image_size: Tuple[int, int]) -> List[torch.Tensor]:
        anchors = []
        for feat, size, stride in zip(features, self.sizes, self.strides):
            anchors.append(self._anchors_for_level(feat, size, stride, image_size))
        return anchors


class RPNHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.objectness = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.box_delta = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        obj_maps, box_maps = [], []
        for x in features:
            t = F.relu(self.conv(x), inplace=True)
            obj_maps.append(self.objectness(t))
            box_maps.append(self.box_delta(t))
        return obj_maps, box_maps


class RoIHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, roi_size: int = 7, hidden_dim: int = 1024):
        super().__init__()
        self.roi_size = roi_size
        self.fc1 = nn.Linear(in_channels * roi_size * roi_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.cls_score = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_pred = nn.Linear(hidden_dim, (num_classes + 1) * 4)

    def forward(self, roi_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = roi_feats.flatten(1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        return self.cls_score(x), self.bbox_pred(x)


class FPNRCNN(nn.Module):
    def __init__(self, num_classes: int = 7, pre_nms_topk: int = 2000, post_nms_topk: int = 400):
        super().__init__()
        self.backbone = TinyCBackbone()
        self.fpn = FPN()
        self.anchor_generator = PyramidAnchorGenerator()
        self.rpn = RPNHead(256, self.anchor_generator.num_anchors_per_location)
        self.roi_head = RoIHead(256, num_classes)

        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.fpn_strides = (8, 16, 32)

    def _flatten_rpn_outputs(self, obj_maps: List[torch.Tensor], box_maps: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        b = obj_maps[0].shape[0]
        obj_all, box_all = [], []

        for obj_map, box_map in zip(obj_maps, box_maps):
            obj_all.append(obj_map.permute(0, 2, 3, 1).reshape(b, -1))
            box_all.append(box_map.permute(0, 2, 3, 1).reshape(b, -1, 4))

        return torch.cat(obj_all, dim=1), torch.cat(box_all, dim=1)

    def _generate_proposals(self, scores: torch.Tensor, deltas: torch.Tensor, anchors: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        scores = scores.sigmoid()
        num_topk = min(self.pre_nms_topk, scores.numel())
        topk_scores, topk_idx = torch.topk(scores, k=num_topk, largest=True)

        proposals = apply_deltas_to_anchors(deltas[topk_idx], anchors[topk_idx])
        proposals = clip_boxes_to_image(proposals, image_size)

        ws = proposals[:, 2] - proposals[:, 0]
        hs = proposals[:, 3] - proposals[:, 1]
        valid = (ws >= 2.0) & (hs >= 2.0)
        proposals = proposals[valid]
        topk_scores = topk_scores[valid]

        keep = nms(proposals, topk_scores, iou_threshold=0.7)
        return proposals[keep[: self.post_nms_topk]]

    def _assign_level(self, proposals: torch.Tensor) -> torch.Tensor:
        ws = proposals[:, 2] - proposals[:, 0]
        hs = proposals[:, 3] - proposals[:, 1]
        areas = (ws * hs).clamp(min=1.0)

        levels = torch.floor(4 + torch.log2(torch.sqrt(areas) / 224.0 + 1e-6))
        levels = levels.clamp(min=3, max=5).to(torch.int64)
        return levels

    def _multi_level_roi_pool(self, fpn_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], proposals: torch.Tensor, batch_idx: int) -> torch.Tensor:
        if proposals.numel() == 0:
            return torch.zeros((0, 256, 7, 7), device=fpn_feats[0].device, dtype=fpn_feats[0].dtype)

        levels = self._assign_level(proposals)
        out = torch.zeros((proposals.shape[0], 256, 7, 7), device=proposals.device, dtype=fpn_feats[0].dtype)

        for level_id, feat, stride in zip((3, 4, 5), fpn_feats, self.fpn_strides):
            idx = torch.where(levels == level_id)[0]
            if idx.numel() == 0:
                continue
            pooled = roi_pool_fallback(feat[batch_idx : batch_idx + 1], proposals[idx], output_size=7, spatial_scale=1.0 / stride)
            out[idx] = pooled

        return out

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        b, _, h, w = images.shape
        image_size = (h, w)

        c3, c4, c5 = self.backbone(images)
        p3, p4, p5 = self.fpn(c3, c4, c5)
        features = (p3, p4, p5)

        obj_maps, box_maps = self.rpn(features)
        obj_flat, box_flat = self._flatten_rpn_outputs(obj_maps, box_maps)

        anchors_by_level = self.anchor_generator(features, image_size)
        anchors = torch.cat(anchors_by_level, dim=0)

        proposals_per_image: List[torch.Tensor] = []
        class_logits_per_image: List[torch.Tensor] = []
        box_deltas_per_image: List[torch.Tensor] = []

        for i in range(b):
            proposals = self._generate_proposals(obj_flat[i], box_flat[i], anchors, image_size)
            proposals_per_image.append(proposals)

            roi_feats = self._multi_level_roi_pool(features, proposals, i)
            cls_logits, box_deltas = self.roi_head(roi_feats)
            class_logits_per_image.append(cls_logits)
            box_deltas_per_image.append(box_deltas)

        return {
            "rpn_objectness": obj_flat,
            "rpn_bbox_deltas": box_flat,
            "proposals": proposals_per_image,
            "class_logits": class_logits_per_image,
            "box_deltas": box_deltas_per_image,
        }


if __name__ == "__main__":
    model = FPNRCNN(num_classes=7)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)

    print("rpn_objectness:", tuple(out["rpn_objectness"].shape))
    print("rpn_bbox_deltas:", tuple(out["rpn_bbox_deltas"].shape))
    for i, (props, logits) in enumerate(zip(out["proposals"], out["class_logits"]), start=1):
        print(f"image {i}: proposals={tuple(props.shape)}, class_logits={tuple(logits.shape)}")
