#!/usr/bin/env python3
"""Convert YOLO prediction txt files to AnyLabeling/LabelMe-style JSON for manual correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
from PIL import Image


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_image(images_root: Path, stem_rel: Path) -> Path | None:
    for ext in IMG_EXTS:
        p = images_root / f"{stem_rel}{ext}"
        if p.exists():
            return p
    return None


def to_shape(label: str, score: float, x1: float, y1: float, x2: float, y2: float) -> Dict:
    return {
        "label": label,
        "score": score,
        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "group_id": None,
        "description": "",
        "difficult": False,
        "shape_type": "rotation",
        "flags": {},
        "attributes": {},
        "kie_linking": [],
        "direction": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-labels", required=True, help="YOLO prediction labels dir (contains labels/) ")
    parser.add_argument("--images", default="../DAS_object_detection/images_pro")
    parser.add_argument("--classes", default="../DAS_object_detection/labels_yolo/classes.txt")
    parser.add_argument("--out", default="pred_json")
    parser.add_argument("--min-conf", type=float, default=0.25)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    pred_root = Path(args.pred_labels).resolve()
    images_root = (here / args.images).resolve()
    classes_file = (here / args.classes).resolve()
    out_root = (here / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    names = [x.strip() for x in classes_file.read_text(encoding="utf-8").splitlines() if x.strip()]

    txt_files = sorted(pred_root.rglob("*.txt"))
    converted = 0
    for txt in txt_files:
        rel_stem = txt.relative_to(pred_root).with_suffix("")
        img_path = find_image(images_root, rel_stem)
        if img_path is None:
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        shapes: List[Dict] = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else 1.0
            if conf < args.min_conf or cid < 0 or cid >= len(names):
                continue
            x1 = (xc - bw / 2) * w
            x2 = (xc + bw / 2) * w
            y1 = (yc - bh / 2) * h
            y2 = (yc + bh / 2) * h
            shapes.append(to_shape(names[cid], conf, x1, y1, x2, y2))

        rel_parent = rel_stem.parent
        target_dir = out_root / rel_parent
        target_dir.mkdir(parents=True, exist_ok=True)
        out_json = target_dir / f"{rel_stem.name}.json"
        payload = {
            "version": "4.0.0-beta.5",
            "flags": {},
            "checked": False,
            "shapes": shapes,
            "imagePath": img_path.name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
        }
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        converted += 1

    print(f"Converted json files: {converted}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
