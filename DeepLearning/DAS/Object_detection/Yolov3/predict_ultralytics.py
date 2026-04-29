#!/usr/bin/env python3
"""Run prediction on all images and export YOLO txt labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from ultralytics import YOLO

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def collect_images_recursively(source_dir: Path) -> List[str]:
    files: List[str] = []
    for ext in IMG_EXTS:
        files.extend(str(p) for p in source_dir.rglob(f"*{ext}"))
        files.extend(str(p) for p in source_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--source", default="../DAS_object_detection/images_pro")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/das")
    parser.add_argument("--name", default="predict_all")
    parser.add_argument("--recursive", action="store_true", default=True)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    weights = Path(args.weights).resolve()
    source = (here / args.source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    infer_source: str | List[str]
    if args.recursive and source.is_dir():
        imgs = collect_images_recursively(source)
        if not imgs:
            raise FileNotFoundError(f"No images found recursively in: {source}")
        print(f"Found {len(imgs)} images under {source}")
        infer_source = imgs
    else:
        infer_source = str(source)

    model = YOLO(str(weights))
    results = model.predict(
        source=infer_source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=False,
        save_txt=True,
        save_conf=True,
        project=str((here / args.project).resolve()),
        name=args.name,
        stream=True,
    )
    # Consume stream to execute inference while keeping memory usage bounded.
    for _ in results:
        pass


if __name__ == "__main__":
    main()
