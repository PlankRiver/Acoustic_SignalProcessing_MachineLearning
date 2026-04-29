#!/usr/bin/env python3
"""Recursive inference and export YOLO txt with preserved relative paths.

This avoids Ultralytics non-recursive source limitations and preserves
folder structure for later conversion to AnyLabeling JSON.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from ultralytics import YOLO

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(root: Path) -> List[Path]:
    images: List[Path] = []
    for ext in IMG_EXTS:
        images.extend(root.rglob(f"*{ext}"))
        images.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", default="../DAS_object_detection/images_pro")
    parser.add_argument("--out", default="runs/das/predict_all_recursive/labels")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save-conf", action="store_true", default=True)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    source_root = (here / args.source).resolve()
    out_root = (here / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    images = list_images(source_root)
    if not images:
        raise FileNotFoundError(f"No images found in: {source_root}")

    model = YOLO(str(Path(args.weights).resolve()))

    print(f"Found images: {len(images)}")
    done = 0
    for img in images:
        rel = img.relative_to(source_root).with_suffix(".txt")
        target = out_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        results = model.predict(
            source=str(img),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            save=False,
        )
        r = results[0]

        lines: List[str] = []
        if r.boxes is not None and len(r.boxes) > 0:
            xywhn = r.boxes.xywhn.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for i in range(len(cls)):
                c = int(cls[i])
                x, y, w, h = xywhn[i].tolist()
                if args.save_conf:
                    lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {float(confs[i]):.6f}")
                else:
                    lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        done += 1
        if done % 100 == 0:
            print(f"Processed {done}/{len(images)}")

    print(f"Done. Wrote YOLO labels to: {out_root}")


if __name__ == "__main__":
    main()
