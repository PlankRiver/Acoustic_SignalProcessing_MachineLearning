#!/usr/bin/env python3
"""Train YOLO model (Ultralytics) on DAS dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/das_yolo/dataset.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/das")
    parser.add_argument("--name", default="train")
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    data = (here / args.data).resolve()

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str((here / args.project).resolve()),
        name=args.name,
        patience=args.patience,
        pretrained=True,
    )


if __name__ == "__main__":
    main()
