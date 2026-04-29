#!/usr/bin/env python3
"""Two-stage YOLOv8 transfer learning for DAS.

Stage 1: freeze most layers for stable warmup on small dataset.
Stage 2: unfreeze all layers for full fine-tuning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def run_stage(
    model: YOLO,
    *,
    data: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    project: Path,
    name: str,
    patience: int,
    freeze: int,
    lr0: float,
    lrf: float,
    amp: bool,
    cos_lr: bool,
    mosaic: float,
) -> None:
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(project),
        name=name,
        patience=patience,
        pretrained=True,
        freeze=freeze,
        lr0=lr0,
        lrf=lrf,
        amp=amp,
        cos_lr=cos_lr,
        mosaic=mosaic,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/std/dataset.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/das")
    parser.add_argument("--name", default="transfer")

    parser.add_argument("--epochs-stage1", type=int, default=30)
    parser.add_argument("--epochs-stage2", type=int, default=120)
    parser.add_argument("--patience", type=int, default=30)

    # For YOLOv8n/s, ~10 is a practical freeze depth to keep head trainable.
    parser.add_argument("--freeze-layers", type=int, default=10)

    parser.add_argument("--lr0-stage1", type=float, default=0.001)
    parser.add_argument("--lr0-stage2", type=float, default=0.0005)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--amp", action="store_true", default=False, help="Enable mixed precision AMP")
    parser.add_argument("--cos-lr", action="store_true", default=False)
    parser.add_argument("--mosaic", type=float, default=0.0)

    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    data = (here / args.data).resolve()
    project = (here / args.project).resolve()

    stage1_name = f"{args.name}_stage1"
    stage2_name = f"{args.name}_stage2"

    print("=== Stage 1: frozen transfer learning ===")
    model = YOLO(args.model)
    run_stage(
        model,
        data=data,
        epochs=args.epochs_stage1,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=project,
        name=stage1_name,
        patience=args.patience,
        freeze=args.freeze_layers,
        lr0=args.lr0_stage1,
        lrf=args.lrf,
        amp=args.amp,
        cos_lr=args.cos_lr,
        mosaic=args.mosaic,
    )

    stage1_best = project / stage1_name / "weights" / "best.pt"
    if not stage1_best.exists():
        raise FileNotFoundError(f"Stage1 best weights not found: {stage1_best}")

    print("=== Stage 2: full fine-tuning ===")
    model = YOLO(str(stage1_best))
    run_stage(
        model,
        data=data,
        epochs=args.epochs_stage2,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=project,
        name=stage2_name,
        patience=args.patience,
        freeze=0,
        lr0=args.lr0_stage2,
        lrf=args.lrf,
        amp=args.amp,
        cos_lr=args.cos_lr,
        mosaic=args.mosaic,
    )

    final_best = project / stage2_name / "weights" / "best.pt"
    print(f"Done. Final best weights: {final_best}")


if __name__ == "__main__":
    main()
