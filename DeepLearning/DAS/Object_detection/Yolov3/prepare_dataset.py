#!/usr/bin/env python3
"""Prepare YOLO dataset split files and data.yaml from existing DAS labels."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def resolve_image(images_root: Path, stem_rel: Path) -> Path | None:
    for ext in IMG_EXTS:
        p = images_root / f"{stem_rel}{ext}"
        if p.exists():
            return p
    return None


def load_samples(images_root: Path, labels_root: Path) -> List[Tuple[Path, Path]]:
    samples: List[Tuple[Path, Path]] = []
    for txt in sorted(labels_root.rglob("*.txt")):
        if txt.name == "classes.txt":
            continue
        rel_stem = txt.relative_to(labels_root).with_suffix("")
        img = resolve_image(images_root, rel_stem)
        if img is None:
            continue
        samples.append((img.resolve(), txt.resolve()))
    return samples


def write_list(path: Path, values: List[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(v) for v in values) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="../DAS_object_detection/images_pro")
    parser.add_argument("--labels", default="../DAS_object_detection/labels_yolo")
    parser.add_argument("--out", default="datasets/das_yolo")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    images_root = (here / args.images).resolve()
    labels_root = (here / args.labels).resolve()
    out_root = (here / args.out).resolve()

    classes_file = labels_root / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(f"classes.txt not found: {classes_file}")

    names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = load_samples(images_root, labels_root)
    if not samples:
        raise RuntimeError("No image/label pairs found")

    rnd = random.Random(args.seed)
    rnd.shuffle(samples)

    n_val = max(1, int(len(samples) * args.val_ratio))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    if not train_samples:
        train_samples = val_samples

    write_list(out_root / "train.txt", [img for img, _ in train_samples])
    write_list(out_root / "val.txt", [img for img, _ in val_samples])
    write_list(out_root / "all_images.txt", [img for img, _ in samples])

    # Optional label lists for debugging.
    write_list(out_root / "train_labels.txt", [lbl for _, lbl in train_samples])
    write_list(out_root / "val_labels.txt", [lbl for _, lbl in val_samples])

    data_yaml = out_root / "dataset.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {images_root}",
                f"train: {out_root / 'train.txt'}",
                f"val: {out_root / 'val.txt'}",
                f"nc: {len(names)}",
                "names:",
                *[f"  {i}: {name}" for i, name in enumerate(names)],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Prepared dataset under: {out_root}")
    print(f"Total pairs: {len(samples)}")
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")
    print(f"Data YAML: {data_yaml}")


if __name__ == "__main__":
    main()
