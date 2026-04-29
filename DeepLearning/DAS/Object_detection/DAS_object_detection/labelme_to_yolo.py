#!/usr/bin/env python3
"""Convert LabelMe/AnyLabeling JSON annotations to YOLO txt labels in batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def bbox_from_points(points: Iterable[Iterable[float]]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for p in points:
        if len(p) < 2:
            continue
        xs.append(float(p[0]))
        ys.append(float(p[1]))
    if not xs or not ys:
        raise ValueError("empty points")
    return min(xs), min(ys), max(xs), max(ys)


def convert_one(json_path: Path, out_dir: Path, class_to_id: Dict[str, int]) -> int:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    width = float(data.get("imageWidth", 0))
    height = float(data.get("imageHeight", 0))
    if width <= 0 or height <= 0:
        raise ValueError("missing imageWidth/imageHeight")

    lines: List[str] = []
    for shape in data.get("shapes", []):
        label = shape.get("label")
        points = shape.get("points")
        if not label or not points:
            continue
        class_id = class_to_id[label]
        x1, y1, x2, y2 = bbox_from_points(points)
        xc = clamp01(((x1 + x2) / 2.0) / width)
        yc = clamp01(((y1 + y2) / 2.0) / height)
        bw = clamp01((x2 - x1) / width)
        bh = clamp01((y2 - y1) / height)
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    rel = json_path.with_suffix(".txt").name
    target = out_dir / rel
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convert LabelMe/AnyLabeling json to YOLO txt")
    parser.add_argument("--input", default="images_pro", help="Input root containing json annotations")
    parser.add_argument("--output", default="labels_yolo", help="Output root for YOLO txt labels")
    parser.add_argument("--ext", default=".json", help="Annotation extension")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_root = (script_dir / args.input).resolve()
    output_root = (script_dir / args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_root.rglob(f"*{args.ext}"))
    if not json_files:
        print(f"No annotation files found in: {input_root}")
        return

    labels = set()
    for p in json_files:
        data = json.loads(p.read_text(encoding="utf-8"))
        for s in data.get("shapes", []):
            lb = s.get("label")
            if lb:
                labels.add(lb)
    class_names = sorted(labels)
    class_to_id = {name: i for i, name in enumerate(class_names)}

    converted = 0
    boxes = 0
    failed = 0
    for jp in json_files:
        try:
            rel_parent = jp.relative_to(input_root).parent
            out_dir = output_root / rel_parent
            out_dir.mkdir(parents=True, exist_ok=True)
            boxes += convert_one(jp, out_dir, class_to_id)
            converted += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {jp}: {exc}")

    (output_root / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    print(f"Done. json={len(json_files)}, converted={converted}, failed={failed}, boxes={boxes}")
    print(f"Output: {output_root}")
    print(f"Classes: {output_root / 'classes.txt'}")


if __name__ == "__main__":
    main()
