#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def read_csv_matrix(csv_path: Path) -> np.ndarray:
    try:
        data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    except Exception:
        data = np.genfromtxt(csv_path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def fix_time_length(data: np.ndarray, target_rows: int) -> np.ndarray:
    if data.shape[0] >= target_rows:
        return data[:target_rows, :]
    pad_rows = target_rows - data.shape[0]
    pad = np.zeros((pad_rows, data.shape[1]), dtype=data.dtype)
    return np.vstack([data, pad])


def robust_amplitude(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float32, copy=False)
    background = np.median(data, axis=0, keepdims=True)
    amp = np.abs(data - background)

    p_low = float(np.percentile(amp, 30.0))
    p_high = float(np.percentile(amp, 99.3))
    if p_high <= p_low:
        return np.zeros_like(amp, dtype=np.float32)

    amp = np.clip((amp - p_low) / (p_high - p_low), 0.0, 1.0)
    amp = np.clip((amp - 0.04) / 0.96, 0.0, 1.0)
    return np.power(amp, 0.60)


def apply_reference_style_colormap(intensity: np.ndarray) -> np.ndarray:
    anchors = np.array(
        [
            [0.00, 135, 135, 135],  # grey background
            [0.18, 135, 135, 135],  # keep weak noise grey
            [0.34,   0, 210,  55],  # green
            [0.52,  35, 240, 225],  # cyan
            [0.70, 255, 240,  25],  # yellow
            [1.00, 255,  25,  20],  # red peaks
        ],
        dtype=np.float32,
    )
    x = anchors[:, 0]
    r = np.interp(intensity, x, anchors[:, 1])
    g = np.interp(intensity, x, anchors[:, 2])
    b = np.interp(intensity, x, anchors[:, 3])
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def to_output_path(csv_path: Path, dataset_root: Path, output_root: Path) -> Path:
    rel = csv_path.relative_to(dataset_root)
    parts = list(rel.parts)
    if "csv" in parts:
        idx = parts.index("csv")
        parts = parts[:idx] + [parts[-1]]
    out_rel = Path(*parts).with_suffix(".png")
    return output_root / out_rel


def convert_one(csv_path: Path, dataset_root: Path, output_root: Path, target_rows: int, image_size: int) -> Path:
    data = read_csv_matrix(csv_path)
    data = fix_time_length(data, target_rows=target_rows)
    intensity = robust_amplitude(data)
    img_arr = apply_reference_style_colormap(intensity)

    # CSV matrix: row=time (Y), col=position (X)
    img = Image.fromarray(img_arr).resize((image_size, image_size), resample=Image.BILINEAR)
    out_path = to_output_path(csv_path, dataset_root, output_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def iter_csv_files(dataset_root: Path):
    for csv_path in dataset_root.rglob("*.csv"):
        parts = set(csv_path.parts)
        if "数据集" in parts:
            continue
        yield csv_path


def main():
    parser = argparse.ArgumentParser(description="Convert DAS CSV files to 640x640 bare images.")
    parser.add_argument("--input-root", type=Path, default=Path("."), help="DAS_object_detection root path.")
    parser.add_argument("--output-root", type=Path, default=None, help="Output image root path.")
    parser.add_argument("--sample-rate", type=int, default=4000, help="Sampling rate (Hz).")
    parser.add_argument("--seconds", type=int, default=15, help="Time length in seconds.")
    parser.add_argument("--size", type=int, default=640, help="Output image width/height.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images.")
    parser.add_argument("--limit", type=int, default=0, help="Only convert first N CSV files (0 means all).")
    args = parser.parse_args()

    dataset_root = args.input_root.resolve()
    output_root = (args.output_root or (dataset_root / "images_640x640")).resolve()
    target_rows = args.sample_rate * args.seconds

    csv_files = list(iter_csv_files(dataset_root))
    if args.limit and args.limit > 0:
        csv_files = csv_files[: args.limit]
    total = len(csv_files)
    if total == 0:
        print("No CSV files found.")
        return

    done = 0
    skipped = 0
    failed = 0
    for i, csv_path in enumerate(csv_files, start=1):
        out_path = to_output_path(csv_path, dataset_root, output_root)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            convert_one(csv_path, dataset_root, output_root, target_rows=target_rows, image_size=args.size)
            done += 1
            if i % 50 == 0 or i == total:
                print(f"[{i}/{total}] converted={done}, skipped={skipped}, failed={failed}")
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {csv_path}: {exc}")

    print(f"Finished. total={total}, converted={done}, skipped={skipped}, failed={failed}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
