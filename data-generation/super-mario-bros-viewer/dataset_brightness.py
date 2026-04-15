#!/usr/bin/env python3
"""Histogram of mean RGB brightness over all JPEG frames in a collected_data tree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def mean_brightness_rgb(path: Path) -> float:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float64)
    return float(arr.mean())


def collect_brightnesses(
    data_dir: Path, frame_stride: int
) -> tuple[list[float], int, int]:
    """Return (brightness values, runs_used, frames_skipped_missing_file)."""
    values: list[float] = []
    runs_used = 0
    missing = 0

    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue
        run_json = entry / "run.json"
        if not run_json.is_file():
            continue

        with open(run_json, encoding="utf-8") as f:
            run = json.load(f)

        frames = run.get("frames") or []
        if not frames:
            continue

        run_had_sample = False
        for i, frame in enumerate(frames):
            if frame_stride > 1 and (i % frame_stride) != 0:
                continue
            fname = frame.get("filename")
            if not fname:
                continue
            img_path = entry / fname
            if not img_path.is_file():
                missing += 1
                continue
            values.append(mean_brightness_rgb(img_path))
            run_had_sample = True

        if run_had_sample:
            runs_used += 1

    return values, runs_used, missing


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "dataset",
        type=Path,
        help="Root folder of the dataset (contains one subdirectory per run with run.json + JPEGs)",
    )
    p.add_argument(
        "--frame-stride",
        type=int,
        default=10,
        metavar="N",
        help="Use every Nth frame from each run's frame list (1 = all frames). Default: 10",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=64,
        help="Histogram bin count (default: 64)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write PNG to this path instead of opening an interactive window",
    )
    args = p.parse_args()

    if args.frame_stride < 1:
        print("Error: --frame-stride must be >= 1", file=sys.stderr)
        sys.exit(1)

    data_dir = args.dataset.expanduser().resolve()
    if not data_dir.is_dir():
        print(f"Error: dataset directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    import matplotlib

    matplotlib.use("TkAgg" if args.output is None else "Agg")
    import matplotlib.pyplot as plt

    values, runs_used, missing = collect_brightnesses(data_dir, args.frame_stride)
    if not values:
        print("No frames sampled.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=args.bins, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Mean RGB pixel value (0–255)")
    ax.set_ylabel("Frame count")
    ax.set_title(
        f"Frame brightness — {len(values)} frames, stride {args.frame_stride}, "
        f"{runs_used} runs"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Wrote {args.output}")
    else:
        plt.show()

    if missing:
        print(f"Warning: {missing} listed frames had missing files.", file=sys.stderr)


if __name__ == "__main__":
    main()
