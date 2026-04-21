#!/usr/bin/env python3
"""Polar histograms of (x_pos, y_pos) movement direction per discrete action over a dataset.

Angles use atan2(dy, dx) with raw RAM deltas: in SMB, y_pos increases upward (see
plot_visited_points.py: default --invert-y maps that to screen Y). This matches the
map overlay convention; do not negate dy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SIMPLE_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
    ["A"],
    ["left"],
]

COMPLEX_MOVEMENT = SIMPLE_MOVEMENT + [
    ["left", "A"],
    ["left", "B"],
    ["left", "A", "B"],
    ["down"],
    ["up"],
]


def _max_action_index_in_dataset(data_dir: Path) -> int:
    m = -1
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue
        run_json = entry / "run.json"
        if not run_json.is_file():
            continue
        with open(run_json, encoding="utf-8") as f:
            run = json.load(f)
        for frame in run.get("frames") or []:
            idx = frame.get("action_index")
            if isinstance(idx, int):
                m = max(m, idx)
    return m


def _get_xy(frame: dict[str, Any]) -> tuple[float, float] | None:
    info = frame.get("info")
    if not isinstance(info, dict):
        return None
    x = info.get("x_pos")
    y = info.get("y_pos")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    return float(x), float(y)


def collect_directions_by_action(
    data_dir: Path,
    max_jump: float,
) -> tuple[
    dict[int, list[float]],
    dict[int, list[float]],
    int,
    int,
    int,
]:
    """Return (angles_by_action, magnitudes_by_action, runs_used, pairs_used, skipped_jump)."""
    angles: dict[int, list[float]] = {}
    magnitudes: dict[int, list[float]] = {}
    runs_used = 0
    pairs_used = 0
    skipped_jump = 0

    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue
        run_json = entry / "run.json"
        if not run_json.is_file():
            continue

        with open(run_json, encoding="utf-8") as f:
            run = json.load(f)

        frames = run.get("frames") or []
        if len(frames) < 2:
            continue

        run_had_pair = False
        for i in range(len(frames) - 1):
            a = frames[i]
            b = frames[i + 1]
            xy0 = _get_xy(a)
            xy1 = _get_xy(b)
            if xy0 is None or xy1 is None:
                continue

            dx = xy1[0] - xy0[0]
            dy = xy1[1] - xy0[1]

            if abs(dx) > max_jump or abs(dy) > max_jump:
                skipped_jump += 1
                fi = a.get("frame", i)
                fj = b.get("frame", i + 1)
                print(
                    f"max-jump skip: run={entry.name} "
                    f"pair index {i}->{i + 1} (frame {fi}->{fj}) "
                    f"dx={dx:.2f} dy={dy:.2f}",
                    file=sys.stderr,
                )
                continue

            theta = math.atan2(dy, dx)
            step_len = math.hypot(dx, dy)
            act = a.get("action_index")
            if not isinstance(act, int):
                continue

            angles.setdefault(act, []).append(theta)
            magnitudes.setdefault(act, []).append(step_len)
            pairs_used += 1
            run_had_pair = True

        if run_had_pair:
            runs_used += 1

    return angles, magnitudes, runs_used, pairs_used, skipped_jump


def _subplot_grid(n: int) -> tuple[int, int]:
    if n <= 7:
        return 3, 3
    return 3, 4


def plot_polar_histograms(
    movement: list[list[str]],
    angles_by_action: dict[int, list[float]],
    magnitudes_by_action: dict[int, list[float]],
    bins: int,
    *,
    map_label: str,
    data_dir: Path,
    runs_used: int,
    pairs_used: int,
    skipped_jump: int,
    output: Path | None,
) -> None:
    import matplotlib

    matplotlib.use("TkAgg" if output is None else "Agg")
    import matplotlib.pyplot as plt

    n_actions = len(movement)
    nrows, ncols = _subplot_grid(n_actions)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        subplot_kw={"projection": "polar"},
        figsize=(4 * ncols, 4 * nrows),
    )
    axes_flat = np.atleast_1d(axes).ravel()

    bin_edges = np.linspace(-math.pi, math.pi, bins + 1)
    width = 2 * math.pi / bins

    for idx in range(n_actions):
        ax = axes_flat[idx]
        thetas = angles_by_action.get(idx, [])
        mags = magnitudes_by_action.get(idx, [])
        n_samples = len(thetas)
        mean_len = float(np.mean(mags)) if mags else float("nan")
        if n_samples:
            counts, _ = np.histogram(thetas, bins=bin_edges)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            ax.bar(centers, counts, width=width, bottom=0.0, align="center")
        label = "+".join(movement[idx]) if movement[idx] else "NOOP"
        len_str = f"{mean_len:.2f}" if mags else "—"
        ax.set_title(
            f"{idx}: {label}  (n={n_samples}, avg |Δp|={len_str})",
            fontsize=9,
        )
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)

    for j in range(n_actions, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{map_label} — pairs={pairs_used}, runs={runs_used}, "
        f"skipped_max_jump={skipped_jump}\n{data_dir}",
        fontsize=10,
    )
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"Wrote {output}")
    else:
        plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "dataset",
        type=Path,
        help="Root folder of the dataset (one subdirectory per run with run.json)",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=36,
        help="Polar histogram bin count (default: 36)",
    )
    p.add_argument(
        "--max-jump",
        type=float,
        default=50.0,
        metavar="PX",
        help="Drop pairs where |dx| or |dy| exceeds this; log run and frame (default: 50)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write PNG to this path instead of opening an interactive window",
    )
    args = p.parse_args()

    if args.bins < 1:
        print("Error: --bins must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.max_jump <= 0:
        print("Error: --max-jump must be > 0", file=sys.stderr)
        sys.exit(1)

    data_dir = args.dataset.expanduser().resolve()
    if not data_dir.is_dir():
        print(f"Error: dataset directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    max_idx = _max_action_index_in_dataset(data_dir)
    if max_idx < 0:
        print("No frames with action_index found.", file=sys.stderr)
        sys.exit(1)

    use_complex = max_idx >= 7
    movement = COMPLEX_MOVEMENT if use_complex else SIMPLE_MOVEMENT
    map_label = "COMPLEX_MOVEMENT" if use_complex else "SIMPLE_MOVEMENT"

    angles_by_action, magnitudes_by_action, runs_used, pairs_used, skipped_jump = (
        collect_directions_by_action(data_dir, args.max_jump)
    )

    if pairs_used == 0:
        print("No valid frame pairs after filtering.", file=sys.stderr)
        sys.exit(1)

    plot_polar_histograms(
        movement,
        angles_by_action,
        magnitudes_by_action,
        args.bins,
        map_label=map_label,
        data_dir=data_dir,
        runs_used=runs_used,
        pairs_used=pairs_used,
        skipped_jump=skipped_jump,
        output=args.output,
    )


if __name__ == "__main__":
    main()
