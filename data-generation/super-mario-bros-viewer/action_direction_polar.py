#!/usr/bin/env python3
"""Cartesian (Δx, Δy) heatmap of movement per discrete action over a dataset.

Each subplot shows a 2D histogram of frame-pair displacements for one action.
Bins are 1-unit wide (matching integer RAM deltas). Color = log(1 + count),
displayed in greyscale (dark = dense).

Angles use atan2(dy, dx) with raw RAM deltas: in SMB, y_pos increases upward.
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


def _trailing_hold_lengths(frames: list[dict[str, Any]]) -> list[int]:
    """For each frame i, return how long the current action_index has been held."""
    n = len(frames)
    hold = [0] * n
    for i in range(n):
        act = frames[i].get("action_index")
        if not isinstance(act, int):
            continue
        hold[i] = (hold[i - 1] + 1) if i > 0 and frames[i - 1].get("action_index") == act else 1
    return hold


def collect_deltas_by_action(
    data_dir: Path,
    max_jump: float,
    min_duration: int,
) -> tuple[
    dict[int, list[float]],
    dict[int, list[float]],
    int,
    int,
    int,
    int,
]:
    """Return (dx_by_action, dy_by_action, runs_used, pairs_used, skipped_jump, skipped_short)."""
    dx_by_action: dict[int, list[float]] = {}
    dy_by_action: dict[int, list[float]] = {}
    runs_used = 0
    pairs_used = 0
    skipped_jump = 0
    skipped_short = 0

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

        hold_lens = _trailing_hold_lengths(frames)

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
                    f"dx={dx:.0f} dy={dy:.0f}",
                    file=sys.stderr,
                )
                continue

            act = a.get("action_index")
            if not isinstance(act, int):
                continue

            if min_duration > 1 and hold_lens[i] < min_duration:
                skipped_short += 1
                continue

            dx_by_action.setdefault(act, []).append(dx)
            dy_by_action.setdefault(act, []).append(dy)
            pairs_used += 1
            run_had_pair = True

        if run_had_pair:
            runs_used += 1

    return dx_by_action, dy_by_action, runs_used, pairs_used, skipped_jump, skipped_short


def _subplot_grid(n: int) -> tuple[int, int]:
    if n <= 7:
        return 3, 3
    return 3, 4


def plot_cartesian_heatmaps(
    movement: list[list[str]],
    dx_by_action: dict[int, list[float]],
    dy_by_action: dict[int, list[float]],
    *,
    max_jump: float,
    map_label: str,
    data_dir: Path,
    runs_used: int,
    pairs_used: int,
    skipped_jump: int,
    skipped_short: int,
    min_duration: int,
    output: Path | None,
) -> None:
    """Square 2D histogram of (Δx, Δy) per action; bins are 1-unit wide (integer deltas)."""
    import matplotlib

    matplotlib.use("TkAgg" if output is None else "Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    n_actions = len(movement)
    nrows, ncols = _subplot_grid(n_actions)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        layout="constrained",
    )
    axes_flat = np.atleast_1d(axes).ravel()

    # Extent: smallest integer L that covers max(|Δx|, |Δy|) across all data, capped by max_jump.
    all_abs = [
        abs(v)
        for lst in list(dx_by_action.values()) + list(dy_by_action.values())
        for v in lst
    ]
    L = int(math.ceil(min(float(np.percentile(all_abs, 99)) if all_abs else max_jump, max_jump)))
    L = max(L, 1)
    # Integer-centered bin edges: [-L-0.5, ..., L+0.5], width=1 per bin.
    edges = np.arange(-L - 0.5, L + 1.5, 1.0)

    zmax = 0.0
    hist_by_idx: dict[int, np.ndarray] = {}
    for idx in range(n_actions):
        dxl = dx_by_action.get(idx, [])
        dyl = dy_by_action.get(idx, [])
        if not dxl:
            continue
        h, _, _ = np.histogram2d(
            np.asarray(dxl, dtype=float),
            np.asarray(dyl, dtype=float),
            bins=[edges, edges],
        )
        h = np.log1p(h)
        hist_by_idx[idx] = h
        zmax = max(zmax, float(np.max(h)))
    zmax = max(zmax, 1e-9)
    norm = mcolors.Normalize(vmin=0.0, vmax=zmax)
    cmap = plt.get_cmap("gray_r")  # white = empty, black = dense

    x_grid, y_grid = np.meshgrid(edges, edges, indexing="ij")

    for idx in range(n_actions):
        ax = axes_flat[idx]
        n_samples = len(dx_by_action.get(idx, []))
        label = "+".join(movement[idx]) if movement[idx] else "NOOP"
        if idx in hist_by_idx:
            ax.pcolormesh(x_grid, y_grid, hist_by_idx[idx], shading="auto", cmap=cmap, norm=norm)
        ax.axhline(0.0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(0.0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Δx")
        ax.set_ylabel("Δy")
        ax.set_title(f"{idx}: {label}  (n={n_samples})", fontsize=9)

    for j in range(n_actions, len(axes_flat)):
        axes_flat[j].set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=list(axes_flat[:n_actions]), shrink=0.7, pad=0.02, label="log(1 + count)")

    extra = (
        f", skipped_short={skipped_short} (min_dur={min_duration})" if min_duration > 1 else ""
    )
    fig.suptitle(
        f"{map_label} — Δx×Δy heatmap — pairs={pairs_used}, runs={runs_used}, "
        f"L={L}, skipped_max_jump={skipped_jump}{extra}\n{data_dir}",
        fontsize=10,
    )

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
        "--max-jump",
        type=float,
        default=50.0,
        metavar="PX",
        help="Drop pairs where |dx| or |dy| exceeds this (default: 50)",
    )
    p.add_argument(
        "--min-duration",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Only count a pair (i, i+1) if the action at frame i has been held "
            ">= N consecutive frames. Default: 1 = no filter."
        ),
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write PNG to this path instead of opening an interactive window",
    )
    args = p.parse_args()

    if args.max_jump <= 0:
        print("Error: --max-jump must be > 0", file=sys.stderr)
        sys.exit(1)
    if args.min_duration < 1:
        print("Error: --min-duration must be >= 1", file=sys.stderr)
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

    (
        dx_by_action,
        dy_by_action,
        runs_used,
        pairs_used,
        skipped_jump,
        skipped_short,
    ) = collect_deltas_by_action(data_dir, args.max_jump, args.min_duration)

    if pairs_used == 0:
        print("No valid frame pairs after filtering.", file=sys.stderr)
        sys.exit(1)

    plot_cartesian_heatmaps(
        movement,
        dx_by_action,
        dy_by_action,
        max_jump=args.max_jump,
        map_label=map_label,
        data_dir=data_dir,
        runs_used=runs_used,
        pairs_used=pairs_used,
        skipped_jump=skipped_jump,
        skipped_short=skipped_short,
        min_duration=args.min_duration,
        output=args.output,
    )


if __name__ == "__main__":
    main()
