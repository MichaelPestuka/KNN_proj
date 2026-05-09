#!/usr/bin/env python3
"""
Compare rollout quality across model checkpoints.

Point at a parent folder that contains one subfolder per checkpoint.
Each subfolder should hold rollout_<id>_gen*.{gif,mp4} files (same layout
as batch_eval.py expects).  The same ground-truth root is used for every
checkpoint.

Output: a single figure where each checkpoint is one coloured band
(mean ± 1 std, with min/max envelope) for PSNR and LPIPS vs frame index,
so the spread of values within each checkpoint is visible at a glance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from eval import eval as run_eval
from batch_eval import (
    DEFAULT_GT_ROOT,
    find_ground_truth_dir,
    iter_rollout_gen_media,
    parse_run_id,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------


def eval_checkpoint(
    checkpoint_dir: Path,
    gt_root: Path,
) -> tuple[list[pd.DataFrame], list[dict]]:
    """Run eval on every rollout inside *checkpoint_dir*.

    Returns
    -------
    frames_list
        One DataFrame per rollout (columns: frame, psnr, lpips).
    summaries
        One dict per rollout with scalar statistics.
    """
    media_paths = iter_rollout_gen_media(checkpoint_dir)
    frames_list: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for media_path in media_paths:
        run_id = parse_run_id(media_path)
        assert run_id is not None

        try:
            gt_dir = find_ground_truth_dir(gt_root, run_id)
        except FileNotFoundError as e:
            print(f"  Skip {media_path.name}: {e}", file=sys.stderr)
            continue

        try:
            df, n_gt, n_gen = run_eval(gt_dir, media_path)
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"  Skip {media_path.name}: {e}", file=sys.stderr)
            continue

        if len(df) == 0:
            print(f"  Skip {media_path.name}: no overlapping frames", file=sys.stderr)
            continue

        frames_list.append(df)
        summaries.append(
            {
                "file": media_path.name,
                "run_id": run_id,
                "n_compared": len(df),
                "n_gt": n_gt,
                "n_gen": n_gen,
                "mean_psnr": float(df["psnr"].mean()),
                "mean_lpips": float(df["lpips"].mean()),
            }
        )
        print(
            f"  {media_path.name}: mean_psnr={summaries[-1]['mean_psnr']:.4f}"
            f"  mean_lpips={summaries[-1]['mean_lpips']:.6f}"
            f"  (pairs={len(df)}, gt={n_gt}, gen={n_gen})"
        )

    return frames_list, summaries


# ---------------------------------------------------------------------------
# Band statistics helpers
# ---------------------------------------------------------------------------


def compute_band(frames_list: list[pd.DataFrame], metric: str) -> pd.DataFrame:
    """Align all rollout series on frame index and compute per-frame statistics.

    Returns a DataFrame with columns: frame, mean, std, lo (min), hi (max).
    """
    # Build a wide matrix: rows = frame indices, columns = rollouts.
    # Use the union of all frame indices so short rollouts contribute NaN.
    all_frames = sorted({int(f) for df in frames_list for f in df["frame"]})
    wide = pd.DataFrame({"frame": all_frames}).set_index("frame")

    for i, df in enumerate(frames_list):
        s = df.set_index("frame")[metric]
        wide[f"r{i}"] = s

    data = wide.to_numpy(dtype=float)  # shape (n_frames, n_rollouts)

    result = pd.DataFrame(
        {
            "frame": all_frames,
            "mean": np.nanmean(data, axis=1),
            "std": np.nanstd(data, axis=1),
            "lo": np.nanmin(data, axis=1),
            "hi": np.nanmax(data, axis=1),
        }
    )
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def save_checkpoint_compare_plot(
    checkpoint_data: list[tuple[str, list[pd.DataFrame]]],
    out_path: Path,
) -> Path:
    """Draw one band per checkpoint for PSNR and LPIPS.

    Each band shows:
      - solid line  : per-frame mean across rollouts
      - dark shading: mean ± 1 std
      - light shading: min/max envelope
    """
    n = len(checkpoint_data)
    cmap = plt.colormaps["tab10"] if n <= 10 else plt.colormaps["tab20"]
    n_colors = 10 if n <= 10 else 20

    fig, (ax_psnr, ax_lpips) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for i, (label, frames_list) in enumerate(checkpoint_data):
        color = cmap((i % n_colors) / max(n_colors - 1, 1))

        for metric, ax in [("psnr", ax_psnr), ("lpips", ax_lpips)]:
            band = compute_band(frames_list, metric)
            frames = band["frame"].to_numpy()

            # Min/max envelope (lightest)
            ax.fill_between(
                frames,
                band["lo"],
                band["hi"],
                alpha=0.12,
                color=color,
                linewidth=0,
            )
            # Mean ± 1 std (medium shading)
            ax.fill_between(
                frames,
                band["mean"] - band["std"],
                band["mean"] + band["std"],
                alpha=0.28,
                color=color,
                linewidth=0,
            )
            # Mean line
            ax.plot(
                frames,
                band["mean"],
                label=label,
                color=color,
                linewidth=1.8,
                alpha=0.95,
            )

    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.set_title("PSNR vs frame — band = mean ± std, envelope = min/max (higher is better)")
    ax_psnr.grid(True, alpha=0.3)

    ax_lpips.set_ylabel("LPIPS")
    ax_lpips.set_xlabel("Frame")
    ax_lpips.set_title("LPIPS vs frame — band = mean ± std, envelope = min/max (lower is better)")
    ax_lpips.grid(True, alpha=0.3)

    handles, leg_labels = ax_psnr.get_legend_handles_labels()
    ncol = min(4, max(1, n))
    nrows = (n + ncol - 1) // ncol
    bottom_margin = min(0.06 + 0.030 * nrows, 0.40)
    fig.legend(
        handles,
        leg_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=ncol,
        fontsize=9,
        title="Checkpoint",
        title_fontsize=9,
    )
    fig.tight_layout(rect=(0, bottom_margin, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate rollouts from multiple checkpoints and compare them in a "
            "single band-plot figure (PSNR and LPIPS vs frame)."
        )
    )
    parser.add_argument(
        "checkpoints_dir",
        type=Path,
        help=(
            "Parent directory whose immediate sub-folders each represent one "
            "checkpoint and contain rollout_<id>_gen*.{gif,mp4} files."
        ),
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=DEFAULT_GT_ROOT,
        help=f"Root folder of ground-truth collected runs (default: {DEFAULT_GT_ROOT})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output PNG path "
            "(default: <checkpoints_dir>/checkpoint_compare_eval.png)"
        ),
    )
    parser.add_argument(
        "--sort",
        choices=["name", "none"],
        default="name",
        help="How to order checkpoint series in the legend (default: name).",
    )
    args = parser.parse_args()

    checkpoints_dir = args.checkpoints_dir.resolve()
    gt_root = args.gt_root.resolve()

    if not checkpoints_dir.is_dir():
        sys.exit(f"Not a directory: {checkpoints_dir}")

    # Collect immediate sub-directories (one per checkpoint).
    subdirs = sorted(
        [d for d in checkpoints_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name if args.sort == "name" else d.stat().st_mtime,
    )
    if not subdirs:
        sys.exit(f"No sub-directories found in {checkpoints_dir}")

    checkpoint_data: list[tuple[str, list[pd.DataFrame]]] = []
    all_summaries: list[dict] = []

    for subdir in subdirs:
        label = subdir.name
        print(f"\n=== Checkpoint: {label} ===")

        frames_list, summaries = eval_checkpoint(subdir, gt_root)

        if not frames_list:
            print(f"  No successful evaluations — skipping checkpoint {label}.")
            continue

        checkpoint_data.append((label, frames_list))

        for s in summaries:
            s["checkpoint"] = label
        all_summaries.extend(summaries)

        ckpt_psnr = np.mean([s["mean_psnr"] for s in summaries])
        ckpt_lpips = np.mean([s["mean_lpips"] for s in summaries])
        print(
            f"  → {len(summaries)} rollouts | "
            f"mean PSNR={ckpt_psnr:.4f}  mean LPIPS={ckpt_lpips:.6f}"
        )

    if not checkpoint_data:
        sys.exit("No successful evaluations for any checkpoint.")

    # Summary table
    print("\n" + "=" * 72)
    summary_df = pd.DataFrame(all_summaries)
    print(summary_df[["checkpoint", "file", "n_compared", "mean_psnr", "mean_lpips"]].to_string(index=False))
    print("=" * 72)

    # Checkpoint-level aggregation
    agg = (
        summary_df.groupby("checkpoint", sort=False)
        .agg(
            n_rollouts=("file", "count"),
            mean_psnr=("mean_psnr", "mean"),
            std_psnr=("mean_psnr", "std"),
            mean_lpips=("mean_lpips", "mean"),
            std_lpips=("mean_lpips", "std"),
        )
        .reset_index()
    )
    print("\nPer-checkpoint summary:")
    print(agg.to_string(index=False))

    out_path = (
        args.out.resolve()
        if args.out
        else checkpoints_dir / "checkpoint_compare_eval.png"
    )
    saved = save_checkpoint_compare_plot(checkpoint_data, out_path)
    print(f"\nSaved comparison plot: {saved}")


if __name__ == "__main__":
    main()
