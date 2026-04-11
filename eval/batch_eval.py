#!/usr/bin/env python3
"""
Batch-evaluate generated rollout GIFs against ground-truth run folders.

Expects media named rollout_<id>_gen*.{gif,mp4}; resolves ground truth by finding
a direct child of collected_data whose folder name contains that run id and which
has run.json.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Same-directory import when run as `python eval/batch_eval.py` from repo root.
_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from eval import eval as run_eval

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_GT_ROOT = Path(
    "/Users/lukasfoukal/Documents/GitHub/KNN_proj/data-generation/super-mario-bros/collected_data"
)

# rollout_<id>_gen*.{gif,mp4} — id is the part between "rollout_" and "_gen".
ROLLOUT_GEN_RE = re.compile(r"^rollout_(.+)_gen.*\.(gif|mp4)$", re.IGNORECASE)


def parse_run_id(media_path: Path) -> str | None:
    m = ROLLOUT_GEN_RE.match(media_path.name)
    return m.group(1) if m else None


def iter_rollout_gen_media(rollouts_dir: Path) -> list[Path]:
    """Paths under rollouts_dir matching rollout_<id>_gen* and .gif / .mp4."""
    out: list[Path] = []
    for p in rollouts_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".gif", ".mp4"):
            continue
        if parse_run_id(p) is not None:
            out.append(p)
    return sorted(out, key=lambda x: x.name.lower())


def find_ground_truth_dir(collected_root: Path, run_id: str) -> Path:
    """Return a subdirectory of collected_root whose name contains run_id and has run.json."""
    collected_root = collected_root.resolve()
    if not collected_root.is_dir():
        raise FileNotFoundError(f"Not a directory: {collected_root}")

    candidates: list[Path] = []
    for d in collected_root.iterdir():
        if not d.is_dir():
            continue
        if run_id not in d.name:
            continue
        if (d / "run.json").is_file():
            candidates.append(d)

    if not candidates:
        raise FileNotFoundError(
            f"No ground-truth folder (name contains {run_id!r}, with run.json) under {collected_root}"
        )

    exact = [c for c in candidates if c.name == run_id]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        names = ", ".join(sorted(c.name for c in exact))
        raise FileNotFoundError(
            f"Multiple folders named exactly {run_id!r}: {names}"
        )

    if len(candidates) == 1:
        return candidates[0]

    names = ", ".join(sorted(c.name for c in candidates))
    raise FileNotFoundError(
        f"Multiple folders match run id {run_id!r}: {names}. "
        "Disambiguate (e.g. rename or use a longer id in the rollout filename)."
    )


def save_all_runs_overlay_plot(
    per_run: list[tuple[str, pd.DataFrame]], out_dir: Path
) -> Path:
    """One figure: PSNR and LPIPS vs frame; each rollout is a colored series (same color in both panels)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "batch_eval_all_runs_frame_metrics.png"

    n = len(per_run)
    cmap = plt.colormaps["tab20"]

    fig, (ax_psnr, ax_lpips) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i, (label, df) in enumerate(per_run):
        color = cmap((i % 20) / 19.0 if 20 > 1 else 0)
        ax_psnr.plot(
            df["frame"],
            df["psnr"],
            label=label,
            color=color,
            linewidth=1.2,
            alpha=0.9,
        )
        ax_lpips.plot(df["frame"], df["lpips"], color=color, linewidth=1.2, alpha=0.9)

    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.set_title("PSNR vs frame (higher is better)")
    ax_psnr.grid(True, alpha=0.3)

    ax_lpips.set_ylabel("LPIPS")
    ax_lpips.set_xlabel("Frame")
    ax_lpips.set_title("LPIPS vs frame (lower is better)")
    ax_lpips.grid(True, alpha=0.3)

    handles, leg_labels = ax_psnr.get_legend_handles_labels()
    ncol = min(4, max(1, n))
    nrows = (n + ncol - 1) // ncol
    bottom_margin = min(0.08 + 0.028 * nrows, 0.42)
    fig.legend(
        handles,
        leg_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=ncol,
        fontsize=7,
    )
    fig.tight_layout(rect=(0, bottom_margin, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch PSNR + LPIPS for rollout_<id>_gen*.{gif,mp4} vs collected_data runs."
    )
    parser.add_argument(
        "rollouts_dir",
        type=Path,
        help="Directory containing rollout_<id>_gen* .gif or .mp4 files",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=DEFAULT_GT_ROOT,
        help=f"Root folder of collected runs (default: {DEFAULT_GT_ROOT})",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Where to save batch_eval_all_runs_frame_metrics.png (default: rollouts_dir)",
    )
    args = parser.parse_args()

    rollouts_dir = args.rollouts_dir.resolve()
    gt_root = args.gt_root.resolve()

    if not rollouts_dir.is_dir():
        sys.exit(f"Not a directory: {rollouts_dir}")

    media_paths = iter_rollout_gen_media(rollouts_dir)
    if not media_paths:
        sys.exit(
            f"No rollout_<id>_gen* .gif or .mp4 files in {rollouts_dir}"
        )

    summaries: list[dict] = []
    per_run_frames: list[tuple[str, pd.DataFrame]] = []

    for media_path in media_paths:
        run_id = parse_run_id(media_path)
        assert run_id is not None

        try:
            gt_dir = find_ground_truth_dir(gt_root, run_id)
        except FileNotFoundError as e:
            print(f"Skip {media_path.name}: {e}", file=sys.stderr)
            continue

        try:
            df, n_gt, n_gen = run_eval(gt_dir, media_path)
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"Skip {media_path.name}: {e}", file=sys.stderr)
            continue

        n = len(df)
        if n == 0:
            print(f"Skip {media_path.name}: no overlapping frames", file=sys.stderr)
            continue

        mean_psnr = float(df["psnr"].mean())
        mean_lpips = float(df["lpips"].mean())
        summaries.append(
            {
                "file": media_path.name,
                "run_id": run_id,
                "n_compared": n,
                "n_gt": n_gt,
                "n_gen": n_gen,
                "mean_psnr": mean_psnr,
                "mean_lpips": mean_lpips,
            }
        )
        per_run_frames.append((media_path.name, df))
        print(
            f"{media_path.name}: run_id={run_id}  "
            f"mean_psnr={mean_psnr:.4f}  mean_lpips={mean_lpips:.6f}  "
            f"(pairs={n}, gt={n_gt}, gen={n_gen})"
        )

    if not summaries:
        sys.exit("No successful evaluations.")

    print()
    summary_df = pd.DataFrame(summaries)
    print(summary_df.to_string(index=False))
    print("-" * 72)
    print(
        f"overall mean  PSNR={summary_df['mean_psnr'].mean():.4f}  "
        f"LPIPS={summary_df['mean_lpips'].mean():.6f}  (n={len(summary_df)} rollouts)"
    )

    plot_dir = args.plot_dir.resolve() if args.plot_dir else rollouts_dir
    plot_path = save_all_runs_overlay_plot(per_run_frames, plot_dir)
    print()
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
