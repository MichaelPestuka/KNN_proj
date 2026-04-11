#!/usr/bin/env python3
"""
Per-frame image quality: PSNR (skimage) and LPIPS (AlexNet) between a collected SMB-style
run and a generated GIF or MP4.

**Command line**

From the ``eval/`` directory (with Pipenv, as elsewhere in this repo)::

    pipenv run python eval.py <ground_truth_dir> <generated.gif_or_mp4>

- ``ground_truth_dir``: folder containing ``run.json`` and the JPEG paths listed in it
  (training / collected-data layout).
- ``generated``: path to a GIF or MP4. Frames are paired by index with ground truth
  (frame ``i`` vs frame ``i``). If lengths differ, only the overlapping prefix is used
  and a note is printed to stderr.

Stdout is a table of ``frame``, ``psnr``, ``lpips`` plus a final mean row.

**Programmatic use**

Import and call ``eval(gt_dir, generated_path)``. It returns
``(per_frame_dataframe, n_gt_frames, n_gen_frames)``; the compared length is ``len(df)``.
Helpers ``load_gt_frames`` / ``load_generated`` load RGB uint8 arrays at 256×240.

All comparisons use that resolution (width × height), matching NES / gym-smb screens.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

# Target size: width × height (NES / gym-smb RGB screen).
TARGET_W, TARGET_H = 256, 240


def load_gt_frames(gt_dir: Path) -> list[np.ndarray]:
    """Load JPEGs listed in run.json in order, as RGB uint8 (H, W, 3)."""
    run_path = gt_dir / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"Missing run.json: {run_path}")

    with open(run_path, encoding="utf-8") as f:
        data = json.load(f)

    frames_meta = data.get("frames")
    if not frames_meta:
        raise ValueError("run.json has no 'frames' list")

    out: list[np.ndarray] = []
    for entry in frames_meta:
        fname = entry.get("filename")
        if not fname:
            raise ValueError("Frame entry missing 'filename'")
        path = gt_dir / fname
        if not path.is_file():
            raise FileNotFoundError(f"Missing frame file: {path}")

        im = Image.open(path).convert("RGB")
        if im.size != (TARGET_W, TARGET_H):
            im = im.resize((TARGET_W, TARGET_H), Image.Resampling.BICUBIC)
        out.append(np.asarray(im, dtype=np.uint8))
    return out


def load_generated(path: Path) -> list[np.ndarray]:
    """Read all frames from a GIF or MP4; resize each to TARGET_W×TARGET_H."""
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")

    try:
        raw_frames = imageio.mimread(path)
    except Exception as e:
        raise OSError(f"Could not read video/GIF ({path}): {e}") from e

    if not raw_frames:
        raise ValueError(f"No frames in: {path}")

    out: list[np.ndarray] = []
    for frame in raw_frames:
        if frame.ndim == 2:
            pil = Image.fromarray(frame).convert("RGB")
        else:
            pil = Image.fromarray(frame).convert("RGB")
        if pil.size != (TARGET_W, TARGET_H):
            pil = pil.resize((TARGET_W, TARGET_H), Image.Resampling.BICUBIC)
        out.append(np.asarray(pil, dtype=np.uint8))
    return out


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lpips_distance(
    loss_fn: torch.nn.Module,
    a: np.ndarray,
    b: np.ndarray,
    device: torch.device,
) -> float:
    """LPIPS between two RGB uint8 arrays (H, W, 3), normalized to [-1, 1]."""
    ta = torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    tb = torch.from_numpy(b).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    ta, tb = ta.to(device), tb.to(device)
    with torch.inference_mode():
        d = loss_fn(ta, tb)
    return float(d.item())


def _eval_frames(gt_frames: list[np.ndarray], gen_frames: list[np.ndarray]) -> pd.DataFrame:
    """Compute per-frame PSNR and LPIPS from already-loaded RGB uint8 frames."""
    n = min(len(gt_frames), len(gen_frames))
    if n == 0:
        return pd.DataFrame(columns=["frame", "psnr", "lpips"])

    device = pick_device()
    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    rows: list[dict] = []
    for i in range(n):
        psnr = peak_signal_noise_ratio(gt_frames[i], gen_frames[i], data_range=255)
        lip = lpips_distance(loss_fn, gt_frames[i], gen_frames[i], device)
        rows.append({"frame": i, "psnr": psnr, "lpips": lip})

    return pd.DataFrame(rows)


def eval(gt_dir: Path, generated_path: Path) -> tuple[pd.DataFrame, int, int]:
    """Load ground truth and generated media, then compute per-frame PSNR and LPIPS.

    Returns (per_frame_df, n_gt_frames, n_gen_frames). Compared count is len(df).
    """
    gt_dir = Path(gt_dir).resolve()
    generated_path = Path(generated_path).resolve()

    gt_frames = load_gt_frames(gt_dir)
    gen_frames = load_generated(generated_path)
    df = _eval_frames(gt_frames, gen_frames)
    return df, len(gt_frames), len(gen_frames)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PSNR + LPIPS vs ground-truth run folder and a GIF or MP4."
    )
    parser.add_argument(
        "ground_truth_dir",
        type=Path,
        help="Folder with run.json and JPEG frames (training data format)",
    )
    parser.add_argument(
        "generated",
        type=Path,
        help="Generated GIF or MP4 (frames aligned by index with ground truth)",
    )
    args = parser.parse_args()

    gt_dir = args.ground_truth_dir.resolve()
    gen_path = args.generated.resolve()

    try:
        df, n_gt, n_gen = eval(gt_dir, gen_path)
    except (FileNotFoundError, ValueError, OSError) as e:
        sys.exit(str(e))

    n = len(df)
    if n == 0:
        sys.exit("No frame pairs to compare.")
    if n_gt != n_gen:
        print(
            f"Note: lengths differ (gt={n_gt}, gen={n_gen}); "
            f"using first {n} pairs.",
            file=sys.stderr,
        )

    print()
    print(df.to_string(index=False))
    print("-" * 32)
    print(f"mean    {df['psnr'].mean():10.4f}  {df['lpips'].mean():10.6f}")


if __name__ == "__main__":
    main()
