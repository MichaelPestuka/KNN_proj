#!/usr/bin/env python3
"""
Overlay visited `(x_pos, y_pos)` points from one or more `run.json` files onto
`SuperMarioBrosMap1-1.png`.

Coordinate mapping:
- Uses per-frame `frames[*].info.x_pos` / `frames[*].info.y_pos`.
- Applies an offset so that the *first* frame matches a target start coordinate
  (defaults to your provided `X=48, Y=46`).
- By default, inverts Y to match typical image coordinates (origin at top-left):
  `pixel_y = map_height - y_corrected`.

This is intentionally configurable via CLI in case your coordinate convention
differs slightly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFilter

import numpy as np


@dataclass(frozen=True)
class RunSpec:
    json_path: Path
    label: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot visited Super Mario Bros points onto the map image."
    )
    p.add_argument(
        "runs",
        nargs="*",
        help=(
            "One or more run paths. Each entry can be either a run directory "
            "containing run.json or a direct path to run.json."
        ),
    )
    p.add_argument(
        "--mode",
        choices=("heatmap", "dots"),
        default="heatmap",
        help="Rendering mode: `heatmap` (default) or `dots`.",
    )
    p.add_argument(
        "--map-image",
        type=Path,
        default=Path("SuperMarioBrosMap1-1.png"),
        help="Map image to draw onto (default: SuperMarioBrosMap1-1.png).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path. If omitted, defaults to "
            "`../super-mario-bros/collected_data/_all_runs_heatmap.png`."
        ),
    )
    p.add_argument(
        "--target-start-x",
        type=float,
        default=48.0,
        help="Corrected coordinate X for the first frame (default: 48).",
    )
    p.add_argument(
        "--target-start-y",
        type=float,
        default=46.0,
        help="Corrected coordinate Y for the first frame (default: 46).",
    )
    p.add_argument(
        "--scale-x",
        type=float,
        default=1.0,
        help="Scale applied to corrected X before drawing.",
    )
    p.add_argument(
        "--scale-y",
        type=float,
        default=1.0,
        help="Scale applied to corrected Y before drawing.",
    )
    p.add_argument(
        "--invert-y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Invert Y when mapping to image coordinates (default: enabled).",
    )
    p.add_argument(
        "--heatmap-downsample",
        type=int,
        default=3,
        help=(
            "Downsample factor for heat accumulation (default: 3). "
            "Larger values are faster and smoother."
        ),
    )
    p.add_argument(
        "--heatmap-percentile",
        type=float,
        default=95.0,
        help=(
            "Percentile for heat normalization (log-space) (default: 95). "
            "Helps prevent a few hot spots from saturating."
        ),
    )
    p.add_argument(
        "--heatmap-gamma",
        type=float,
        default=0.5,
        help="Gamma curve for heatmap contrast (default: 0.5).",
    )
    p.add_argument(
        "--heatmap-blur-radius",
        type=float,
        default=1.5,
        help="Gaussian blur radius in downsampled-pixel units (default: 1.5).",
    )
    p.add_argument(
        "--heatmap-max-alpha",
        type=int,
        default=200,
        help="Max overlay opacity (0-255) for the heatmap (default: 200).",
    )
    p.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Marker point radius in pixels (default: 2).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Dot opacity in [0, 1] (only used in `dots` mode; default: 0.1).",
    )
    p.add_argument(
        "--connect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally connect points with a thin line (can help visualize motion).",
    )
    p.add_argument(
        "--colors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use per-run colors (default: monochrome/gray). "
            "When disabled, all traces are drawn in gray/black tones."
        ),
    )
    p.add_argument(
        "--collected-data-dir",
        type=Path,
        default=None,
        help=(
            "If no positional `runs` are provided, discover all run.json files "
            "under this directory. Defaults to "
            "`../super-mario-bros/collected_data` relative to this script."
        ),
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=2000,
        help="Safety cap when auto-discovering runs (default: 2000).",
    )
    return p.parse_args()


def _coerce_run_specs(paths: Iterable[str]) -> list[RunSpec]:
    out: list[RunSpec] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            json_path = p / "run.json"
            label = p.name
        else:
            json_path = p
            label = p.parent.name
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing run.json: {json_path}")
        out.append(RunSpec(json_path=json_path.resolve(), label=label))
    return out


def _discover_run_specs(collected_data_dir: Path, max_runs: int | None) -> list[RunSpec]:
    collected_data_dir = collected_data_dir.resolve()
    if not collected_data_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {collected_data_dir}")

    # Discover runs by convention: every run lives in a folder containing run.json.
    json_paths = sorted(collected_data_dir.rglob("run.json"), key=lambda p: str(p))
    if max_runs is not None:
        json_paths = json_paths[:max_runs]

    out: list[RunSpec] = []
    for jp in json_paths:
        if not jp.is_file():
            continue
        out.append(RunSpec(json_path=jp, label=jp.parent.name))
    return out


def _load_frames(json_path: Path) -> tuple[float, float, list[tuple[float, float]]]:
    with json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    frames = meta.get("frames") or []
    if not frames:
        raise ValueError(f"No frames in {json_path}")

    pts: list[tuple[float, float]] = []
    for fr in frames:
        info = fr.get("info") or {}
        if "x_pos" not in info or "y_pos" not in info:
            continue
        pts.append((float(info["x_pos"]), float(info["y_pos"])))

    if not pts:
        raise ValueError(f"No (x_pos, y_pos) points in {json_path}")

    x0, y0 = pts[0]
    return x0, y0, pts


def _draw_point(draw: ImageDraw.ImageDraw, x: float, y: float, r: int, color: tuple[int, int, int, int]) -> None:
    # Draw a filled circle.
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def main() -> int:
    args = _parse_args()

    script_dir = Path(__file__).resolve().parent
    default_collected = (script_dir / ".." / "super-mario-bros" / "collected_data").resolve()
    collected_dir = args.collected_data_dir or default_collected

    output_path = args.output or (collected_dir / "_all_runs_heatmap.png")

    if args.runs:
        runs = _coerce_run_specs(args.runs)
    else:
        runs = _discover_run_specs(collected_dir, max_runs=args.max_runs)
        if not runs:
            raise FileNotFoundError(
                f"No run.json files found under {collected_dir}. "
                f"Either add positional run paths or fix `--collected-data-dir`."
            )

    map_img = Image.open(args.map_image).convert("RGBA")
    w, h = map_img.size
    out_img = map_img.copy()
    draw = ImageDraw.Draw(out_img, "RGBA")

    for run_i, run in enumerate(runs):
        x0_json, y0_json, pts = _load_frames(run.json_path)

        # Correction so that the first frame maps to the user-provided start.
        dx = float(args.target_start_x) - x0_json
        dy = float(args.target_start_y) - y0_json

    if args.mode == "heatmap":
        # Build heat accumulation grid at a lower resolution for speed.
        ds = max(1, int(args.heatmap_downsample))
        hs = max(1, (h + ds - 1) // ds)
        ws = max(1, (w + ds - 1) // ds)
        heat = np.zeros((hs, ws), dtype=np.float32)
        markers: list[tuple[float, float, float, float]] = []

        def in_bounds(px: float, py: float) -> bool:
            return -5 <= px <= (w + 5) and -5 <= py <= (h + 5)

        for run_i, run in enumerate(runs):
            x0_json, y0_json, pts = _load_frames(run.json_path)

            dx = float(args.target_start_x) - x0_json
            dy = float(args.target_start_y) - y0_json

            # Save start/end marker positions (so we don't re-parse the JSON).
            (x_start, y_start) = pts[0]
            (x_end, y_end) = pts[-1]

            def to_px(x_pos: float, y_pos: float) -> tuple[float, float]:
                x_corr = (x_pos + dx) * args.scale_x
                y_corr = (y_pos + dy) * args.scale_y
                px = x_corr
                py = (h - y_corr) if args.invert_y else y_corr
                return px, py

            psx, psy = to_px(float(x_start), float(y_start))
            pex, pey = to_px(float(x_end), float(y_end))
            markers.append((psx, psy, pex, pey))

            for (x_pos, y_pos) in pts:
                x_corr = (x_pos + dx) * args.scale_x
                y_corr = (y_pos + dy) * args.scale_y
                px = x_corr
                py = (h - y_corr) if args.invert_y else y_corr
                if not in_bounds(px, py):
                    continue

                hx = int(px // ds)
                hy = int(py // ds)
                if 0 <= hx < ws and 0 <= hy < hs:
                    heat[hy, hx] += 1.0

        # Normalize with log + percentile to avoid saturation.
        heat_log = np.log1p(heat)
        nonzero = heat_log[heat_log > 0]
        if nonzero.size == 0:
            nonzero_scale = 1.0
        else:
            nonzero_scale = float(np.percentile(nonzero, args.heatmap_percentile))
            if nonzero_scale <= 0:
                nonzero_scale = 1.0

        heat_norm = np.clip(heat_log / nonzero_scale, 0.0, 1.0)
        heat_scaled = np.power(heat_norm, float(args.heatmap_gamma))
        alpha_small = np.clip(heat_scaled * float(args.heatmap_max_alpha), 0, 255).astype(
            np.uint8
        )

        alpha_img_small = Image.fromarray(alpha_small, mode="L")
        # Smooth in low-res, then scale back up.
        if float(args.heatmap_blur_radius) > 0:
            alpha_img_small = alpha_img_small.filter(
                ImageFilter.GaussianBlur(radius=float(args.heatmap_blur_radius))
            )
        alpha_img = alpha_img_small.resize((w, h), resample=Image.Resampling.BILINEAR)

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        overlay.putalpha(alpha_img)
        out_img = Image.alpha_composite(out_img, overlay)
        draw = ImageDraw.Draw(out_img, "RGBA")

        # Draw start/end markers on top so you can orient yourself.
        r = max(1, int(args.radius))
        for (psx, psy, pex, pey) in markers:
            if args.colors:
                start_color = (60, 220, 80, 255)
                end_color = (255, 70, 70, 255)
            else:
                start_color = (40, 40, 40, 255)
                end_color = (140, 140, 140, 255)

            if in_bounds(psx, psy):
                _draw_point(draw, psx, psy, r=r, color=start_color)
            if in_bounds(pex, pey):
                _draw_point(draw, pex, pey, r=r, color=end_color)

    else:
        # Dots mode: draw per-frame points (optionally connected).
        for run_i, run in enumerate(runs):
            x0_json, y0_json, pts = _load_frames(run.json_path)

            dx = float(args.target_start_x) - x0_json
            dy = float(args.target_start_y) - y0_json

            alpha_255 = max(0, min(255, int(float(args.alpha) * 255.0)))

            if args.colors:
                base_colors = [
                    (30, 120, 255),
                    (255, 120, 30),
                    (120, 30, 255),
                    (30, 200, 120),
                    (200, 60, 180),
                ]
                rgb = base_colors[run_i % len(base_colors)]
                dot_color = (rgb[0], rgb[1], rgb[2], alpha_255)
                start_color = (60, 220, 80, alpha_255)
                end_color = (255, 70, 70, alpha_255)
            else:
                rgb = (110, 110, 110)
                dot_color = (rgb[0], rgb[1], rgb[2], alpha_255)
                start_color = (60, 60, 60, alpha_255)
                end_color = (160, 160, 160, alpha_255)

            pixel_points: list[tuple[float, float]] = []
            for (x_pos, y_pos) in pts:
                x_corr = (x_pos + dx) * args.scale_x
                y_corr = (y_pos + dy) * args.scale_y
                px = x_corr
                py = (h - y_corr) if args.invert_y else y_corr
                pixel_points.append((px, py))

            def in_bounds(px: float, py: float) -> bool:
                return -5 <= px <= (w + 5) and -5 <= py <= (h + 5)

            if args.connect and len(pixel_points) >= 2:
                line_points = [
                    (px, py) for (px, py) in pixel_points if in_bounds(px, py)
                ]
                if len(line_points) >= 2:
                    draw.line(
                        line_points,
                        fill=(rgb[0], rgb[1], rgb[2], max(0, alpha_255 - 60)),
                        width=1,
                    )

            r = max(1, int(args.radius))
            for i, (px, py) in enumerate(pixel_points):
                if not in_bounds(px, py):
                    continue
                color = dot_color
                if i == 0:
                    color = start_color
                elif i == len(pixel_points) - 1:
                    color = end_color
                _draw_point(draw, px, py, r=r, color=color)

    out_img.save(output_path)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

