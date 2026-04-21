# Mario run replay viewer

Small GUI to scrub through collected runs (`run.json` + JPEG frames) from [`super-mario-bros`](../super-mario-bros/). Uses **tkinter** (stdlib) and **Pillow** only—no gym / NES stack.

## Setup

**Tk is not a pip package.** Your Python must be built with Tcl/Tk (`import tkinter` / `_tkinter`).

### macOS (Homebrew)

Install Python 3.12 **and** the matching Tk bindings, then point Pipenv at that interpreter:

```bash
brew install python@3.12 python-tk@3.12
cd data-generation/super-mario-bros-viewer
pipenv install --python "$(brew --prefix python@3.12)/bin/python3.12"
```

If you already created `.venv` with a Python that lacks Tk, remove it and run `pipenv install` again with the command above.

### python.org installer

The official macOS installer usually includes Tk; use that `python3.12` for `pipenv install --python /path/to/that/python`.

## Run

From this directory, pass a **run folder** (the directory that contains `run.json`):

```bash
pipenv run python replay.py ../super-mario-bros/collected_data/random_w1s1_a1b2c3d4
```

Playback uses **Play / Pause** and speeds **0.5×–10×** (about 60 fps at 1×). Drag the timeline to scrub.

## Plot Visited Points (Heatmap)

This folder also contains a script for visualizing where Mario went during your
training runs.

- `plot_visited_points.py` overlays visited `(x_pos, y_pos)` points onto
  `SuperMarioBrosMap1-1.png`.
- By default it renders a heatmap and auto-discovers all
  `../super-mario-bros/collected_data/**/run.json`.
- Default output: `../super-mario-bros/collected_data/_all_runs_heatmap.png`.

Generate the heatmap for all runs:

```bash
pipenv run python3 plot_visited_points.py
```

Quick iteration (cap number of runs):

```bash
pipenv run python3 plot_visited_points.py --max-runs 200
```

Choose output path:

```bash
pipenv run python3 plot_visited_points.py --output my_heatmap.png
```

Optional rendering modes:

```bash
# Heatmap (default)
pipenv run python3 plot_visited_points.py --mode heatmap

# Per-frame dots instead of heatmap
pipenv run python3 plot_visited_points.py --mode dots --alpha 0.05 --radius 1
```

If you want per-run colors instead of monochrome/gray:

```bash
pipenv run python3 plot_visited_points.py --colors
```

If the heatmap is too faint or too saturated, tune:
`--heatmap-max-alpha`, `--heatmap-percentile`, `--heatmap-gamma`,
`--heatmap-blur-radius`, and `--heatmap-downsample`.

## Action direction (polar histograms)

`action_direction_polar.py` walks a **dataset root** (a folder of run directories, each with `run.json`), computes per-frame `(dx, dy)` from `info.x_pos` / `info.y_pos`, and plots one polar bar histogram per discrete action. Direction is `atan2(dy, dx)` (SMB RAM `y_pos` increases **upward**, consistent with `plot_visited_points.py` and `--invert-y`). Each subplot title includes the sample count and **average step length** `avg |Δp|` (mean of `sqrt(dx²+dy²)` in RAM units). It picks **SIMPLE** vs **COMPLEX** action naming from the data (`max(action_index) >= 7` → complex). Pairs where `|dx|` or `|dy|` exceed `--max-jump` are skipped and logged to stderr (run folder name and frame pair).

```bash
pipenv run python action_direction_polar.py ../super-mario-bros/collected_data2 -o action_polar.png
```

## Why a separate folder?

The [`super-mario-bros`](../super-mario-bros/) environment is for data collection (gym, `numpy<2`, etc.). The viewer only needs Pillow and a Tk-capable Python. Keeping it here avoids tying collection to GUI/Tk and lets you use Homebrew’s `python-tk` for the viewer venv only.
