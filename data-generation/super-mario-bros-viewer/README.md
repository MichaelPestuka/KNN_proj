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

## Why a separate folder?

The [`super-mario-bros`](../super-mario-bros/) environment is for data collection (gym, `numpy<2`, etc.). The viewer only needs Pillow and a Tk-capable Python. Keeping it here avoids tying collection to GUI/Tk and lets you use Homebrew’s `python-tk` for the viewer venv only.
