#!/usr/bin/env python3
"""
Replay a collected Super Mario Bros run: frames + actions in one window.

Usage (from this directory):
  pipenv run python replay.py path/to/run_folder

Example:
  pipenv run python replay.py ../super-mario-bros/collected_data/random_w1s1_001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError as exc:  # pragma: no cover
    print(
        "Tkinter is not available for this Python (_tkinter missing).\n"
        "Tk is not installed via pip—it must be compiled into Python.\n\n"
        "On macOS (Homebrew), install Tcl/Tk for Python 3.12 and use that interpreter:\n"
        "  brew install python@3.12 python-tk@3.12\n"
        "  cd data-generation/super-mario-bros-viewer\n"
        "  pipenv install --python \"$(brew --prefix python@3.12)/bin/python3.12\"\n\n"
        "Or use the python.org macOS installer (includes Tk).",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from PIL import Image, ImageTk

try:
    _RESAMPLE = Image.Resampling.NEAREST
except AttributeError:
    _RESAMPLE = Image.NEAREST  # type: ignore[attr-defined]

# ~NES 60 Hz
BASE_FRAME_MS = 1000 / 60
SCALE = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay a run folder (run.json + JPEG frames) with actions."
    )
    p.add_argument(
        "run_dir",
        type=Path,
        help="Path to a run directory containing run.json and frame_*.jpg files",
    )
    return p.parse_args()


class MarioReplayViewer:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()
        self.json_path = self.run_dir / "run.json"
        if not self.json_path.is_file():
            raise FileNotFoundError(f"Missing run.json: {self.json_path}")

        with open(self.json_path, encoding="utf-8") as f:
            self.meta: dict[str, Any] = json.load(f)

        self.frames_data: list[dict[str, Any]] = self.meta.get("frames") or []
        if not self.frames_data:
            raise ValueError(f"No frames in {self.json_path}")

        self.n_frames = len(self.frames_data)
        self.images: list[Image.Image] = []
        self._photo: ImageTk.PhotoImage | None = None
        self._current_index = 0
        self._playing = False
        self._after_id: str | None = None
        self._speed = 1.0

        self.root = tk.Tk()
        self.root.title(f"Mario replay — {self.run_dir.name}")
        self.root.minsize(400, 300)

        self._build_ui()
        self.root.after(50, self._start_load)

    def _build_ui(self) -> None:
        self.loading_frame = tk.Frame(self.root)
        self.loading_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        tk.Label(
            self.loading_frame,
            text="Loading frames…",
            font=("TkDefaultFont", 12),
        ).pack(pady=(0, 8))
        self.progress = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            self.loading_frame,
            variable=self.progress,
            maximum=100.0,
            length=400,
        )
        self.progress_bar.pack(fill=tk.X)
        self.progress_label = tk.Label(self.loading_frame, text="0 / 0")
        self.progress_label.pack(pady=8)

        self.main_frame = tk.Frame(self.root)

        # Frame image
        self.img_canvas = tk.Canvas(
            self.main_frame,
            highlightthickness=1,
            highlightbackground="#333",
            bg="#111",
        )
        self.img_canvas.pack(pady=(0, 4))

        # Controller + game-over row
        ctrl_row = tk.Frame(self.main_frame)
        ctrl_row.pack(fill=tk.X, pady=4)

        self.pad_canvas = tk.Canvas(
            ctrl_row,
            width=420,
            height=140,
            highlightthickness=1,
            highlightbackground="#555",
            bg="#222",
        )
        self.pad_canvas.pack(side=tk.LEFT, padx=(0, 8))

        self.game_over_label = tk.Label(
            ctrl_row,
            text="",
            fg="#c44",
            font=("TkDefaultFont", 11, "bold"),
            anchor="w",
        )
        self.game_over_label.pack(side=tk.LEFT, fill=tk.Y)

        # Transport
        transport = tk.Frame(self.main_frame)
        transport.pack(fill=tk.X, pady=4)

        for text, cmd in (
            ("|<", self._first_frame),
            ("<<", self._step_back),
            ("Play / Pause", self._toggle_play),
            (">>", self._step_fwd),
            (">|", self._last_frame),
        ):
            tk.Button(transport, text=text, command=cmd, width=10).pack(
                side=tk.LEFT, padx=2
            )

        slider_row = tk.Frame(self.main_frame)
        slider_row.pack(fill=tk.X, pady=4)
        # Do not use Scale(command=...): Tk may invoke it asynchronously after
        # programmatic .set(), which cleared our guard and cancelled playback
        # every frame. User scrubbing uses mouse bindings only.
        self.timeline = tk.Scale(
            slider_row,
            from_=0,
            to=max(0, self.n_frames - 1),
            orient=tk.HORIZONTAL,
            showvalue=False,
            length=560,
        )
        self.timeline.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.timeline.bind("<B1-Motion>", self._on_timeline_scrub)
        self.timeline.bind("<ButtonRelease-1>", self._on_timeline_scrub)
        self.frame_counter = tk.Label(slider_row, text="0 / 0", width=14, anchor="e")
        self.frame_counter.pack(side=tk.RIGHT, padx=(8, 0))

        speed_row = tk.Frame(self.main_frame)
        speed_row.pack(fill=tk.X, pady=2)
        tk.Label(speed_row, text="Speed:").pack(side=tk.LEFT, padx=(0, 8))
        self.speed_var = tk.DoubleVar(value=1.0)
        for label, val in (
            ("0.5x", 0.5),
            ("1x", 1.0),
            ("2x", 2.0),
            ("4x", 4.0),
            ("10x", 10.0),
        ):
            tk.Radiobutton(
                speed_row,
                text=label,
                variable=self.speed_var,
                value=val,
                command=self._on_speed_change,
            ).pack(side=tk.LEFT, padx=4)

        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.bind("<Left>", lambda e: self._step_back())
        self.root.bind("<Right>", lambda e: self._step_fwd())
        self.root.bind("<Home>", lambda e: self._first_frame())
        self.root.bind("<End>", lambda e: self._last_frame())

    def _on_speed_change(self) -> None:
        self._speed = float(self.speed_var.get())

    def _start_load(self) -> None:
        self._load_all_images()

    def _load_all_images(self) -> None:
        """Decode every JPEG into a PIL Image (original size) in memory."""
        n = self.n_frames
        self.images: list[Image.Image] = []
        for i, entry in enumerate(self.frames_data):
            fname = entry.get("filename")
            if not fname:
                raise ValueError(f"Frame {i} missing filename")
            path = self.run_dir / fname
            if not path.is_file():
                raise FileNotFoundError(f"Missing image: {path}")
            img = Image.open(path).convert("RGB")
            self.images.append(img)
            if i % 50 == 0 or i == n - 1:
                pct = 100.0 * (i + 1) / n
                self.progress.set(pct)
                self.progress_label.config(text=f"{i + 1} / {n}")
                self.root.update_idletasks()

        self.loading_frame.destroy()
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        w, h = self.images[0].size
        self._disp_w, self._disp_h = w * SCALE, h * SCALE
        self.img_canvas.config(width=self._disp_w, height=self._disp_h)

        self._draw_pad_static()
        self._show_frame(0)
        self.frame_counter.config(text=f"1 / {self.n_frames}")

    def _action_set(self, entry: dict[str, Any]) -> set[str]:
        raw = entry.get("action") or []
        if not isinstance(raw, list):
            return set()
        out: set[str] = set()
        for a in raw:
            if isinstance(a, str) and a != "NOOP":
                out.add(a)
        return out

    def _is_game_over_sequence(self, entry: dict[str, Any]) -> bool:
        info = entry.get("info")
        if not isinstance(info, dict):
            return False
        return bool(info.get("game_over_sequence"))

    def _show_frame(self, index: int) -> None:
        index = max(0, min(self.n_frames - 1, index))
        self._current_index = index
        entry = self.frames_data[index]

        pil = self.images[index]
        scaled = pil.resize(
            (pil.size[0] * SCALE, pil.size[1] * SCALE), _RESAMPLE
        )
        self._photo = ImageTk.PhotoImage(scaled)
        self.img_canvas.delete("all")
        self.img_canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

        self._update_pad_highlights(self._action_set(entry))

        if self._is_game_over_sequence(entry):
            self.game_over_label.config(text="GAME OVER SEQUENCE")
        else:
            self.game_over_label.config(text="")

        self.frame_counter.config(text=f"{index + 1} / {self.n_frames}")
        self.timeline.set(index)

    def _draw_pad_static(self) -> None:
        c = self.pad_canvas
        c.delete("all")
        cx, cy = 90, 70
        u = 28
        # D-pad cross (outline)
        for name, dx, dy in (
            ("left", -u, 0),
            ("right", u, 0),
            ("up", 0, -u),
            ("down", 0, u),
        ):
            x0, y0 = cx + dx - u // 2, cy + dy - u // 2
            x1, y1 = x0 + u, y0 + u
            c.create_rectangle(
                x0, y0, x1, y1, fill="#444", outline="#666", tags=("pad", name)
            )
        # Center
        c.create_rectangle(
            cx - u // 2,
            cy - u // 2,
            cx + u // 2,
            cy + u // 2,
            fill="#333",
            outline="#555",
            tags=("pad", "center"),
        )

        # B, A (NES order: B left of A)
        self._bx0, self._by0 = 260, 45
        r = 22
        c.create_oval(
            self._bx0 - r,
            self._by0 - r,
            self._bx0 + r,
            self._by0 + r,
            fill="#444",
            outline="#666",
            tags=("btn", "B"),
        )
        c.create_text(self._bx0, self._by0, text="B", fill="#ccc", font=("Helvetica", 12, "bold"))

        self._ax0, self._ay0 = 330, 45
        c.create_oval(
            self._ax0 - r,
            self._ay0 - r,
            self._ax0 + r,
            self._ay0 + r,
            fill="#444",
            outline="#666",
            tags=("btn", "A"),
        )
        c.create_text(self._ax0, self._ay0, text="A", fill="#ccc", font=("Helvetica", 12, "bold"))

    def _update_pad_highlights(self, actions: set[str]) -> None:
        c = self.pad_canvas
        active = "#c62828"
        idle_pad = "#444"
        idle_btn = "#444"
        idle_center = "#333"

        for name in ("left", "right", "up", "down"):
            fill = active if name in actions else idle_pad
            try:
                c.itemconfigure(name, fill=fill)
            except tk.TclError:
                pass

        try:
            c.itemconfigure("center", fill=idle_center)
        except tk.TclError:
            pass

        for letter in ("A", "B"):
            fill = active if letter in actions else idle_btn
            try:
                c.itemconfigure(letter, fill=fill)
            except tk.TclError:
                pass

    def _on_timeline_scrub(self, event: object = None) -> None:
        """Slider moved by the user (not by programmatic .set() during playback)."""
        self._cancel_play()
        idx = int(round(float(self.timeline.get())))
        self._show_frame(idx)

    def _cancel_play(self) -> None:
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self._playing = False

    def _toggle_play(self) -> None:
        if self._playing:
            self._cancel_play()
        else:
            self._playing = True
            self._schedule_next()

    def _schedule_next(self) -> None:
        if not self._playing:
            return
        delay = max(1, int(BASE_FRAME_MS / self._speed))
        self._after_id = self.root.after(delay, self._tick)

    def _tick(self) -> None:
        self._after_id = None
        if not self._playing:
            return
        nxt = self._current_index + 1
        if nxt >= self.n_frames:
            self._playing = False
            return
        self._show_frame(nxt)
        self._schedule_next()

    def _step_back(self) -> None:
        self._cancel_play()
        self._show_frame(self._current_index - 1)

    def _step_fwd(self) -> None:
        self._cancel_play()
        self._show_frame(self._current_index + 1)

    def _first_frame(self) -> None:
        self._cancel_play()
        self._show_frame(0)

    def _last_frame(self) -> None:
        self._cancel_play()
        self._show_frame(self.n_frames - 1)

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    args = parse_args()
    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        return 1
    try:
        app = MarioReplayViewer(run_dir)
    except (OSError, ValueError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        return 1
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
