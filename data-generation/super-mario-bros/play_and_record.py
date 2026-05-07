#!/usr/bin/env python3
"""
Play Super Mario Bros from the keyboard and record frames + actions like collect_data.py.

Uses the same JoypadSpace / COMPLEX_MOVEMENT env as the random agent path. Keyboard mapping
matches game-client/client.py (arrows, Space = A, Ctrl = B).

Nothing is written to disk until you press R; play from spawn first, then R begins a run from
the current game state. --max-steps caps recorded frames only.

After each episode: press N for a new run, Q to quit.
"""

from __future__ import annotations

import argparse
import json
import platform
import secrets
import socket
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import tkinter as tk
except ImportError as exc:  # pragma: no cover
    print(
        "Tkinter is not available for this Python (_tkinter missing).\n"
        "Tk is not installed via pip—it must be compiled into Python.\n\n"
        "On macOS (Homebrew), install Tcl/Tk for Python 3.12 and use that interpreter:\n"
        "  brew install python@3.12 python-tk@3.12\n\n"
        "Or use the python.org macOS installer (includes Tk).",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

import numpy as np
from PIL import Image, ImageTk

try:
    _RESAMPLE = Image.Resampling.NEAREST
except AttributeError:
    _RESAMPLE = Image.NEAREST  # type: ignore[attr-defined]

from collect_data import (
    GAME_OVER_EXTRA_STEPS,
    action_to_list,
    capture_raw_screen,
    get_base_env,
    get_joypad_env,
    make_env,
    save_frame_jpg,
    _json_safe_info,
    _run_folder_name,
    _seed_note,
)

AGENT = "human"
FRAME_MS = int(1000 / 60)


def action_from_keys(held: set[str]) -> int:
    """Map keyboard state to COMPLEX_MOVEMENT index (0..11). Same as game-client/client.py."""
    left = "Left" in held
    right = "Right" in held
    down = "Down" in held
    up = "Up" in held
    z = "space" in held or "Space" in held
    x = "Control_L" in held or "Control_R" in held or "Control" in held

    if right and z and x:
        return 4
    if right and z:
        return 2
    if right and x:
        return 3
    if right:
        return 1
    if left and z and x:
        return 9
    if left and z:
        return 7
    if left and x:
        return 8
    if left:
        return 6
    if down:
        return 10
    if up:
        return 11
    if z:
        return 5
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Play SMB from keyboard; save frames + run.json like collect_data.py"
    )
    p.add_argument("--world", type=int, default=1, help="World 1–8 (default: 1)")
    p.add_argument("--stage", type=int, default=1, help="Stage 1–4 (default: 1)")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "test_data",
        help="Output directory for run folders (default: ./test_data next to this script)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Safety cap on recorded env steps per episode after pressing R (default: 5000)",
    )
    p.add_argument(
        "--replay-seed",
        type=int,
        default=None,
        help="Exact RNG/env seed (same as collect_data --replay-seed)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="Display scale (nearest-neighbor); default 3",
    )
    return p.parse_args()


class HumanRecordApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_dir = args.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._held: set[str] = set()
        self._display_scale = float(args.scale)

        self.env: Any = None
        self.run_dir: Path | None = None
        self.run_uuid: str = ""
        self.frames_rows: list[dict[str, Any]] = []
        self.frame_idx = 0
        self.total_reward = 0.0
        self.outcome = "max_steps"
        self.flag_get = False
        self.effective_seed = 0
        self.seed_source = ""
        self.timestamp = ""
        self.hostname = ""
        self.plat = ""
        self.py_ver = ""
        self._after_play: str | None = None
        self._menu_mode = False
        self.recording = False

        self.root = tk.Tk()
        self.root.title("SMB — human record")
        self.status_var = tk.StringVar(value="Loading…")
        self.action_var = tk.StringVar(value="action: NOOP (0)")

        self._photo: ImageTk.PhotoImage | None = None
        self._canvas_size: tuple[int, int] | None = None

        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(top, textvariable=self.status_var, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.canvas = tk.Canvas(
            self.root,
            width=256,
            height=240,
            highlightthickness=1,
            highlightbackground="#333",
            bg="#111",
        )
        self.canvas.pack(padx=8, pady=4)

        bot = tk.Frame(self.root)
        bot.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(bot, textvariable=self.action_var, anchor="w", font=("TkDefaultFont", 10)).pack(fill=tk.X)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.focus_set()

    def _on_key_press(self, event: tk.Event) -> None:  # type: ignore[name-defined]
        keysym = event.keysym
        if getattr(self, "_menu_mode", False):
            k = keysym.lower() if keysym else ""
            if k == "n":
                self._menu_mode = False
                self.start_new_run()
            elif k == "q":
                self._menu_mode = False
                self.root.destroy()
            return

        if (
            keysym in ("r", "R")
            and self.env is not None
            and not self.recording
        ):
            self._begin_recording()
            return

        if keysym:
            self._held.add(keysym)
        self._update_action_label()

    def _on_key_release(self, event: tk.Event) -> None:  # type: ignore[name-defined]
        keysym = event.keysym
        if keysym in self._held:
            self._held.discard(keysym)
        self._update_action_label()

    def _update_action_label(self) -> None:
        a = action_from_keys(self._held)
        labels = (
            "NOOP",
            "right",
            "right+A",
            "right+B",
            "right+A+B",
            "A",
            "left",
            "left+A",
            "left+B",
            "left+A+B",
            "down",
            "up",
        )
        self.action_var.set(f"action: {labels[a]} ({a})")

    def start_new_run(self) -> None:
        if self.args.replay_seed is not None:
            self.effective_seed = self.args.replay_seed
            self.seed_source = "cli_replay"
        else:
            self.effective_seed = secrets.randbelow(2**31)
            self.seed_source = "generated_at_run_start"

        self.run_uuid = ""
        self.run_dir = None
        self.timestamp = ""
        self.hostname = ""
        self.plat = ""
        self.py_ver = ""
        self.frames_rows = []
        self.frame_idx = 0
        self.total_reward = 0.0
        self.outcome = "max_steps"
        self.flag_get = False
        self.recording = False

        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass

        self.env = make_env(self.args.world, self.args.stage)
        get_base_env(self.env).seed(self.effective_seed)
        self.env.action_space.seed(self.effective_seed)
        _ob = self.env.reset()
        if isinstance(_ob, tuple):
            _ob = _ob[0]

        self._menu_mode = False
        self.status_var.set("Play — press R to start saving frames (arrows, Space=A, Ctrl=B)")
        self._cancel_after_play()
        self._after_play = self.root.after(FRAME_MS, self._tick_playing)
        self._refresh_canvas()

    def _begin_recording(self) -> None:
        if self.recording or self.env is None or self._menu_mode:
            return
        self.run_uuid = uuid.uuid4().hex[:8]
        folder_name = _run_folder_name(AGENT, self.args.world, self.args.stage, self.run_uuid)
        self.run_dir = self.output_dir / folder_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().isoformat()
        self.hostname = socket.gethostname()
        self.plat = platform.platform()
        self.py_ver = platform.python_version()
        self.frames_rows = []
        self.frame_idx = 0
        self.total_reward = 0.0
        self.outcome = "max_steps"
        self.flag_get = False
        self.recording = True
        self.status_var.set(f"Recording → {self.run_dir.name}  (arrows, Space=A, Ctrl=B)")

    def _cancel_after_play(self) -> None:
        if self._after_play is not None:
            try:
                self.root.after_cancel(self._after_play)
            except Exception:
                pass
            self._after_play = None

    def _filename_for(self, i: int) -> str:
        return f"frame_{self.run_uuid}_{i:06d}.jpg"

    def _append_terminal_and_gameover(self, info: dict[str, Any]) -> None:
        assert self.env is not None and self.run_dir is not None
        term_fname = self._filename_for(self.frame_idx)
        save_frame_jpg(self.run_dir / term_fname, capture_raw_screen(self.env))
        term_info = _json_safe_info(dict(info))
        self.frames_rows.append(
            {
                "frame": self.frame_idx,
                "filename": term_fname,
                "action_index": None,
                "action": None,
                "reward": 0.0,
                "done": True,
                "info": term_info,
                "note": "terminal_observation_after_done",
            }
        )
        self.frame_idx += 1

        if not self.flag_get:
            base = get_base_env(self.env)
            noop_byte = int(get_joypad_env(self.env)._action_map[0])
            for g in range(GAME_OVER_EXTRA_STEPS):
                base._frame_advance(noop_byte)
                raw = capture_raw_screen(self.env)
                go_fname = self._filename_for(self.frame_idx)
                save_frame_jpg(self.run_dir / go_fname, raw)
                self.frames_rows.append(
                    {
                        "frame": self.frame_idx,
                        "filename": go_fname,
                        "action_index": 0,
                        "action": action_to_list(0),
                        "reward": 0.0,
                        "done": True,
                        "info": {"game_over_sequence": True, "sequence_index": g},
                        "note": "game_over_sequence",
                    }
                )
                self.frame_idx += 1

    def _write_run_json(self) -> None:
        assert self.run_dir is not None
        run_json: dict[str, Any] = {
            "agent": AGENT,
            "environment": f"SuperMarioBros-{self.args.world}-{self.args.stage}-v0",
            "world": self.args.world,
            "stage": self.args.stage,
            "run_id": self.run_uuid,
            "timestamp": self.timestamp,
            "hostname": self.hostname,
            "platform": self.plat,
            "python_version": self.py_ver,
            "total_frames": len(self.frames_rows),
            "outcome": self.outcome,
            "total_reward": self.total_reward,
            "max_steps_cap": self.args.max_steps,
            "seed": self.effective_seed,
            "seed_source": self.seed_source,
            "seed_note": _seed_note(self.seed_source),
            "frames": self.frames_rows,
        }
        with open(self.run_dir / "run.json", "w", encoding="utf-8") as f:
            json.dump(run_json, f, indent=2)

    def _end_episode(self) -> None:
        self.recording = False
        self._cancel_after_play()
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

        written = ""
        if self.run_dir is not None:
            self._write_run_json()
            written = str(self.run_dir)
        self.run_dir = None

        self._menu_mode = True
        self.status_var.set(
            f"Saved ({self.outcome}). N = new run, Q = quit."
            + (f"  → {written}" if written else "")
        )

    def _end_episode_no_save(self) -> None:
        """Episode ended (death / flag) before R was pressed — no run folder."""
        self._cancel_after_play()
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None
        self.run_dir = None
        self.run_uuid = ""
        self.recording = False
        self._menu_mode = True
        self.status_var.set("Episode ended (nothing saved — press R next time to record). N = new run, Q = quit.")

    def _tick_playing(self) -> None:
        self._after_play = None
        if self.env is None:
            return

        if not self.recording:
            action_index = action_from_keys(self._held)
            step_out = self.env.step(action_index)
            if len(step_out) == 5:
                _next, _reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                _next, _reward, done, info = step_out
            if isinstance(_next, tuple):
                _next = _next[0]

            self._refresh_canvas()

            if done:
                self._end_episode_no_save()
                return

            self._after_play = self.root.after(FRAME_MS, self._tick_playing)
            return

        assert self.run_dir is not None

        if self.frame_idx >= self.args.max_steps:
            self._end_episode()
            return

        action_index = action_from_keys(self._held)
        rgb_before = capture_raw_screen(self.env)
        fname = self._filename_for(self.frame_idx)
        save_frame_jpg(self.run_dir / fname, rgb_before)

        step_out = self.env.step(action_index)
        if len(step_out) == 5:
            _next, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            _next, reward, done, info = step_out
        if isinstance(_next, tuple):
            _next = _next[0]

        self.total_reward += float(reward)
        info_safe = _json_safe_info(dict(info))
        self.frames_rows.append(
            {
                "frame": self.frame_idx,
                "filename": fname,
                "action_index": action_index,
                "action": action_to_list(action_index),
                "reward": float(reward),
                "done": bool(done),
                "info": info_safe,
            }
        )
        self.frame_idx += 1

        self._refresh_canvas()

        if done:
            self.flag_get = bool(info.get("flag_get", False))
            self.outcome = "flag" if self.flag_get else "death"
            self._append_terminal_and_gameover(info)
            self._end_episode()
            return

        self._after_play = self.root.after(FRAME_MS, self._tick_playing)

    def _refresh_canvas(self) -> None:
        if self.env is None:
            return
        rgb = capture_raw_screen(self.env)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        img = Image.fromarray(rgb)
        w, h = img.size
        dw = max(1, int(round(w * self._display_scale)))
        dh = max(1, int(round(h * self._display_scale)))
        scaled = img.resize((dw, dh), _RESAMPLE)
        self._photo = ImageTk.PhotoImage(scaled)
        if (dw, dh) != self._canvas_size:
            self.canvas.config(width=dw, height=dh)
            self._canvas_size = (dw, dh)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    def _on_close(self) -> None:
        self._cancel_after_play()
        if (
            self.recording
            and self.env is not None
            and self.run_dir is not None
            and self.frames_rows
        ):
            self.outcome = "aborted"
            self._write_run_json()
        self.recording = False
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None
        self.root.destroy()

    def run(self) -> None:
        self.start_new_run()
        self.root.mainloop()


def main() -> int:
    args = parse_args()
    if not (1 <= args.world <= 8):
        print("--world must be 1–8", file=sys.stderr)
        return 1
    if not (1 <= args.stage <= 4):
        print("--stage must be 1–4", file=sys.stderr)
        return 1
    if args.max_steps < 1:
        print("--max-steps must be >= 1", file=sys.stderr)
        return 1

    app = HumanRecordApp(args)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
