#!/usr/bin/env python3
"""
Local Tkinter client: SSH to remote GPU host, stream actions, display JPEG frames.

Example:
  pipenv run python client.py --mode gamengen
  pipenv run python client.py --mode controlnet
"""

from __future__ import annotations

import argparse
import io
import posixpath
import queue
import shlex
import struct
import subprocess
import sys
import threading
import time

try:
    import tkinter as tk
except ImportError as exc:  # pragma: no cover
    print("Tkinter is required. Install a Python build with Tk support.", file=sys.stderr)
    raise SystemExit(1) from exc

from PIL import Image, ImageTk

try:
    _RESAMPLE_NEAREST = Image.Resampling.NEAREST
    _RESAMPLE_SMOOTH = Image.Resampling.LANCZOS
except AttributeError:
    _RESAMPLE_NEAREST = Image.NEAREST  # type: ignore[attr-defined]
    _RESAMPLE_SMOOTH = Image.LANCZOS  # type: ignore[attr-defined]

READY_SUBSTRING = "READY"

DEFAULT_SSH_TARGET = "vpsuser@pro6000b.foukec.cz"

# Filled after --mode if --remote-dir / --model-folder omitted
MODE_DEFAULTS: dict[str, dict[str, str]] = {
    "gamengen": {
        "remote_dir": "/home/vpsuser/KNN_proj/remote-server-gamengen",
        "model_folder": "/mnt/pro6000/data/gameNgen-checkpoints/sd-full-dataset-250k-steps",
    },
    "controlnet": {
        "remote_dir": "/home/vpsuser/KNN_proj/remote-server-controlnet",
        "model_folder": "/home/vpsuser/KNN_proj/controlnet_based/inference_checkpoints",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Game client over SSH (Tkinter + stdin/stdout protocol; GameNGen or ControlNet)"
    )
    p.add_argument(
        "--ssh-target",
        default=DEFAULT_SSH_TARGET,
        help=f"SSH destination (default: {DEFAULT_SSH_TARGET})",
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=["gamengen", "controlnet"],
        help="Remote server: GameNGen UNet loop or ControlNet-based loop",
    )
    p.add_argument(
        "--remote-dir",
        default=None,
        help="Remote cwd for pipenv/inference_loop.py (default depends on --mode)",
    )
    p.add_argument(
        "--model-folder",
        default=None,
        help="Remote model folder (default depends on --mode)",
    )
    p.add_argument(
        "--initial-image",
        default=None,
        dest="initial_image",
        help="ControlNet only: remote path to initial frame (default: ../remote-server-gamengen/sample_images/start.jpg from --remote-dir)",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="Forwarded to inference_loop.py on the remote (default: 85)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Display scale vs native frame size (default: 2 for gamengen 320x240, 1 for controlnet 512x512)",
    )
    p.add_argument(
        "--ssh-opts",
        default="",
        help='Extra ssh CLI flags as one string, e.g. \'-p 2222 -i ~/.ssh/key\'',
    )
    return p.parse_args()


def apply_mode_defaults(args: argparse.Namespace) -> None:
    defaults = MODE_DEFAULTS[args.mode]
    if args.remote_dir is None:
        args.remote_dir = defaults["remote_dir"]
    if args.model_folder is None:
        args.model_folder = defaults["model_folder"]
    if args.mode == "controlnet" and args.initial_image is None:
        args.initial_image = posixpath.normpath(
            posixpath.join(
                args.remote_dir,
                "..",
                "remote-server-gamengen",
                "sample_images",
                "start.jpg",
            )
        )
    if args.scale is None:
        args.scale = 1.0 if args.mode == "controlnet" else 2.0


def read_exact(stream: io.BufferedReader, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError(f"expected {n} bytes, got {len(buf)}")
        buf += chunk
    return buf


def build_ssh_command(args: argparse.Namespace) -> list[str]:
    if args.mode == "gamengen":
        remote_inner = (
            f"cd {shlex.quote(args.remote_dir)} && "
            f"exec pipenv run python inference_loop.py "
            f"--model_folder {shlex.quote(args.model_folder)} "
            f"--jpeg-quality {int(args.jpeg_quality)}"
        )
    else:
        assert args.initial_image is not None
        remote_inner = (
            f"cd {shlex.quote(args.remote_dir)} && "
            f"exec pipenv run python inference_loop.py "
            f"--model_folder {shlex.quote(args.model_folder)} "
            f"--initial-image {shlex.quote(args.initial_image)} "
            f"--jpeg-quality {int(args.jpeg_quality)}"
        )
    extra = shlex.split(args.ssh_opts.strip()) if args.ssh_opts.strip() else []
    return ["ssh", "-T", *extra, args.ssh_target, remote_inner]


def action_from_keys(held: set[str]) -> int:
    """Map keyboard state to COMPLEX_MOVEMENT index (0..11).

    Arrow keys = directions, Space = A button (jump), Ctrl = B button (run).
    """
    left = "Left" in held
    right = "Right" in held
    down = "Down" in held
    up = "Up" in held
    z = "space" in held or "Space" in held
    x = "Control_L" in held or "Control_R" in held or "Control" in held

    if right and z and x:
        return 4   # right + A + B
    if right and z:
        return 2   # right + A
    if right and x:
        return 3   # right + B
    if right:
        return 1   # right
    if left and z and x:
        return 9   # left + A + B
    if left and z:
        return 7   # left + A
    if left and x:
        return 8   # left + B
    if left:
        return 6   # left
    if down:
        return 10  # down
    if up:
        return 11  # up
    if z:
        return 5   # A
    return 0       # NOOP


class GameClientApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._display_scale = float(args.scale)
        self._held: set[str] = set()
        self._action_lock = threading.Lock()
        self._current_action = 0
        self._proc: subprocess.Popen | None = None
        self._frame_queue: queue.Queue[tuple[Image.Image, float] | Exception | None] = queue.Queue(
            maxsize=8
        )
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._action_thread: threading.Thread | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

        self.root = tk.Tk()
        title = "ControlNet (SSH)" if args.mode == "controlnet" else "GameNGen (SSH)"
        self.root.title(title)
        self.status_var = tk.StringVar(value="Connecting…")
        self.action_var = tk.StringVar(value="action: NOOP (0)")
        self.fps_var = tk.StringVar(value="fps: —")

        self._photo: ImageTk.PhotoImage | None = None
        self._last_frame_time: float | None = None
        self._fps_ema: float | None = None
        self._canvas_size: tuple[int, int] | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.focus_set()

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(top, textvariable=self.status_var, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(top, textvariable=self.fps_var, width=18, anchor="e").pack(side=tk.RIGHT)

        if self.args.mode == "controlnet":
            init_w = int(512 * self._display_scale)
            init_h = int(512 * self._display_scale)
        else:
            init_w = int(320 * self._display_scale)
            init_h = int(240 * self._display_scale)

        self.canvas = tk.Canvas(
            self.root,
            width=max(1, init_w),
            height=max(1, init_h),
            highlightthickness=1,
            highlightbackground="#333",
            bg="#111",
        )
        self.canvas.pack(padx=8, pady=4)

        bot = tk.Frame(self.root)
        bot.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(bot, textvariable=self.action_var, anchor="w", font=("TkDefaultFont", 10)).pack(
            fill=tk.X
        )

    def _on_key_press(self, event: tk.Event) -> None:  # type: ignore[name-defined]
        keysym = event.keysym
        if keysym:
            self._held.add(keysym)
        self._update_action_from_keys()

    def _on_key_release(self, event: tk.Event) -> None:  # type: ignore[name-defined]
        keysym = event.keysym
        if keysym in self._held:
            self._held.discard(keysym)
        self._update_action_from_keys()

    def _update_action_from_keys(self) -> None:
        a = action_from_keys(self._held)
        with self._action_lock:
            self._current_action = a
        labels = (
            "NOOP", "right", "right+A", "right+B", "right+A+B",
            "A", "left", "left+A", "left+B", "left+A+B", "down", "up",
        )
        self.action_var.set(f"action: {labels[a]} ({a})")

    def _stderr_reader(self, proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        try:
            for line in iter(proc.stderr.readline, b""):
                if not line:
                    break
                try:
                    text = line.decode("utf-8", errors="replace")
                except Exception:
                    text = str(line)
                sys.stderr.write(text)
                sys.stderr.flush()
                if READY_SUBSTRING in text:
                    self._ready.set()
        except Exception as e:  # pragma: no cover
            self._frame_queue.put(e)

    def _action_sender(self, proc: subprocess.Popen) -> None:
        """Send current action at ~60 Hz, independent of frame receipt."""
        assert proc.stdin is not None
        try:
            if not self._ready.wait(timeout=600):
                self._frame_queue.put(TimeoutError("Remote did not become READY in time"))
                return

            while not self._stop.is_set():
                with self._action_lock:
                    action_byte = self._current_action & 0xFF
                proc.stdin.write(bytes([action_byte]))
                proc.stdin.flush()
                time.sleep(1 / 60)
        except (BrokenPipeError, OSError):
            pass
        except Exception as e:
            if not self._stop.is_set():
                self._frame_queue.put(e)

    def _frame_reader(self, proc: subprocess.Popen) -> None:
        """Read frames continuously, as fast as the server produces them."""
        assert proc.stdout is not None
        try:
            if not self._ready.wait(timeout=600):
                return

            while not self._stop.is_set():
                raw_len = read_exact(proc.stdout, 4)
                (length,) = struct.unpack(">I", raw_len)
                if length > 50 * 1024 * 1024:
                    raise ValueError(f"frame too large: {length}")
                jpeg = read_exact(proc.stdout, length)
                img = Image.open(io.BytesIO(jpeg)).convert("RGB")
                now = time.perf_counter()
                try:
                    self._frame_queue.put((img, now), timeout=1.0)
                except queue.Full:
                    while True:
                        try:
                            self._frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    self._frame_queue.put((img, now))
        except Exception as e:
            if not self._stop.is_set():
                self._frame_queue.put(e)

    def start_ssh(self) -> None:
        cmd = build_ssh_command(self.args)
        sys.stderr.write(f"[client] {shlex.join(cmd)}\n")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._stderr_thread = threading.Thread(
            target=self._stderr_reader, args=(self._proc,), daemon=True
        )
        self._stderr_thread.start()
        self._action_thread = threading.Thread(
            target=self._action_sender, args=(self._proc,), daemon=True
        )
        self._action_thread.start()
        self._reader_thread = threading.Thread(
            target=self._frame_reader, args=(self._proc,), daemon=True
        )
        self._reader_thread.start()
        self.status_var.set("Connected — streaming")
        self._poll_frames()

    def _poll_frames(self) -> None:
        item: tuple[Image.Image, float] | Exception | None = None
        while True:
            try:
                nxt = self._frame_queue.get_nowait()
            except queue.Empty:
                break
            item = nxt

        if item is None:
            self.root.after(8, self._poll_frames)
            return
        if isinstance(item, Exception):
            self.status_var.set(f"Error: {item!s}")
            return

        img, t = item
        if self._last_frame_time is not None:
            dt = t - self._last_frame_time
            if dt > 1e-6:
                inst_fps = 1.0 / dt
                self._fps_ema = inst_fps if self._fps_ema is None else 0.2 * inst_fps + 0.8 * self._fps_ema
                self.fps_var.set(f"fps: {self._fps_ema:.1f}")
        self._last_frame_time = t

        w, h = img.size
        dw = max(1, int(round(w * self._display_scale)))
        dh = max(1, int(round(h * self._display_scale)))
        resample = _RESAMPLE_SMOOTH if self.args.mode == "controlnet" else _RESAMPLE_NEAREST
        scaled = img.resize((dw, dh), resample)
        self._photo = ImageTk.PhotoImage(scaled)
        if (dw, dh) != self._canvas_size:
            self.canvas.config(width=dw, height=dh)
            self._canvas_size = (dw, dh)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

        self.root.after(8, self._poll_frames)

    def _on_close(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self.root.destroy()

    def run(self) -> None:
        self.start_ssh()
        self.root.mainloop()


def main() -> int:
    args = parse_args()
    apply_mode_defaults(args)
    app = GameClientApp(args)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
