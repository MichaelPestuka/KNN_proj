#!/usr/bin/env python3
"""
Local Tkinter client: SSH to remote GPU host, stream actions, display JPEG frames.

Example:
  pipenv run python client.py --ssh-target user@gpu-server
"""

from __future__ import annotations

import argparse
import io
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
    _RESAMPLE = Image.Resampling.NEAREST
except AttributeError:
    _RESAMPLE = Image.NEAREST  # type: ignore[attr-defined]

SCALE = 3
READY_SUBSTRING = "READY"

# Default remote layout (expanded on the remote shell)
DEFAULT_REMOTE_DIR = "/home/vpsuser/KNN_proj/remote-server-gamengen"
DEFAULT_MODEL_FOLDER = "/home/vpsuser/KNN/gameNgen-repro/sd-full-dataset-180k-steps"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GameNGen client over SSH (Tkinter + stdin/stdout protocol)")
    p.add_argument(
        "--ssh-target",
        required=True,
        help="SSH destination (e.g. user@host or Host from ~/.ssh/config)",
    )
    p.add_argument(
        "--remote-dir",
        default=DEFAULT_REMOTE_DIR,
        help=f"Remote path to remote-server-gamengen (default: {DEFAULT_REMOTE_DIR})",
    )
    p.add_argument(
        "--model-folder",
        default=DEFAULT_MODEL_FOLDER,
        help=f"Remote path to model weights (default: {DEFAULT_MODEL_FOLDER})",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="Forwarded to inference_loop.py on the remote (default: 85)",
    )
    p.add_argument(
        "--ssh-opts",
        default="",
        help='Extra ssh CLI flags as one string, e.g. \'-p 2222 -i ~/.ssh/key\'',
    )
    return p.parse_args()


def read_exact(stream: io.BufferedReader, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError(f"expected {n} bytes, got {len(buf)}")
        buf += chunk
    return buf


def build_ssh_command(args: argparse.Namespace) -> list[str]:
    remote_inner = (
        f"cd {shlex.quote(args.remote_dir)} && "
        f"exec pipenv run python inference_loop.py "
        f"--model_folder {shlex.quote(args.model_folder)} "
        f"--jpeg-quality {int(args.jpeg_quality)}"
    )
    extra = shlex.split(args.ssh_opts.strip()) if args.ssh_opts.strip() else []
    return ["ssh", "-T", *extra, args.ssh_target, remote_inner]


def action_from_keys(held: set[str]) -> int:
    """Map keyboard state to SIMPLE_MOVEMENT index (0..6)."""
    left = "Left" in held
    right = "Right" in held
    z = "z" in held or "Z" in held
    x = "x" in held or "X" in held

    if right and z and x:
        return 4
    if right and z:
        return 2
    if right and x:
        return 3
    if right:
        return 1
    if left:
        return 6
    if z:
        return 5
    return 0


class GameClientApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._held: set[str] = set()
        self._action_lock = threading.Lock()
        self._current_action = 0
        self._proc: subprocess.Popen | None = None
        self._frame_queue: queue.Queue[tuple[Image.Image, float] | Exception | None] = queue.Queue(
            maxsize=2
        )
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._action_thread: threading.Thread | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

        self.root = tk.Tk()
        self.root.title("GameNGen (SSH)")
        self.status_var = tk.StringVar(value="Connecting…")
        self.action_var = tk.StringVar(value="action: NOOP (0)")
        self.fps_var = tk.StringVar(value="fps: —")

        self._photo: ImageTk.PhotoImage | None = None
        self._last_frame_time: float | None = None
        self._fps_ema: float | None = None

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

        self.canvas = tk.Canvas(
            self.root,
            width=320 * SCALE,
            height=240 * SCALE,
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
        labels = ("NOOP", "right", "right+A", "right+B", "right+A+B", "A", "left")
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
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
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
        try:
            item = self._frame_queue.get_nowait()
        except queue.Empty:
            self.root.after(16, self._poll_frames)
            return

        if item is None:
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
        scaled = img.resize((w * SCALE, h * SCALE), _RESAMPLE)
        self._photo = ImageTk.PhotoImage(scaled)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

        self.root.after(16, self._poll_frames)

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
    app = GameClientApp(args)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
