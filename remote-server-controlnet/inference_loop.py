#!/usr/bin/env python3
"""
Remote ControlNet loop: read 1-byte COMPLEX_MOVEMENT action indices from stdin,
write length-prefixed JPEG frames to stdout.

Protocol (same as remote-server-gamengen/inference_loop.py):
  - Client sends action bytes continuously (e.g. at 60 Hz).
  - Server reads them in a background thread, keeping only the latest.
  - Server runs inference as fast as possible using the most recent action,
    and writes length-prefixed JPEG frames to stdout.

Designed to be launched by the local client over SSH, e.g.:
  ssh -T user@host 'cd /path/to/remote-server-controlnet && pipenv run python inference_loop.py ...'
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import struct
import sys
import threading

import numpy as np
import torch
from PIL import Image

# Import InferenceEngine from sibling package controlnet_based/
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CONTROLNET_BASED = os.path.join(_REPO_ROOT, "controlnet_based")
if _CONTROLNET_BASED not in sys.path:
    sys.path.insert(0, _CONTROLNET_BASED)

from inference_engine import InferenceEngine  # noqa: E402

READY_LINE = "READY\n"

# gym_super_mario_bros.actions.COMPLEX_MOVEMENT (12 actions) -> 5-D [left, right, A, B, NOOP]
NUM_ACTIONS = 12
_ACTION_VECTORS: list[list[float]] = [
    [0.0, 0.0, 0.0, 0.0, 1.0],  # NOOP
    [0.0, 1.0, 0.0, 0.0, 0.0],  # right
    [0.0, 1.0, 1.0, 0.0, 0.0],  # right + A
    [0.0, 1.0, 0.0, 1.0, 0.0],  # right + B
    [0.0, 1.0, 1.0, 1.0, 0.0],  # right + A + B
    [0.0, 0.0, 1.0, 0.0, 0.0],  # A
    [1.0, 0.0, 0.0, 0.0, 0.0],  # left
    [1.0, 0.0, 1.0, 0.0, 0.0],  # left + A
    [1.0, 0.0, 0.0, 1.0, 0.0],  # left + B
    [1.0, 0.0, 1.0, 1.0, 0.0],  # left + A + B
    [0.0, 0.0, 0.0, 0.0, 1.0],  # down -> NOOP in 5-D space
    [0.0, 0.0, 0.0, 0.0, 1.0],  # up -> NOOP in 5-D space
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    default_initial = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "remote-server-gamengen", "sample_images", "start.jpg")
    )
    p = argparse.ArgumentParser(description="ControlNet stdin/stdout inference loop")
    p.add_argument(
        "--model_folder",
        type=str,
        required=True,
        help="Folder containing transformer.pt and controlnet/ subfolder",
    )
    p.add_argument(
        "--initial-image",
        type=str,
        default=default_initial,
        dest="initial_image",
        help=f"Path to initial RGB frame (default: {default_initial})",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for frames sent to stdout (default: 85)",
    )
    return p.parse_args()


def to_jpeg(arr: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def main() -> None:
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    target_dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32

    transformer_path = os.path.join(args.model_folder, "transformer.pt")
    controlnet_path = os.path.join(args.model_folder, "controlnet")

    logger.info(
        "Loading ControlNet engine: transformer=%s controlnet=%s device=%s",
        transformer_path,
        controlnet_path,
        device,
    )

    engine = InferenceEngine(
        transformer_path=transformer_path,
        controlnet_path=controlnet_path,
        device=str(device),
        target_dtype=target_dtype,
        compile=False,
    )
    engine.reset(initial_image_path=args.initial_image)

    # Pre-build action tensors on device
    action_tensors = [
        torch.tensor(v, device=engine.device, dtype=engine.dtype) for v in _ACTION_VECTORS
    ]

    sys.stderr.write(READY_LINE)
    sys.stderr.flush()
    logger.info("Model ready; streaming with latest stdin actions")

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    latest_action = 0
    action_lock = threading.Lock()
    eof_event = threading.Event()

    def action_reader() -> None:
        nonlocal latest_action
        try:
            while True:
                b = stdin.read(1)
                if not b:
                    eof_event.set()
                    return
                val = b[0]
                if val >= NUM_ACTIONS:
                    val = 0
                with action_lock:
                    latest_action = val
        except Exception:
            eof_event.set()

    threading.Thread(target=action_reader, daemon=True).start()

    try:
        while not eof_event.is_set():
            with action_lock:
                idx = latest_action
            action_tensor = action_tensors[idx]
            img = engine.step(action_tensor)
            jpeg_bytes = to_jpeg(img, quality=args.jpeg_quality)
            header = struct.pack(">I", len(jpeg_bytes))
            stdout.write(header + jpeg_bytes)
            stdout.flush()
    except BrokenPipeError:
        logger.info("Broken pipe on stdout; exiting")
    except Exception:
        logger.exception("Fatal error in inference loop")
        raise


if __name__ == "__main__":
    main()
