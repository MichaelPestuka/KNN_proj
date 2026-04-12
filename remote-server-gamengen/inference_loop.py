#!/usr/bin/env python3
"""
Remote GameNGen loop: read 1-byte action indices from stdin, write length-prefixed JPEG frames to stdout.

Protocol (decoupled):
  - Client sends action bytes continuously (e.g. at 60 Hz).
  - Server reads them in a background thread, keeping only the latest.
  - Server runs inference as fast as possible, using the most recent action,
    and writes length-prefixed JPEG frames to stdout without waiting for the
    client to acknowledge each frame.

Designed to be launched by the local client over SSH, e.g.:
  ssh -T user@host 'cd /home/knn/KNN_proj/remote-server-gamengen && pipenv run python inference_loop.py --model_folder ~/weights'
"""

from __future__ import annotations

import argparse
import logging
import struct
import sys
import threading

import torch

from config import NUM_ACTIONS
from inference import InferenceEngine
from model import load_model

READY_LINE = "READY\n"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GameNGen stdin/stdout inference loop")
    p.add_argument(
        "--model_folder",
        type=str,
        required=True,
        help="Path to model weights (local path on this machine)",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for frames sent to stdout (default: 85)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    vae_device = None
    if device.type == "cuda" and torch.cuda.device_count() >= 2:
        vae_device = torch.device("cuda", 1)
        logger.info(
            "Dual GPU detected: UNet on %s, VAE on %s (pipelined)",
            device,
            vae_device,
        )
    logger.info("Loading model from %s on %s", args.model_folder, device)

    unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder = load_model(
        args.model_folder, device, vae_device=vae_device
    )
    engine = InferenceEngine(
        unet,
        vae,
        noise_scheduler,
        action_embedding,
        tokenizer,
        text_encoder,
        device,
        vae_device=vae_device,
    )

    sys.stderr.write(READY_LINE)
    sys.stderr.flush()
    logger.info("Model ready; streaming with NOOP until stdin sends actions")

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    # --- Shared state: latest action byte, updated by reader thread ----------
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
                action = latest_action

            jpeg_bytes = engine.step_jpeg(action, quality=args.jpeg_quality)
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
