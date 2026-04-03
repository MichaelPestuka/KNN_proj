#!/usr/bin/env python3
"""
Remote GameNGen loop: read 1-byte action indices from stdin, write length-prefixed JPEG frames to stdout.

Designed to be launched by the local client over SSH, e.g.:
  ssh -T user@host 'cd /home/knn/KNN_proj/remote-server-gamengen && pipenv run python inference_loop.py --model_folder ~/weights'
"""

from __future__ import annotations

import argparse
import io
import logging
import struct
import sys

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
    logger.info("Loading model from %s on %s", args.model_folder, device)

    unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder = load_model(
        args.model_folder, device
    )
    engine = InferenceEngine(
        unet,
        vae,
        noise_scheduler,
        action_embedding,
        tokenizer,
        text_encoder,
        device,
    )

    sys.stderr.write(READY_LINE)
    sys.stderr.flush()
    logger.info("Model ready; waiting for actions on stdin")

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    try:
        while True:
            chunk = stdin.read(1)
            if not chunk:
                logger.info("EOF on stdin; exiting")
                break
            action_index = chunk[0]
            if action_index >= NUM_ACTIONS:
                logger.warning("Invalid action byte %s; using NOOP", action_index)
                action_index = 0
            pil_image = engine.step(action_index)
            buf = io.BytesIO()
            pil_image.save(buf, format="JPEG", quality=args.jpeg_quality)
            jpeg_bytes = buf.getvalue()
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
