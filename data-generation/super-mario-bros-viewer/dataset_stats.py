#!/usr/bin/env python3
"""Print basic action statistics for the collected Super Mario Bros dataset."""

import argparse
import json
import os
import sys

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "super-mario-bros")

DEFAULT_DATA_DIR = "collected_data"


def is_noop(action):
    return action is None or action == ["NOOP"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=DEFAULT_DATA_DIR,
        help=f"Name of the data subdirectory under super-mario-bros/ (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(os.path.join(BASE_DIR, args.data_dir))
    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    total_frames = 0
    noop_frames = 0
    left_frames = 0
    down_frames = 0
    runs_processed = 0

    for entry in sorted(os.listdir(data_dir)):
        run_json = os.path.join(data_dir, entry, "run.json")
        if not os.path.isfile(run_json):
            continue

        with open(run_json) as f:
            run = json.load(f)

        for frame in run["frames"]:
            total_frames += 1
            action = frame["action"]
            if is_noop(action):
                noop_frames += 1
            elif "left" in action:
                left_frames += 1
            elif "down" in action:
                down_frames += 1

        runs_processed += 1

    if total_frames == 0:
        print("No frames found.")
        sys.exit(1)

    noop_pct = noop_frames / total_frames * 100
    left_pct = left_frames / total_frames * 100
    down_pct = down_frames / total_frames * 100

    print(f"Runs processed:      {runs_processed}")
    print(f"Total frames:        {total_frames}")
    print(f"NOOP frames:         {noop_frames} ({noop_pct:.2f}%)")
    print(f"Left frames:         {left_frames} ({left_pct:.2f}%)")
    print(f"Down frames:         {down_frames} ({down_pct:.2f}%)")


if __name__ == "__main__":
    main()
