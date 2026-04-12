#!/usr/bin/env python3
"""Print basic action statistics for the collected Super Mario Bros dataset."""

import json
import os
import sys

DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "super-mario-bros",
    "collected_data",
)


def main():
    data_dir = os.path.abspath(DATA_DIR)
    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    total_frames = 0
    noop_frames = 0
    left_frames = 0
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
            if action is None:
                noop_frames += 1
            elif "left" in action:
                left_frames += 1

        runs_processed += 1

    if total_frames == 0:
        print("No frames found.")
        sys.exit(1)

    noop_pct = noop_frames / total_frames * 100
    left_pct = left_frames / total_frames * 100

    print(f"Runs processed:      {runs_processed}")
    print(f"Total frames:        {total_frames}")
    print(f"NOOP frames:         {noop_frames} ({noop_pct:.2f}%)")
    print(f"Left frames:         {left_frames} ({left_pct:.2f}%)")


if __name__ == "__main__":
    main()
