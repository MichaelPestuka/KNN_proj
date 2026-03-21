# Super Mario Bros — data collection

Uses [gym-super-mario-bros](https://pypi.org/project/gym-super-mario-bros/) (Python 3.10+ with `numpy<2`).

## Setup

```bash
cd data-generation/super-mario-bros
pipenv install --python 3.12   # or 3.10 / 3.11
```

## Collect data

```bash
pipenv run python collect_data.py \
  --agents random rightmove \
  --worlds 1 2 \
  --stages 1 2 3 4 \
  --runs-per-combo 1 \
  --max-steps 50000 \
  --workers 4 \
  --output-dir ./collected_data
```

Outputs one folder per run under `collected_data/` (see `run.json` + JPEG frames). Default output is `./collected_data` next to this script.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--agents` | `random` `rightmove` | Which agents (`random`, `rightmove`) |
| `--worlds` | `1` … `8` | Worlds to play |
| `--stages` | `1` … `4` | Stages per world |
| `--runs-per-combo` | `1` | Runs per (agent, world, stage); folder counter `001`, `002`, … |
| `--max-steps` | `50000` | **Safety cap** on environment steps per episode. Stops the run early if reached (recorded as `outcome: max_steps` in `run.json`). Raise this if episodes are cut off too soon. |
| `--replay-seed` | *(none)* | Exact RNG/env seed (copy `seed` from a previous `run.json`). If omitted, a random 31-bit seed is generated, logged as `seed` with `seed_source: generated_at_run_start`, and you can replay with `--replay-seed <that value>`. |
| `--workers` | `1` | Parallel processes (`1` = sequential) |
| `--output-dir` | `./collected_data` | Where run folders are written |

`run.json` always includes `seed`, `seed_source` (`cli_replay` or `generated_at_run_start`), and `seed_note`.

Run `pipenv run python collect_data.py --help` for the full argparse text.

## Replay a run (GUI)

The replay viewer lives in a **separate** folder so it does not share the gym/collection stack and can use a Python build with Tk (see [`super-mario-bros-viewer/README.md`](../super-mario-bros-viewer/README.md)).

```bash
cd ../super-mario-bros-viewer
pipenv run python replay.py ../super-mario-bros/collected_data/random_w1s1_001
```
