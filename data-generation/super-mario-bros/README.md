# Super Mario Bros — data collection

Uses [gym-super-mario-bros](https://pypi.org/project/gym-super-mario-bros/) (Python 3.10+ with `numpy<2`).

## Setup

```bash
cd data-generation/super-mario-bros
pipenv install --python 3.12   # or 3.10 / 3.11
```

The `ppo` and `combined` agents need **Stable-Baselines3**, **PyTorch**, and **shimmy** (installed via Pipfile) to load the pretrained `.zip` alongside the usual `gym-super-mario-bros` stack.

## Collect data

```bash
pipenv run python collect_data.py \
  --worlds 1 2 \
  --stages 1 2 3 4 \
  --runs-per-combo 1 \
  --workers 4 \
  --output-dir ./collected_data
```

Outputs one folder per run under `collected_data/` (see `run.json` + JPEG frames). Default output is `./collected_data` next to this script.

### Agents

| Agent | Behavior |
|-------|----------|
| `random` | Uniform random action over the discrete action set. |
| `ppo` | Pretrained [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) PPO policy. It observes a **RAM-derived tile grid** (see `smb_utils.py` / `smb_ram_wrapper.py`); **saved frames are still full RGB** from the emulator, same as the other agents. |
| `combined` | Same RAM env and PPO checkpoint as `ppo`, but control alternates in **5–40 frame** stretches: at the start of each stretch, **PPO** vs the same **random** policy as `random` is chosen with **2:1** odds favoring PPO for that entire stretch. |

For `ppo` and `combined`, the default checkpoint is `models/pre-trained-1.zip` (RAM stack `n_stack=4`, `n_skip=4`). Override the path or stack settings if you use another SB3 `.zip` trained with the same observation wrapper:

```bash
pipenv run python collect_data.py \
  --agents ppo \
  --worlds 1 --stages 1 \
  --model-path ./models/pre-trained-1.zip \
  --n-stack 4 --n-skip 4 \
  --max-steps 5000
```

`run.json` for `ppo` and `combined` also includes `ppo_model_path`, `ppo_n_stack`, and `ppo_n_skip`. `combined` adds `combined_period_min_frames` / `combined_period_max_frames` (currently 5 and 40) and `combined_ppo_period_prob` (currently 2/3).

#### Attribution (PPO / RAM observation code)

The RAM grid (`smb_grid`), observation wrapper (`SMBRamWrapper`), and bundled checkpoint `models/pre-trained-1.zip` come from the MIT-licensed project **[super-mario-bros-reinforcement-learning](https://github.com/yumouwei/super-mario-bros-reinforcement-learning)** by yumouwei. We vendored and lightly adapted `smb_utils.py` and the wrapper into this repo for data collection; see the file headers for details.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--agents` | `random` `ppo` | Which agents: `random`, `ppo`, `combined` |
| `--worlds` | `1` … `8` | Worlds to play |
| `--stages` | `1` … `4` | Stages per world |
| `--runs-per-combo` | `1` | Runs per (agent, world, stage); each run gets a unique folder suffix (8-char hex, same as `run_id` in `run.json`) |
| `--max-steps` | `5000` | **Safety cap** on environment steps per episode. Stops the run early if reached (recorded as `outcome: max_steps` in `run.json`). Raise this if episodes are cut off too soon. |
| `--replay-seed` | *(none)* | Exact RNG/env seed (copy `seed` from a previous `run.json`). If omitted, a random 31-bit seed is generated, logged as `seed` with `seed_source: generated_at_run_start`, and you can replay with `--replay-seed <that value>`. |
| `--workers` | `1` | Parallel processes (`1` = sequential) |
| `--output-dir` | `./collected_data` | Where run folders are written |
| `--model-path` | `./models/pre-trained-1.zip` when `ppo` or `combined` is selected | SB3 PPO checkpoint (`.zip`). Required file must exist if `--agents` includes `ppo` or `combined` (defaults to the path next to this script). |
| `--n-stack` | `4` | RAM observation: number of stacked frames (must match the checkpoint). |
| `--n-skip` | `4` | RAM observation: frame stride between stacked slices (must match the checkpoint). |

`run.json` always includes `seed`, `seed_source` (`cli_replay` or `generated_at_run_start`), and `seed_note`.

Run `pipenv run python collect_data.py --help` for the full argparse text.

## Replay a run (GUI)

The replay viewer lives in a **separate** folder so it does not share the gym/collection stack and can use a Python build with Tk (see [`super-mario-bros-viewer/README.md`](../super-mario-bros-viewer/README.md)).

```bash
cd ../super-mario-bros-viewer
pipenv run python replay.py ../super-mario-bros/collected_data/random_w1s1_a1b2c3d4
```
