# Remote ControlNet inference server

Same stdin/stdout protocol as `remote-server-gamengen/inference_loop.py`: one byte per tick (COMPLEX_MOVEMENT index 0–11), length-prefixed JPEG frames on stdout, `READY\n` on stderr.

Loads [`controlnet_based/inference_engine.py`](../controlnet_based/inference_engine.py) from the repo root.

## Layout

- `--model_folder` must contain `transformer.pt` and a `controlnet/` directory (ControlNet weights).
- Default `--initial-image` points at `../remote-server-gamengen/sample_images/start.jpg`.

## Setup (GPU host)

```bash
cd /path/to/KNN_proj/remote-server-controlnet
pipenv install
```

## Manual test

```bash
pipenv run python inference_loop.py \
  --model_folder /path/to/inference_checkpoints \
  --initial-image /path/to/start.jpg
```

## Client

From the dev machine, use `game-client` with `--mode controlnet` (see [game-client/README.md](../game-client/README.md)).
