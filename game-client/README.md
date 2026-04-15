# Game SSH client

Tkinter UI + keyboard controls; launches `inference_loop.py` on the GPU host over `ssh -T`.

Supports **GameNGen** (`--mode gamengen`) or **ControlNet** (`--mode controlnet`); `--mode` is required.

## Setup

```bash
pipenv install --deploy
```

Omit `--deploy` if you are refreshing the lock after editing `Pipfile`.

## Run

Defaults include SSH target, remote directory, and model folder (per mode). Minimal:

```bash
pipenv run python client.py --mode gamengen
```

**GameNGen** (explicit paths):

```bash
pipenv run python client.py \
  --mode gamengen \
  --ssh-target vpsuser@pro6000b.foukec.cz \
  --remote-dir /home/vpsuser/KNN_proj/remote-server-gamengen \
  --model-folder /mnt/pro6000/data/gameNgen-checkpoints/sd-full-dataset-250k-steps
```

**ControlNet** (remote `remote-server-controlnet/` + checkpoints under `controlnet_based/inference_checkpoints/`):

```bash
pipenv run python client.py --mode controlnet
```

Optional: `--initial-image` (remote path); if omitted, defaults to `../remote-server-gamengen/sample_images/start.jpg` relative to `--remote-dir`.

Use `--ssh-opts` for extra flags, e.g. `--ssh-opts '-p 2222'`.

On macOS, if Tkinter is missing, install a Python build with Tcl/Tk (see `data-generation/super-mario-bros-viewer/README.md`).
