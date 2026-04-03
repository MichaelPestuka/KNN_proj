# GameNGen SSH client

Tkinter UI + keyboard controls; launches `inference_loop.py` on the GPU host over `ssh -T`.

## Setup

```bash
pipenv install --deploy
```

Omit `--deploy` if you are refreshing the lock after editing `Pipfile`.

## Run

Only `--ssh-target` is required (defaults match the usual remote layout):

```bash
pipenv run python client.py --ssh-target user@gpu-server
```

Equivalent to:

```bash
pipenv run python client.py \
  --ssh-target user@gpu-server \
  --remote-dir /home/knn/KNN_proj/remote-server-gamengen \
  --model-folder ~/gamengenmario/good-prototype
```

Use `--ssh-opts` for extra flags, e.g. `--ssh-opts '-p 2222'`.

On macOS, if Tkinter is missing, install a Python build with Tcl/Tk (see `data-generation/super-mario-bros-viewer/README.md`).
