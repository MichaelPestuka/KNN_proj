# remote-server-gamengen (inference only)

PyTorch is pinned to **ROCm 6.4** wheels from PyTorch’s index. When you need CUDA, swap the `[[source]]` and `torch` / `torchvision` / triton entries in `Pipfile` for the matching CUDA index and run `pipenv lock`.

## Setup

```bash
pipenv install --deploy
```

Omit `--deploy` if you are refreshing the lock after editing `Pipfile`.

## Run (one-shot validation)

```bash
pipenv run python inference.py --model_folder /path/to/weights
```

## Run (interactive loop over stdin/stdout)

Used by the local `game-client` over SSH. Reads one byte per frame (action index 0–6), writes a big-endian uint32 length plus a JPEG frame.

```bash
pipenv run python inference_loop.py --model_folder /path/to/weights
```

See [../game-client/client.py](../game-client/client.py) for the Tkinter client that launches this via `ssh -T`.

## Lock file

`Pipfile.lock` is checked in so `pipenv install --deploy` resolves cleanly (including `pytorch-triton-rocm`). Regenerate after changing dependencies:

```bash
pipenv lock
```
