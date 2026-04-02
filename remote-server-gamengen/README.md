# remote-server-gamengen (inference only)

PyTorch is pinned to **ROCm 6.4** wheels from PyTorch’s index. When you need CUDA, swap the `[[source]]` and `torch` / `torchvision` / triton entries in `Pipfile` for the matching CUDA index and run `pipenv lock`.

## Setup

```bash
pipenv install --deploy
```

Omit `--deploy` if you are refreshing the lock after editing `Pipfile`.

## Run

```bash
pipenv run python inference.py --model_folder /path/to/weights
```

## Lock file

`Pipfile.lock` is checked in so `pipenv install --deploy` resolves cleanly (including `pytorch-triton-rocm`). Regenerate after changing dependencies:

```bash
pipenv lock
```
