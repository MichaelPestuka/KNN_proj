#!/usr/bin/env python3
"""
Collect Super Mario Bros gameplay frames and actions using gym-super-mario-bros.

Each run is stored under collected_data/<agent>_w<world>s<stage>_<counter>/ with
JPG frames and run.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import secrets
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

import gym_super_mario_bros  # noqa: F401 — registers envs
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


AGENTS = ("random", "rightmove")
DEFAULT_WORLDS = tuple(range(1, 9))
DEFAULT_STAGES = tuple(range(1, 5))
GAME_OVER_EXTRA_STEPS = 45


def _json_safe_info(info: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in info.items():
        if isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def make_env(world: int, stage: int) -> Any:
    env_id = f"SuperMarioBros-{world}-{stage}-v0"
    env = gym_super_mario_bros.make(env_id)
    return JoypadSpace(env, SIMPLE_MOVEMENT)


def get_base_env(env: Any) -> Any:
    return env.unwrapped


def capture_raw_screen(env: Any) -> np.ndarray:
    """RGB screen from underlying NESEnv (same shape as observation)."""
    base = get_base_env(env)
    screen = base.screen
    return np.array(screen, dtype=np.uint8)


def save_frame_jpg(path: Path, rgb: np.ndarray) -> None:
    """rgb: HxWx3 uint8."""
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(path, format="JPEG", quality=92)


def make_random_agent(rng: np.random.Generator) -> Callable[[int], int]:
    def act(n_actions: int) -> int:
        return int(rng.integers(0, n_actions))

    return act


def make_rightmove_agent(rng: np.random.Generator) -> Callable[[int], int]:
    """Bias toward moving right + jump / run."""

    def act(n_actions: int) -> int:
        preferred = (1, 2, 3, 4)  # right, right+A, right+B, right+A+B
        if rng.random() < 0.75:
            return int(rng.choice(preferred))
        return int(rng.integers(0, n_actions))

    return act


def get_agent_fn(name: str, rng: np.random.Generator) -> Callable[[int], int]:
    if name == "random":
        return make_random_agent(rng)
    if name == "rightmove":
        return make_rightmove_agent(rng)
    raise ValueError(f"Unknown agent: {name}")


def action_to_list(action_index: int | None) -> list[str] | None:
    if action_index is None:
        return None
    return list(SIMPLE_MOVEMENT[action_index])


@dataclass(frozen=True)
class RunTask:
    """One collection job. ``seed`` is ``--replay-seed`` when set; otherwise generated per run."""

    agent: str
    world: int
    stage: int
    run_index: int
    output_dir: str
    max_steps: int
    seed: int | None


def _run_folder_name(agent: str, world: int, stage: int, run_index: int) -> str:
    return f"{agent}_w{world}s{stage}_{run_index:03d}"


def _seed_note(seed_source: str) -> str:
    if seed_source == "cli_replay":
        return (
            "exact seed from --replay-seed; used for NumPy RNG, env.unwrapped.seed(), and action_space.seed()"
        )
    return (
        "cryptographically random seed (secrets.randbelow(2**31)); used for NumPy RNG, env.unwrapped.seed(), "
        "and action_space.seed() — replay with --replay-seed <seed> from this JSON"
    )


def run_single_episode(task: RunTask) -> dict[str, Any]:
    """Execute one episode; return summary dict (for logging)."""
    # Effective seed is always defined and logged so runs can be replayed (see --replay-seed).
    if task.seed is not None:
        effective_seed = task.seed
        seed_source = "cli_replay"
    else:
        effective_seed = secrets.randbelow(2**31)
        seed_source = "generated_at_run_start"

    rng = np.random.default_rng(effective_seed)

    agent_fn = get_agent_fn(task.agent, rng)
    n_actions = len(SIMPLE_MOVEMENT)

    run_uuid = uuid.uuid4().hex[:8]
    folder_name = _run_folder_name(task.agent, task.world, task.stage, task.run_index)
    run_dir = Path(task.output_dir) / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    hostname = socket.gethostname()
    plat = platform.platform()
    py_ver = platform.python_version()

    env = make_env(task.world, task.stage)
    # JoypadSpace.reset does not forward seed=; seed the base NESEnv + action space.
    get_base_env(env).seed(effective_seed)
    env.action_space.seed(effective_seed)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    frames_rows: list[dict[str, Any]] = []
    total_reward = 0.0
    frame_idx = 0
    outcome = "max_steps"
    flag_get = False

    def filename_for(i: int) -> str:
        return f"frame_{run_uuid}_{i:06d}.jpg"

    while frame_idx < task.max_steps:
        action_index = agent_fn(n_actions)
        fname = filename_for(frame_idx)
        save_frame_jpg(run_dir / fname, obs)

        next_obs, reward, done, info = env.step(action_index)
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]

        total_reward += float(reward)
        info_safe = _json_safe_info(dict(info))
        frames_rows.append(
            {
                "frame": frame_idx,
                "filename": fname,
                "action_index": action_index,
                "action": action_to_list(action_index),
                "reward": float(reward),
                "done": bool(done),
                "info": info_safe,
            }
        )
        frame_idx += 1

        if done:
            flag_get = bool(info.get("flag_get", False))
            outcome = "flag" if flag_get else "death"

            # Terminal gameplay frame (death / pole)
            term_fname = filename_for(frame_idx)
            save_frame_jpg(run_dir / term_fname, next_obs)
            term_info = _json_safe_info(dict(info))
            frames_rows.append(
                {
                    "frame": frame_idx,
                    "filename": term_fname,
                    "action_index": None,
                    "action": None,
                    "reward": 0.0,
                    "done": True,
                    "info": term_info,
                    "note": "terminal_observation_after_done",
                }
            )
            frame_idx += 1

            # GAME OVER: nes-py marks env done so gym cannot step(); advance raw emulator.
            if not flag_get:
                base = get_base_env(env)
                noop_byte = int(env._action_map[0])
                for g in range(GAME_OVER_EXTRA_STEPS):
                    base._frame_advance(noop_byte)
                    raw = capture_raw_screen(env)
                    go_fname = filename_for(frame_idx)
                    save_frame_jpg(run_dir / go_fname, raw)
                    frames_rows.append(
                        {
                            "frame": frame_idx,
                            "filename": go_fname,
                            "action_index": 0,
                            "action": action_to_list(0),
                            "reward": 0.0,
                            "done": True,
                            "info": {"game_over_sequence": True, "sequence_index": g},
                            "note": "game_over_sequence",
                        }
                    )
                    frame_idx += 1
            break

        obs = next_obs

    env.close()

    run_json = {
        "agent": task.agent,
        "environment": f"SuperMarioBros-{task.world}-{task.stage}-v0",
        "world": task.world,
        "stage": task.stage,
        "run_id": run_uuid,
        "timestamp": timestamp,
        "hostname": hostname,
        "platform": plat,
        "python_version": py_ver,
        "total_frames": len(frames_rows),
        "outcome": outcome,
        "total_reward": total_reward,
        "max_steps_cap": task.max_steps,
        "seed": effective_seed,
        "seed_source": seed_source,
        "seed_note": _seed_note(seed_source),
        "frames": frames_rows,
    }

    with open(run_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(run_json, f, indent=2)

    return {"folder": str(run_dir), "outcome": outcome, "frames": len(frames_rows)}


def _worker_star(args: tuple[RunTask,]) -> dict[str, Any]:
    (task,) = args
    return run_single_episode(task)


def build_tasks(
    agents: Iterable[str],
    worlds: Iterable[int],
    stages: Iterable[int],
    runs_per_combo: int,
    output_dir: Path,
    max_steps: int,
    replay_seed: int | None,
) -> list[RunTask]:
    tasks: list[RunTask] = []
    for agent in agents:
        for w in worlds:
            for s in stages:
                for r in range(1, runs_per_combo + 1):
                    tasks.append(
                        RunTask(
                            agent=agent,
                            world=w,
                            stage=s,
                            run_index=r,
                            output_dir=str(output_dir),
                            max_steps=max_steps,
                            seed=replay_seed,
                        )
                    )
    return tasks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Mario frames + actions (gym-super-mario-bros)")
    p.add_argument(
        "--agents",
        nargs="+",
        choices=list(AGENTS),
        default=list(AGENTS),
        help="Agents to run",
    )
    p.add_argument(
        "--worlds",
        nargs="+",
        type=int,
        default=list(DEFAULT_WORLDS),
        help="World numbers 1-8",
    )
    p.add_argument(
        "--stages",
        nargs="+",
        type=int,
        default=list(DEFAULT_STAGES),
        help="Stage numbers 1-4",
    )
    p.add_argument(
        "--runs-per-combo",
        type=int,
        default=1,
        help="Number of runs per (agent, world, stage)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "collected_data",
        help="Output directory (default: ./collected_data next to this script)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes (default 1)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=50000,
        help="Safety cap on steps per episode",
    )
    p.add_argument(
        "--replay-seed",
        type=int,
        default=None,
        help="Exact RNG/env seed (copy from run.json 'seed'); omit to generate a random seed and log it",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(
        agents=args.agents,
        worlds=args.worlds,
        stages=args.stages,
        runs_per_combo=args.runs_per_combo,
        output_dir=output_dir,
        max_steps=args.max_steps,
        replay_seed=args.replay_seed,
    )

    if not tasks:
        print("No tasks to run.")
        return

    if args.workers <= 1:
        for t in tasks:
            summary = run_single_episode(t)
            print(summary)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            results = pool.map(_worker_star, [(t,) for t in tasks])
        for r in results:
            print(r)


if __name__ == "__main__":
    main()
