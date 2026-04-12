#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Observation wrapper: stacked RAM tile grids for SMB RL.

Adapted from yumouwei/super-mario-bros-reinforcement-learning (MIT).
"""

from __future__ import annotations

from typing import Any

import gym
import gym_super_mario_bros  # noqa: F401 — registers envs
import numpy as np
from gym import spaces
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from smb_utils import smb_grid


class SMBRamWrapper(gym.ObservationWrapper):
    """
    crop_dim: [x0, x1, y0, y1]
    obs shape = (height, width, n_stack), n_stack=0 is the most recent frame
    n_skip: e.g. n_stack=4, n_skip=4, use frames [0, 4, 8, 12] in the internal buffer
    """

    def __init__(
        self,
        env: gym.Env,
        crop_dim: list[int] | None = None,
        n_stack: int = 4,
        n_skip: int = 2,
    ):
        if crop_dim is None:
            crop_dim = [0, 16, 0, 13]
        super().__init__(env)
        self.crop_dim = crop_dim
        self.n_stack = n_stack
        self.n_skip = n_skip
        self.width = crop_dim[1] - crop_dim[0]
        self.height = crop_dim[3] - crop_dim[2]
        self.observation_space = spaces.Box(
            low=-1,
            high=2,
            shape=(self.height, self.width, self.n_stack),
            dtype=np.int64,
        )

        self.frame_stack = np.zeros((self.height, self.width, (self.n_stack - 1) * self.n_skip + 1))

    def observation(self, obs: Any) -> np.ndarray:
        del obs  # RGB from inner env; we rebuild from RAM
        grid = smb_grid(self.env)
        frame = grid.rendered_screen
        frame = self.crop_obs(frame)

        self.frame_stack[:, :, 1:] = self.frame_stack[:, :, :-1]
        self.frame_stack[:, :, 0] = frame
        return self.frame_stack[:, :, :: self.n_skip]

    def reset(self, **kwargs: Any) -> Any:
        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple):
            _obs_inner, info = ret
        else:
            info = {}

        self.frame_stack = np.zeros((self.height, self.width, (self.n_stack - 1) * self.n_skip + 1))
        grid = smb_grid(self.env)
        frame = grid.rendered_screen
        frame = self.crop_obs(frame)
        for i in range(self.frame_stack.shape[-1]):
            self.frame_stack[:, :, i] = frame
        out = self.frame_stack[:, :, :: self.n_skip]
        if kwargs.get("return_info", False):
            return out, info
        return out

    def crop_obs(self, im: np.ndarray) -> np.ndarray:
        [x0, x1, y0, y1] = self.crop_dim
        return im[y0:y1, x0:x1]


def make_ram_env(
    world: int,
    stage: int,
    crop_dim: list[int] | None = None,
    n_stack: int = 4,
    n_skip: int = 4,
) -> SMBRamWrapper:
    env_id = f"SuperMarioBros-{world}-{stage}-v0"
    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    return SMBRamWrapper(env, crop_dim, n_stack=n_stack, n_skip=n_skip)
