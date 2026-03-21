#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAM tile grid for Super Mario Bros (gym-super-mario-bros / nes-py).

Adapted from yumouwei/super-mario-bros-reinforcement-learning (MIT).
"""

from __future__ import annotations

import numpy as np


class smb_grid:
    def __init__(self, env):
        self.ram = env.unwrapped.ram
        self.screen_size_x = 16  # rendered screen size
        self.screen_size_y = 13

        self.mario_level_x = self.ram[0x6D] * 256 + self.ram[0x86]
        self.mario_x = self.ram[0x3AD]  # mario's position on the rendered screen
        self.mario_y = self.ram[0x3B8] + 16  # top edge of (big) mario

        self.x_start = self.mario_level_x - self.mario_x  # left edge pixel of the rendered screen in level
        self.rendered_screen = self.get_rendered_screen()

    def tile_loc_to_ram_address(self, x, y):
        """
        convert (x, y) in Current tile (32x13, stored as 16x26 in ram) to ram address
        x: 0 to 31
        y: 0 to 12
        """
        page = x // 16
        x_loc = x % 16
        y_loc = page * 13 + y

        address = 0x500 + x_loc + y_loc * 16

        return address

    def get_rendered_screen(self):
        """
        Get the rendered screen (16 x 13) from ram
        empty: 0
        tile: 1
        enemy: -1
        mario: 2
        """
        rendered_screen = np.zeros((self.screen_size_y, self.screen_size_x))
        screen_start = int(np.rint(self.x_start / 16))

        for i in range(self.screen_size_x):
            for j in range(self.screen_size_y):
                x_loc = (screen_start + i) % (self.screen_size_x * 2)
                y_loc = j
                address = self.tile_loc_to_ram_address(x_loc, y_loc)

                if self.ram[address] != 0:
                    rendered_screen[j, i] = 1

        x_loc = (self.mario_x + 8) // 16
        y_loc = (self.mario_y - 32) // 16
        if x_loc < 16 and y_loc < 13:
            rendered_screen[y_loc, x_loc] = 2

        for i in range(5):
            if self.ram[0xF + i] == 1:
                enemy_x = self.ram[0x6E + i] * 256 + self.ram[0x87 + i] - self.x_start
                enemy_y = self.ram[0xCF + i]
                x_loc = (enemy_x + 8) // 16
                y_loc = (enemy_y + 8 - 32) // 16

                if 0 <= x_loc < 16 and 0 <= y_loc < 13:
                    rendered_screen[y_loc, x_loc] = -1

        return rendered_screen
