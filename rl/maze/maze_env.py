# -*- coding: utf-8 -*-

import numpy as np
import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width

DOG_POSE = (0, 0)
HELL_POSES = [
    (2, 1),
    # (1, 2),
    # (1, 3)
]
FOOD_POSE = (3, 3)

MAP_CONFIG = {
    "maze_h": MAZE_H,
    "maze_w": MAZE_W,
    "dog_pos": DOG_POSE,
    "hell_poses": HELL_POSES,
    "food_pos": FOOD_POSE
}


class Maze(tk.Tk, object):
    def __init__(self, map_config=None):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.map_config = self.load_map(map_config)
        self.n_spaces = self.maze_h * self.maze_w
        self.title('maze')
        # self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))
        self._build_maze()

    def load_map(self, map_config=None):
        self.map_states = MAP_CONFIG.copy()
        if map_config:
            self.map_states.update(map_config)
        self.maze_h = self.map_states["maze_h"]
        self.maze_w = self.map_states["maze_w"]

        self.dog_pos = np.array(self.map_states["dog_pos"])
        self.map_states["dog_pos"] = self.dog_pos

        self.hell_poses = [np.array(pos) for pos in self.map_states["hell_poses"]]
        self.map_states["hell_poses"] = self.hell_poses

        self.food_pos = np.array(self.map_states["food_pos"])
        self.map_states["food_pos"] =  self.food_pos
        return map_config

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=self.maze_h * UNIT,
                           width=self.maze_w * UNIT)

        # create grids
        for c in range(0, self.maze_w * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, self.maze_h * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.maze_h * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, self.maze_w * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([UNIT, UNIT]) / 2
        self.origin = origin

        # hell
        self.hells = []
        for x, y in self.hell_poses:
            hell_center = origin + np.array([UNIT * x, UNIT * y])
            hell = self.canvas.create_rectangle(
                hell_center[0] - 15, hell_center[1] - 15,
                hell_center[0] + 15, hell_center[1] + 15,
                fill='black')
            self.hells.append(hell)

        # create oval
        x, y = self.food_pos
        oval_center = origin + np.array([UNIT * x, UNIT * y])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self._make_dog()

        # pack all
        self.canvas.pack()

    def _make_dog(self):
        origin = self.origin
        x, y = self.dog_pos
        rect_center = origin + np.array([UNIT * x, UNIT * y])
        self.rect = self.canvas.create_rectangle(
            rect_center[0] - 15, rect_center[1] - 15,
            rect_center[0] + 15, rect_center[1] + 15,
            fill='red')

    def _move_dog(self, move_step):
        move_size = move_step * UNIT
        self.canvas.move(self.rect, move_size[0], move_size[1]) # move agent
        self.dog_pos += move_step

    def reset(self):
        self.update()
        self.load_map(self.map_config)
        self.canvas.delete(self.rect)
        self._make_dog()
        # return observation
        return self.map_states

    def step(self, action):
        s = self.canvas.coords(self.rect)
        move_step = np.array([0, 0])
        reward = 0
        if action == 0:   # up
            if s[1] > UNIT:
                move_step[1] -= 1
            else:
                reward = -1
        elif action == 1:   # down
            if s[1] < (self.maze_h - 1) * UNIT:
                move_step[1] += 1
            else:
                reward = -1
        elif action == 2:   # right
            if s[0] < (self.maze_w - 1) * UNIT:
                move_step[0] += 1
            else:
                reward = -1
        elif action == 3:   # left
            if s[0] > UNIT:
                move_step[0] -= 1
            else:
                reward = -1

        self._move_dog(move_step) # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        info = ''
        if s_ == self.canvas.coords(self.oval):
            reward = 10
            done = True
            info = "Win"
        elif s_ in [self.canvas.coords(hell) for hell in self.hells]:
            reward = -1
            done = True
            info = "Lose"
        else:
            done = False
        s_ = self.map_states
        return s_, reward, done, info

    def render(self):
        self.update()


if __name__ == '__main__':
    map_cfg = {
        "maze_h": 1,
        "maze_w": 5,
        "dog_pos": (0, 0),
        "hell_poses": [],
        "food_pos": (4, 0)
    }
    env = Maze(map_cfg)
    env.reset()
    env.render()
    env.mainloop()
