# -*- coding: utf-8 -*-

import numpy as np
import time
import sys
import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width

DOG_POSE = (0, 1)
HELL_POSES = [(2, 1), (1, 2), (1, 3)]
FOOD_POSE = (3, 3)


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        self.hells = []
        for x, y in HELL_POSES:
            hell_center = origin + np.array([UNIT * x, UNIT * y])
            hell = self.canvas.create_rectangle(
                hell_center[0] - 15, hell_center[1] - 15,
                hell_center[0] + 15, hell_center[1] + 15,
                fill='black')
            self.hells.append(hell)

        # create oval
        x, y = FOOD_POSE
        oval_center = origin + np.array([UNIT * x, UNIT * y])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        x, y = DOG_POSE
        rect_center = origin + np.array([UNIT * x, UNIT * y])
        self.rect = self.canvas.create_rectangle(
            rect_center[0] - 15, rect_center[1] - 15,
            rect_center[0] + 15, rect_center[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        x, y = DOG_POSE
        rect_center = origin + np.array([UNIT * x, UNIT * y])
        self.rect = self.canvas.create_rectangle(
            rect_center[0] - 15, rect_center[1] - 15,
            rect_center[0] + 15, rect_center[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        reward = 0
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
            else:
                reward = -1
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
            else:
                reward = -1
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
            else:
                reward = -1
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
            else:
                reward = -1

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 10
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(hell) for hell in self.hells]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            done = False
        info = ''
        return s_, reward, done, info

    def render(self):
        time.sleep(0.05)
        self.update()


if __name__ == '__main__':
    env = Maze()
    env.reset()
    env.render()
    env.mainloop()
