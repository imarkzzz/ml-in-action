# -*- coding: utf-8 -*-

import os
import time
import numpy as np

class Island(object):
    def __init__(self):
        self.states = [1, 2, 3, 4, 5, 6, 7, 8]
        self.terminal_states = {6, 7, 8}
        self.actions = ['n', 'e', 's', 'w']
        self.rewards ={
            (1, 's'): -1.0,
            (3, 's'): 1.0,
            (5, 's'): -1.0,
        }
        self.env_background = ["_"] * len(self.states)
        self.skeletons_is_there = [6, 8]
        self.treasure_is_there = 7
        self.trans = {
            (1, 's'): 6,
            (1, 'e'): 2,
            (2, 'w'): 1,
            (2, 'e'): 3,
            (3, 's'): 7,
            (3, 'w'): 2,
            (3, 'e'): 4,
            (4, 'w'): 3,
            (4, "e"): 5,
            (5, 's'): 8,
            (5, 'w'): 4,
        }
        self.info_dict = {
            6: 'You died!',
            7: 'You got treasure!',
            8: 'You died!',
        }

    def reset(self):
        self.mark_is_here = 1

    def render(self):
        env_to_show = self.env_background[:]
        def offset_replace(state, icon):
            env_to_show[state - 1] = icon
        for skeleton in self.skeletons_is_there:
            offset_replace(skeleton, 'X')
        offset_replace(self.treasure_is_there, 'T')
        offset_replace(self.mark_is_here, 'O')
        cmd = "cls"
        os.system(cmd)
        for i in range(5):
            print env_to_show[i],
        print ''
        for i in range(5, 7):
            print env_to_show[i], 'W',
        print env_to_show[7]



    def step(self, action):
        key = (self.mark_is_here, action)
        next_state = self.mark_is_here
        reward = 0
        info = ''
        done = False
        if key in self.trans:
            next_state = self.trans[key]
            self.mark_is_here = next_state
        else:
            info = 'You didn\'t move!'
        if key in self.rewards:
            reward = self.rewards[key]
        if next_state in self.terminal_states:
            done = True
        if next_state in self.info_dict:
            info = self.info_dict[next_state]
        return next_state, reward, done, info




def random_policy():
    island = Island()
    island.reset()
    island.render()
    done = False
    while not done:
        time.sleep(0.3)
        action_name = np.random.choice(island.actions)
        state, reward, done, info = island.step(action_name)
        island.render()
        print info
    # island.render()
# random_policy()

def sarsa():
    island = Island()
    ACTIONS = island.actions
    STATES = island.states
    import pandas as pd
    def build_q_table(n_states, actions):
        table = pd.DataFrame(
            np.zeros((n_states, len(actions))),
            columns=actions,
        )
        return table
    q_table = build_q_table( len(STATES) + 1, ACTIONS )

    def choose_action(state, q_table):
        state_actions = q_table.iloc[state, :]
        # exploration and exploitation trade-off
        if (np.random.uniform() > 0.9) or ((state_actions == 0).all()):
            action_name = np.random.choice(ACTIONS)
        else:
            action_name = state_actions.idxmax()
        return action_name
    for episode in range(20):
        step_counter = 0
        done = False
        island.reset()
        S = 1
        island.render()
        A = choose_action(S, q_table)
        while not done:
           S_, R, done, info = island.step(A)
           q_predict = q_table.loc[S, A]
           if done:
               q_targe = R
           else:
               A_ = choose_action(S_, q_table)
               q_targe = R + 0.9 * q_table.loc[S_, A_]
           q_table.loc[S, A] += 0.1 * (q_targe - q_predict)
           S = S_
           A = A_
           island.render()
           print info
           step_counter += 1
           time.sleep(0.3)
        interaction = 'Episode %s: total_steps = %s, got = %s' % (episode + 1, step_counter, R)
        print interaction
        time.sleep(1)

    return q_table


sarsa()
