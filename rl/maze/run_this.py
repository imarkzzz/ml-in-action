import pandas as pd
import numpy as np
import time
from maze.maze_env import Maze

def build_q_table(n_states, n_actions):
    table = pd.DataFrame(
        np.zeros((n_states, n_actions)),
        # columns=actions,
    )
    return table

def random_policy():
    env = Maze()
    env.reset()
    env.render()
    done = False
    rewards = []
    while not done:
        action_name = np.random.choice(env.n_actions)
        state, reward, done, info = env.step(action_name)
        rewards.append(reward)
        env.render()
        time.sleep(0.3)
    env.mainloop()


def sarsa():
    q_table = build_q_table(36, 4)
    N_ACTIONS = 4
    s_dict = {}
    env = Maze()
    def choose_action(state, q_table):
        state_actions = q_table.iloc[state, :]
        # exploration and exploitation trade-off
        if (np.random.uniform() > 0.9) or ((state_actions == 0).all()):
            action_name = np.random.choice(N_ACTIONS)
        else:
            action_name = state_actions.idxmax()
        return action_name
    for episode in range(200):
        step_counter = 0
        done = False
        S = env.reset()
        env.render()
        S_key = str(S)
        if S_key in s_dict:
            S_idx = s_dict[S_key]
        else:
            S_idx = len(s_dict)
            s_dict[S_key] = S_idx
        env.render()
        A_ = choose_action(S_idx, q_table)
        A = A_
        while not done:
           S_, R, done, info = env.step(A)
           q_predict = q_table.loc[S_idx, A]
           S_key_ = str(S_)
           if S_key_ in s_dict:
               S_idx_ = s_dict[S_key_]
           else:
               S_idx_ = len(s_dict)
               s_dict[S_key_] = S_idx_
           if done:
               q_targe = R
           else:
               A_ = choose_action(S_idx_, q_table)
               q_targe = R + 0.9 * q_table.loc[S_idx_, A_]
           q_table.loc[S_idx, A] += 0.1 * (q_targe - q_predict)
           S = S_
           A = A_
           env.render()
           step_counter += 1
           time.sleep(0.3)
        interaction = 'Episode %s: total_steps = %s, got = %s' % (episode + 1, step_counter, R)
        print(interaction)
        time.sleep(1)

# random_policy()
sarsa()