import pandas as pd
import numpy as np
import time
from maze.maze_env import Maze


EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 200


def build_q_table(n_states, n_actions):
    table = pd.DataFrame(
        np.zeros((n_states, n_actions)),
        # columns=actions,
    )
    return table

def random_policy():
    env = Maze()

    def choose_action():
        action_name = np.random.choice(env.n_actions)
        return action_name

    for episode in range(200):
        step_counter = 0
        done = False
        S = env.reset()
        env.render()
        while not done:
            A = choose_action()
            S_, R, done, info = env.step(A)
            env.render()
            step_counter += 1
            time.sleep(0.3)
        interaction = 'Episode %s: total_steps = %s, got = %s' % (episode + 1, step_counter, R)
        print(interaction)
        time.sleep(1)


def qlearning():
    env = Maze()
    s_dict = {}
    q_table = build_q_table(env.n_states, env.n_actions)

    def choose_action(state, q_table):
        state_actions = q_table.iloc[state, :]
        # exploration and exploitation trade-off
        if (np.random.uniform() > 0.9) or ((state_actions == 0).all()):
            action_name = np.random.choice(env.n_actions)
        else:
            action_name = state_actions.idxmax()
        return action_name

    def S2idx(S):
        S_key = str(S)
        if S_key in s_dict:
            S_idx = s_dict[S_key]
        else:
            S_idx = len(s_dict)
            s_dict[S_key] = S_idx
        return S_idx

    for episode in range(200):
        step_counter = 0
        done = False
        S = env.reset()
        env.render()
        while not done:
            A = choose_action(S2idx(S), q_table)
            S_, R, done, info = env.step(A)
            q_predict = q_table.loc[S2idx(S), A]
            if done:
                q_targe = R
            else:
                q_targe = R + 0.9 * q_table.loc[S2idx(S_), :].max()
            q_table.loc[S2idx(S), A] += 0.1 * (q_targe - q_predict)
            S = S_
            env.render()
            step_counter += 1
            time.sleep(0.3)
        interaction = 'Episode %s: total_steps = %s, got = %s' % (episode + 1, step_counter, R)
        print(interaction)
        time.sleep(1)


def sarsa():
    env = Maze()
    s_dict = {}
    q_table = build_q_table(env.n_states, env.n_actions)

    def choose_action(state, q_table):
        state_actions = q_table.iloc[state, :]
        # exploration and exploitation trade-off
        if (np.random.uniform() > 0.9) or ((state_actions == 0).all()):
            action_name = np.random.choice(env.n_actions)
        else:
            action_name = state_actions.idxmax()
        return action_name

    def S2idx(S):
        S_key = str(S)
        if S_key in s_dict:
            S_idx = s_dict[S_key]
        else:
            S_idx = len(s_dict)
            s_dict[S_key] = S_idx
        return S_idx

    for episode in range(200):
        step_counter = 0
        done = False
        S = env.reset()
        env.render()
        A_ = choose_action(S2idx(S), q_table)
        A = A_
        while not done:
           S_, R, done, info = env.step(A)
           q_predict = q_table.loc[S2idx(S), A]
           if done:
               q_targe = R
           else:
               A_ = choose_action(S2idx(S_), q_table)
               q_targe = R + 0.9 * q_table.loc[S2idx(S_), A_]
           q_table.loc[S2idx(S_), A] += 0.1 * (q_targe - q_predict)
           S = S_
           A = A_
           env.render()
           step_counter += 1
           time.sleep(0.3)
        interaction = 'Episode %s: total_steps = %s, got = %s' % (episode + 1, step_counter, R)
        print(interaction)
        time.sleep(1)


def qlearning2():
    env = Maze()
    s_dict = {}
    q_table = build_q_table(env.n_states, env.n_actions)

    def choose_action(state, q_table):
        state_actions = q_table.iloc[state, :]
        # exploration and exploitation trade-off
        if (np.random.uniform() > 0.9) or ((state_actions == 0).all()):
            action_name = np.random.choice(env.n_actions)
        else:
            action_name = state_actions.idxmax()
        return action_name

    def S2idx(S):
        S_key = str(S)
        if S_key in s_dict:
            S_idx = s_dict[S_key]
        else:
            S_idx = len(s_dict)
            s_dict[S_key] = S_idx
        return S_idx

    def check_hesitation(A, A_pre):
        action_space = env.action_space
        act_A = action_space[A]
        act_A_pre = action_space[A_pre]
        if act_A == 'u' and act_A_pre == 'd':
            return True
        if act_A == 'd' and act_A_pre == 'u':
            return True
        if act_A == 'l' and act_A_pre == 'r':
            return True
        if act_A == 'r' and act_A_pre == 'l':
            return True
        return False

    for episode in range(200):
        step_counter = 0
        done = False
        S = env.reset()
        env.render()
        A_pre = None
        while not done:
            A = choose_action(S2idx(S), q_table)
            S_, R, done, info = env.step(A)
            q_predict = q_table.loc[S2idx(S), A]
            if not A_pre:
                A_pre = A
            if check_hesitation(A, A_pre):
                R -= 0.5
            A_pre == A
            if done:
                q_targe = R
            else:
                q_targe = R + 0.9 * q_table.loc[S2idx(S_), :].max()
            q_table.loc[S2idx(S), A] += 0.1 * (q_targe - q_predict)
            S = S_
            env.render()
            step_counter += 1
            time.sleep(0.3)
        interaction = 'Episode %s: total_steps = %s, got = %s' % (episode + 1, step_counter, R)
        print(interaction)
        time.sleep(1)


def sarsa_lambda():
    env = Maze()
    s_dict = {}
    q_table = build_q_table(env.n_states, env.n_actions)

    def choose_action(state, q_table):
        state_actions = q_table.iloc[state, :]
        # exploration and exploitation trade-off
        if (np.random.uniform() > 0.9) or ((state_actions == 0).all()):
            action_name = np.random.choice(env.n_actions)
        else:
            action_name = state_actions.idxmax()
        return action_name

    def S2idx(S):
        S_key = str(S)
        if S_key in s_dict:
            S_idx = s_dict[S_key]
        else:
            S_idx = len(s_dict)
            s_dict[S_key] = S_idx
        return S_idx

    eligibility_trace = q_table.copy()
    for episode in range(MAX_EPISODES):
        eligibility_trace *= 0
        step_counter = 0
        done = False
        S = env.reset()
        env.render()
        A_ = choose_action(S2idx(S), q_table)
        A = A_
        while not done:
            S_, R, done, info = env.step(A)
            q_predict = q_table.loc[S2idx(S), A]
            if done:
                q_targe = R
            else:
                A_ = choose_action(S2idx(S_), q_table)
                q_targe = R + GAMMA * q_table.loc[S2idx(S_), A_]
            error = (q_targe - q_predict)
            eligibility_trace.loc[S2idx(S), :] *= 0
            eligibility_trace.loc[S2idx(S), A] = 1
            q_table += ALPHA * error * eligibility_trace
            eligibility_trace *= GAMMA * ALPHA
            S = S_
            A = A_
            env.render()
            step_counter += 1
            time.sleep(0.3)
        interaction = 'Episode %s: total_steps = %s, got = %s' % (episode + 1, step_counter, R)
        print(interaction)
        time.sleep(1)


# random_policy()
# qlearning()
# sarsa()

# qlearning2()
sarsa_lambda()