# -*- coding: utf-8 -*-
"""
Filename: treasure_on_the_right.py
Function:
Author: zhangzhengzhang@sunlands.com
Create: 2018/2/26 15:11

"""
import numpy as np
import pandas as pd
import time

# np.random.seed(2)

N_STATES = 6
# N_STATES = 10
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 13
FRESH_TIME = 0.3

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )
    return table

def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    # exploration and exploitation trade-off
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R

def update_env(S, episode, step_counter):
    env_list = ['-']*(N_STATES-1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def random_policy():
    avg_step = 0
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = np.random.choice(ACTIONS)
            S_, _ = get_env_feedback(S, A)
            if S_ == 'terminal':
                is_terminated = True
            S = S_
            update_env(S, episode, step_counter +1)
            step_counter += 1
        avg_step += step_counter
    avg_step /= MAX_EPISODES * 1.0
    print('avg_step: %s' % avg_step)

def rl_q_learning():
    avg_step = 0
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.loc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_
            update_env(S, episode, step_counter + 1)
            step_counter += 1
        avg_step += step_counter
    avg_step /= MAX_EPISODES * 1.0
    print('avg_step: %s' % avg_step)
    return q_table

def rl_sarsa():
    avg_step = 0
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        A = choose_action(S, q_table)
        while not is_terminated:
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                A_ = choose_action(S_, q_table)
                q_target = R + GAMMA * q_table.loc[S_, A_]
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_
            A = A_
            update_env(S, episode, step_counter + 1)
            step_counter += 1
        avg_step += step_counter
    avg_step /= MAX_EPISODES * 1.0
    print('avg_step: %s' % avg_step)
    return q_table

def rl_sarsa_lambda():
    avg_step = 0
    q_table = build_q_table(N_STATES, ACTIONS)
    eligibility_trace = q_table.copy()
    for episode in range(MAX_EPISODES):
        eligibility_trace *= 0
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        A = choose_action(S, q_table)
        while not is_terminated:
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                A_ = choose_action(S_, q_table)
                q_target = R + GAMMA * q_table.loc[S_, A_]
            else:
                q_target = R
                is_terminated = True
            error = (q_target - q_predict)
            # method 1:
            eligibility_trace.loc[S, :] *= 0
            eligibility_trace.loc[S, A] = 1
            #
            # # method 2:
            # eligibility_trace.loc[S, A] = 1

            # method 3:
            # eligibility_trace.loc[S, A] += 1

            q_table += ALPHA * error * eligibility_trace
            eligibility_trace *= GAMMA * ALPHA
            # eligibility_trace *= GAMMA * 1
            # eligibility_trace *= GAMMA * 0
            S = S_
            A = A_
            update_env(S, episode, step_counter + 1)
            step_counter += 1
        avg_step += step_counter
    avg_step /= MAX_EPISODES * 1.0
    print('avg_step: %s' % avg_step)
    return q_table

if __name__ == '__main__':
    print("random policy:")
    random_policy()
    print("q-learning:")
    q_table_q_learning = rl_q_learning()
    print(q_table_q_learning)
    print("sarsa:")
    q_table_sarsa = rl_sarsa()
    print(q_table_sarsa)
    print("sara lambda:")
    q_table_sarsa_lambda = rl_sarsa_lambda()
    print(q_table_sarsa_lambda)
