# -*- coding: utf-8 -*-
"""
The Q-Learning algorithm goes as follows:

    1. Set the gamma parameter, and environment rewards in matrix R.

    2. Initialize matrix Q to zero.

    3. For each episode:

        Select a random initial state.

        Do While the goal state hasn't been reached.

            * Select one among all possible actions for the current state.
            * Using this possible action, consider going to the next state.
            * Get maximum Q value for this next state based on all possible actions.
            * Compute: Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
            * Set the next state as the current state.

        End Do

       End For

The algorithm above is used by the agent to learn from experience.  Each episode is equivalent to one training session.  In each training session, the agent explores the environment (represented by matrix R ), receives the reward (if any) until it reaches the goal state. The purpose of the training is to enhance the 'brain' of our agent, represented by matrix Q.  More training results in a more optimized matrix Q.  In this case, if the matrix Q has been enhanced, instead of exploring around, and going back and forth to the same rooms, the agent will find the fastest route to the goal state.

The Gamma parameter has a range of 0 to 1 (0 <= Gamma > 1).  If Gamma is closer to zero, the agent will tend to consider only immediate rewards.  If Gamma is closer to one, the agent will consider future rewards with greater weight, willing to delay the reward.

To use the matrix Q, the agent simply traces the sequence of states, from the initial state to goal state.  The algorithm finds the actions with the highest reward values recorded in matrix Q for current state:

Algorithm to utilize the Q matrix:

    1. Set current state = initial state.

    2. From current state, find the action with the highest Q value.

    3. Set current state = next state.

    4. Repeat Steps 2 and 3 until current state = goal state.

The algorithm above will return the sequence of states from the initial state to the goal state.

"""
import numpy as np
import pandas as pd
import time
import sys

np.random.seed(2)  # reproducible

N_STATES = 10   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
LAMBDA = 0.9    # discount factor
MAX_EPISODES = 20   # maximum episodes
FRESH_TIME = 0.2    # fresh time for one move


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    print("\nInitial state of Q-Table:\n")
    print(table)    # show this predefined q-table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    # act non-greedy or state-action have no value
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.argmax()
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-'] * (N_STATES - 1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = '\nEpisode %s: total_steps = %s \n' % (
            episode + 1, step_counter)
        sys.stdout.write(interaction)
        sys.stdout.flush()
        time.sleep(1)
    else:
        env_list[S] = 'o'
        interaction = "\r" + ''.join(env_list)
        sys.stdout.write(interaction)
        sys.stdout.flush()
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            # take action & get next state and reward
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.ix[S, A]
            if S_ != 'terminal':
                # next state is not terminal
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.ix[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state
            step_counter += 1
            update_env(S, episode, step_counter)

    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\nQ-table:\n')
    print(q_table)
