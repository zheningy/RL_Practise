#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import gym
import deeprl_hw1.lake_envs as lake_env
import time
import rl
import matplotlib.pyplot as plt
import numpy as np


def run_random_policy(env):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    initial_state = env.reset()
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(
            env.action_space.sample())
        env.render()

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        time.sleep(1)

    return total_reward, num_steps

def run_policy(env, policy):
    gamma = 0.9
    initial_state = env.reset()
    # env.render()
    # time.sleep(1)

    total_reward = 0
    num_steps = 0
    discount = 1
    curt_state = initial_state
    while True:
        #print('current state: ', curt_state)
        
        curt_state, reward, is_terminal, debug_info = env.step(policy[curt_state])
        #env.render()

        total_reward += reward * discount
        num_steps += 1
        discount *= gamma

        if is_terminal:
            break

        #time.sleep(1)
    return total_reward, num_steps



def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nS, env.nA))


def print_model_info(env, state, action):
    transition_table_row = env.P[state][action]
    print(
        ('According to transition function, '
         'taking action %s(%d) in state %d leads to'
         ' %d possible outcomes') % (lake_env.action_names[action],
                                     action, state, len(transition_table_row)))
    for prob, nextstate, reward, is_terminal in transition_table_row:
        state_type = 'terminal' if is_terminal else 'non-terminal'
        print(
            '\tTransitioning to %s state %d with probability %f and reward %f'
            % (state_type, nextstate, prob, reward))


def main():
    # create the environment
    env_size = int(input('Input environment size(4/8): '))
    env_class = input('Deterministic(D) or Stochastic(S): ')
    env_list = ['Deterministic-4x4-FrozenLake-v0', 
                'Deterministic-8x8-FrozenLake-v0',
                'Stochastic-4x4-FrozenLake-v0',
                'Stochastic-8x8-FrozenLake-v0']
    unexist = False
    if (env_size == 4):
        if (env_class == 'D' or env_class == 'd'):
            env = gym.make(env_list[0])
        elif(env_class == 'S' or env_class == 's'):
            env = gym.make(env_list[2])
        else:
            unexist = True
    elif (env_size == 4):
        if (env_class == 'D' or env_class == 'd'):
            env = gym.make(env_list[1])
        elif(env_class == 'S' or env_class == 's'):
            env = gym.make(env_list[3])
        else:
            unexist = True
    else:
        unexist = True

    if unexist:
        print("Unexisted environment")
        return

    gamma = 0.9
    
    print_env_info(env)
    print()
    method = input('Input iteration method (name/0~6): ')
    method_list = [                
                "policy_sync",
                "policy_async_ordered",
                "policy_async_randperm",
                "value_sync",
                "value_async_ordered",
                "value_async_randperm",
                "value_async_custom" ]
    if int(method) >= 0 and int(method) <=6:
        method = method_list[int(method)]
    print('Applying %s method...\n' % method)
    start_time = time.time()
    policy_iter_func = {
                "policy_sync" : rl.policy_iteration_sync,
                "policy_async_ordered" : rl.policy_iteration_async_ordered,
                "policy_async_randperm": rl.policy_iteration_async_randperm
    }

    value_iter_func = {
                "value_sync" : rl.value_iteration_sync,
                "value_async_ordered" : rl.value_iteration_async_ordered,
                "value_async_randperm" : rl.value_iteration_async_randperm,
                "value_async_custom" : rl.value_iteration_async_custom
    }

    if method in policy_iter_func:
        policy, value_func,  num_policy_improve, num_val_iteration = policy_iter_func[method](env, gamma)
        print('policy improve %d times' % num_policy_improve)
    elif method in value_iter_func:
        value_func, num_val_iteration = value_iter_func[method](env, gamma)
        policy = rl.value_function_to_policy(env, gamma, value_func)
    else:
        print("Invalid method")
        return
    if method != "value_async_custom": 
        print('val iteration %d times' % (num_val_iteration))
    else: print('Individual state update %d times' % (num_val_iteration))
    print("---Spent %s micro seconds calculate value function/policy ---" % ((time.time() - start_time)*1000))

    print()


    # Output value function
    dirt = ['L', 'D', 'R', 'U']
    print("Optimal Policy:")
    for s in range(env.nS):
        print(dirt[policy[s]], end ='')
        if (s+1)%env_size == 0:
            print()
    print()
    # Output optimal policy
    print("value function:")
    for s in range(env.nS):
        print(value_func[s], end = ' ')
        if (s+1)%env_size == 0:
            print()

    print()
    length = np.rint(np.sqrt(env.nS)).astype(int)
    plt.imshow(np.reshape(value_func, (length, length)))
    plt.colorbar()
    #plt.savefig('S_4_v.png')
    plt.show()

    run_times = int(input('Input simulated times: '))

    total_reward = 0.0
    num_steps = 0 
    for i in range(run_times):
        reward, steps = run_policy(env, policy)
        total_reward += reward
        num_steps += steps

    print('Agent received average reward of: %f' % (total_reward/run_times))
    print('Agent took %d steps' % (num_steps/run_times))



if __name__ == '__main__':
    main()
