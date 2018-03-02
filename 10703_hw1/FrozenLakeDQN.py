# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import random
import math

def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """
    policy_stable = True
    policy = np.zeros(env.nS)
    for s in range(env.nS):
      all_value = []
      for action in range(env.nA):
        tem = 0
        for t in env.P[s][action]:
          tem += t[0] * (t[2] + gamma * value_function[t[1]])
        all_value.append(tem)

      policy[s] = all_value.index(max(all_value))

    return policy.astype(int)    



def evaluate_policy_sync(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    V = np.zeros(env.nS)
    delta = 100
    count = 0
    while delta > tol and count < max_iterations:
      count += 1
      delta = 0 
      new_V = np.zeros(V.shape)
      #new_V = V
      #print('count: ',count)
      for s in range(env.nS):
        pre_v = V[s]
        tem = 0;
        for t in env.P[s][policy[s]]:
          tem += t[0] * (t[2] + gamma * V[t[1]])
        new_V[s] = tem
        #print(tem,' ', end='')
        delta = max(delta, abs(pre_v - tem))
      #print ()
      V = new_V
    return V, count


def evaluate_policy_async_ordered(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy by asynchronous DP.  Updates states in
    their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    V = np.zeros(env.nS)
    delta = 100
    count = 0
    while delta > tol and count < max_iterations:
      count += 1
      delta = 0 
      for s in range(env.nS):
        pre_v = V[s]
        tem = 0;
        for t in env.P[s][policy[s]]:
          tem += t[0] * (t[2] + gamma * V[t[1]])
        V[s] = tem
        delta = max(delta, abs(pre_v - tem))


    return V, count


def evaluate_policy_async_randperm(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a policy.  Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    V = np.zeros(env.nS)
    delta = 100
    count = 0
    while delta > tol and count < max_iterations:
      count += 1
      delta = 0
      rand_perm = [i for i in range(env.nS)]
      random.shuffle(rand_perm) 
      for s in rand_perm:
        pre_v = V[s]
        tem = 0;
        for t in env.P[s][policy[s]]:
          tem += t[0] * (t[2] + gamma * V[t[1]])
        V[s] = tem
        delta = max(delta, abs(pre_v - tem))

    return V, count


def evaluate_policy_async_custom(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluate the value of a policy. Updates states by a student-defined
    heuristic. 

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    return np.zeros(env.nS), 0


def improve_policy(env, gamma, value_func, policy):
    """Performs policy improvement.
    
    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    policy_stable = True

    for s in range(env.nS):
      old_action = policy[s]
      all_value = []
      for action in range(env.nA):
        tem = 0
        for t in env.P[s][action]:
          tem += t[0] * (t[2] + gamma * value_func[t[1]])
        all_value.append(tem)

      policy[s] = all_value.index(max(all_value))
      if policy[s] != old_action:
        policy_stable = False

    return policy_stable, policy


def policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 85 of the Sutton & Barto Second Edition book.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.
    
    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    num_policy_improve = 0
    num_val_iteration = 0
    policy_stable = False
    while not policy_stable:
      value_func, eval_steps = evaluate_policy_sync(env, gamma, policy)
      num_val_iteration += eval_steps
      policy_stable, policy = improve_policy(env, gamma, value_func, policy)
      num_policy_improve += 1
      #print(num_val_iteration, '--', num_policy_improve)


    return policy, value_func, num_policy_improve, num_val_iteration


def policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    num_policy_improve = 0
    num_val_iteration = 0
    policy_stable = False
    while not policy_stable:
      value_func, eval_steps = evaluate_policy_async_ordered(env, gamma, policy)
      num_val_iteration += eval_steps
      policy_stable, policy = improve_policy(env, gamma, value_func, policy)
      num_policy_improve += 1
      #print(num_val_iteration, '--', num_policy_improve)


    return policy, value_func, num_policy_improve, num_val_iteration


def policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                    tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    num_policy_improve = 0
    num_val_iteration = 0
    policy_stable = False
    while not policy_stable:
      value_func, eval_steps = evaluate_policy_async_randperm(env, gamma, policy)
      num_val_iteration += eval_steps
      policy_stable, policy = improve_policy(env, gamma, value_func, policy)
      num_policy_improve += 1
      #print(num_val_iteration, '--', num_policy_improve)


    return policy, value_func, num_policy_improve, num_val_iteration


def policy_iteration_async_custom(env, gamma, max_iterations=int(1e3),
                                  tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_custom methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)

    return policy, value_func, 0, 0


def value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    V = np.zeros(env.nS)
    delta = 100
    count = 0

    while delta > tol and count < max_iterations:
      count += 1
      delta = 0 
      new_V = np.zeros(V.shape)
      for s in range(env.nS):
        all_value = []
        for action in range(env.nA):
          tem = 0;
          for t in env.P[s][action]:
            tem += t[0] * (t[2] + gamma * V[t[1]])
          all_value.append(tem)
        new_V[s] = max(all_value)

        delta = max(delta, abs(new_V[s] - V[s]))
      V = new_V

    return V, count


def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    V = np.zeros(env.nS)
    delta = 100
    count = 0

    while delta > tol and count < max_iterations:
      count += 1
      delta = 0 
      for s in range(env.nS):
        pre_v = V[s]
        all_value = []
        for action in range(env.nA):
          tem = 0;
          for t in env.P[s][action]:
            tem += t[0] * (t[2] + gamma * V[t[1]])
          all_value.append(tem)
        V[s] = max(all_value)

        delta = max(delta, abs(pre_v - V[s]))

    return V, count


def value_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    V = np.zeros(env.nS)
    delta = 100
    count = 0

    while delta > tol and count < max_iterations:
      count += 1
      delta = 0 
      rand_perm = [i for i in range(env.nS)]
      random.shuffle(rand_perm) 
      for s in rand_perm:
        pre_v = V[s]
        all_value = []
        for action in range(env.nA):
          tem = 0;
          for t in env.P[s][action]:
            tem += t[0] * (t[2] + gamma * V[t[1]])
          all_value.append(tem)
        V[s] = max(all_value)

        delta = max(delta, abs(pre_v - V[s]))

    return V, count


def value_iteration_async_custom(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    V = np.zeros(env.nS)
    size = int(math.sqrt(env.nS)) # get environment size
    goal_x = 0
    goal_y = 0
    if size == 4:
      goal_x = 0
      goal_y = 1
    elif size == 8:
      goal_x = 0
      goal_y = 7

    delta = 100
    count = 0
    state_update_count = 0



    while delta > tol and count < max_iterations:
      count += 1
      delta = 0
      for dist in range(count + 1):
        for s in range(env.nS):
          state_x = s%size
          state_y = int(s/size)
          if (abs(state_x - goal_x) + abs(state_y - goal_y)) == dist: #only update state in the range of move
            state_update_count += 1
            pre_v = V[s]
            all_value = []
            for action in range(env.nA):
              tem = 0;
              for t in env.P[s][action]:
                tem += t[0] * (t[2] + gamma * V[t[1]])
              all_value.append(tem)
            V[s] = max(all_value)

            delta = max(delta, abs(pre_v - V[s]))

    return V, state_update_count

