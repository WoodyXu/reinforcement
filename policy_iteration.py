"""Dynamic Programming solution for reinforcement learning. There are two sub solutions:
    1. Policy Iteration: Policy Evaluation + Policy Improvement
    2. Value Iteration
The characteristic of DP solution is model-based, i.e. we perfectly know the environment.
At this environment, we assume that the observation and action space are both discrete, and the
policy is deterministic, not probablistic, i.e. just the q-value of best action equals one, others
equal zero.

This module introduces policy iteration.
"""

import numpy as np

def policy_evaluate(policy, env, discount_factor=1.0, threshold=1e-6):
    """The first step of policy iteration, assuming we have a specific policy, we want to evaluate the
    value function of each state.
    Parameters:
        policy: [observation_num, action_num] shaped matrix representing the policy
        env: OpenAI Gym-like environment 
        discount_factor: discount factor for future reward
        threshold: the threshold delta for stopping evaluation
    Returns:
        the value function for each observation
    """
    values = np.zeros(env.nS)
    while True:
        delta = 0
        # iterate each observation
        for state in range(env.nS):
            current_value = 0
            # iterate each possible action
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    # Bellman equation
                    current_value += action_prob * prob * (reward + discount_factor * values[next_state])
            delta = max(delta, np.abs(current_value - values[state]))
            values[state] = current_value
        if delta < threshold:
            break

    return np.array(values)

def policy_improve(env, policy, values, discount_factor=1.0):
    """The second step of policy iteration, aussming we have a specific value function, we want to
    improve the policy.
    Paramters:
        env: OpenAI Gym-like environment
        policy: [observation_num, action_num] shaped matrix representing policy
        values: value function for each observation
        discount_factor: discount factor for future reward
    Returns:
        is_policy_stable: boolean to show update or not 
        updated_policy: the updated policy
    """
    is_policy_stable = True
    for state in range(env.nS):
        best_action = np.argmax(policy[state])
        # start to update the policy
        updated_action_values = np.zeros(env.nA)
        for action in range(env.nA):
            for prob, next_state, reward, done in env.P[state][action]:
                # Bellman equation
                updated_action_values[action] += prob * (reward + discount_factor * values[next_state])
        updated_best_action = np.argmax(updated_action_values)
        if best_action != updated_best_action:
            is_policy_stable = False
        policy[state] = np.eye(env.nA)[updated_best_action]

    return is_policy_stable, policy

def policy_iteration(env, discount_factor=1.0, threshold=1e-6):
    """The whole process of policy iteration, first do policy evaluation, then policy improvement.
    Parameters:
       env: OpenAI Gym-like environment
       discount_factor: discount factor for future reward
       threshold: the threshold delta for stopping evaluation
    Returns:
       values: final value function
       policy: final policy
       iter_count: total iteration count
    """
    policy = np.ones([env.nS, env.nA]) / env.nA
    values = np.zeros(env.nS)
    iter_count = 0

    while True:
        iter_count += 1
        # first step: policy evaluation
        values = policy_evaluate(policy, env, discount_factor, threshold)
        # second step: policy improvement
        is_policy_stable, policy = policy_improve(env, policy, values, discount_factor)
        if is_policy_stable:
            break

    return values, policy, iter_count
