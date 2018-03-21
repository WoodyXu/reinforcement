"""Dynamic Programming solution for reinforcement learning. There are two sub solutions:
    1. Policy Iteration: Policy Evaluation + Policy Improvement
    2. Value Iteration
The characteristic of DP solution is model-based, i.e. we perfectly know the environment.
At this environment, we assume that the observation and action space are both discrete, and the
policy is deterministic, not probablistic, i.e. just the q-value of best action equals one, others
equal zero.

This module introduces value iteration.
"""

import numpy as np

def value_iteration(env, discount_factor=1.0, threshold=1e-6):
    """Value Iteration is straightforward, we iterate best value evalution, and at last, we get the
    final policy.
    Parameters:
        env: OpenAI Gym-like environment
        discount_factor: the discount factor for future reward
        threshold: the threshold delta for stopping iteration
    Returns:
        values: final value function
        policy: final policy
        iter_count: total iteration count
    """
    values = np.zeros(env.nS)
    iter_count = 0

    while True:
        iter_count += 1
        delta = 0
        # iterate each state
        for state in range(env.nS):
            # iterate the q-value
            state_policy = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    # Bellman equation
                    state_policy[action] += prob * (reward + discount_factor * values[next_state])
            best_action_value = np.max(state_policy)
            delta = max(delta, np.abs(values[state] - best_action_value))
            values[state] = best_action_value
        if delta < threshold:
            break

    # at last, we output the final policy
    policy = np.zeros((env.nS, env.nA)) 
    for state in range(env.nS):
        state_policy = np.zeros(env.nA)
        for action in range(env.nA):
            for prob, next_state, reward, done in env.P[state][action]:
                # Bellman equation
                state_policy[action] += prob * (reward + discount_factor * values[next_state])
        best_action = np.argmax(state_policy)
        policy[state, best_action] = 1.0

    return values, policy, iter_count
