"""Monte Carlo solution for reinforcement learning. It has the following characteristics:
      1. Model free, don't need to have full knowledge of how the environment works;
      2. The game must have a terminal state, meaning an episode has limited steps;
      3. Update at the end of each episode, so it has high variance but low bias.
"""

import numpy as np
from collections import defaultdict

def mc_prediction(policy, env, episodes, discount_factor=1.0):
    """Calculate value function based on a given policy using samplings.
    Parameters:
        policy: observation to action probabilities mapping
        env: OpenAI Gym-like environment
        episodes: number of episodes to play
        discount_factor: discount factor for future reward
    Returns:
        values: a dict from state to value
    """
    returned_sum = defaultdict(float)
    returned_count = defaultdict(float)

    values = defaultdict(float)

    for index in range(1, episodes + 1):
        # start a new game
        episode = []
        state = env.reset()
        # at most 100 steps
        for step in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        unique_states_in_episode = set([tuple(item[0]) for item in episode])
        for state in unique_states_in_episode:
            # we use first visit strategy
            first_visit_index = next(i for i, item in enumerate(episode) if item[0] == state)
            expected_reward = sum([(discount_factor ** i) * item[2]  for i, item in enumerate(episode[first_visit_index:])])
            returned_sum[state] += expected_reward
            returned_count[state] += 1.0
            # update value, so we need not to iterate in the outer loop
            values[state] = returned_sum[state] / returned_count[state]

    return values


def create_epsilon_greedy_policy(q_values, epsilon, num_actions):
    """e-greedy policy, exploration & exploitation
    Parameters:
        q_values: {state: [action1_value, action2_value, ...], ...} 
        epsilon: the probability to select a random action
        num_actions: number of actions
    Returns:
        a wrapped policy function, from state to action probabilities
    """
    def policy_fn(observation):
        actions = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(q_values[observation])
        actions[best_action] += (1.0 - epsilon)
        return actions

    return policy_fn


def create_random_policy(num_actions):
    """Random policy
    Parameters:
        num_actions: number of actions
    Returns:
        a wrapped policy function, from state to action probabilities
    """
    actions = np.ones(num_actions, dtype=float) / num_actions
    def policy_fn(observation):
        return actions

    return policy_fn


def create_greedy_policy(q_values):
    """a total greedy policy
    Parameters:
        q_values: {state: [action1_value, action2_value, ...], ...}
    Returns:
        a wrapped policy function, from state to action probabilities
    """
    def policy_fn(observation):
        actions = np.zeros_like(q_values[observation], dtype=float)
        best_action = np.argmax(q_values[observation])
        actions[best_action] = 1.0
        return actions

    return policy_fn


def mc_control_importance_sampling(env, episodes, behavior_policy, discount_factor=1.0):
    """Monte Carlo off policy control using weighted importance sampling
    Paramters:
        env: OpenAI Gym-like environment
        episodes: number of episodes to play
        behavior_policy: the behavior policy when generating actions in episodes
        discount_factor: discount factor for future reward
    Returns:
        q_values: final policy
    """
    q_values = defaultdict(lambda: np.zeros(env.action_space.n))
    cumulative = defaultdict(lambda: np.zeros(env.action_space.n))
    target_policy = create_greedy_policy(q_values)

    for index in range(1, episodes + 1):
        episode = []
        state = env.reset()
        for steps in range(100):
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        expected_reward = 0.0
        weight = 1.0

        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            # update the total reward from step t
            expected_reward = discount_factor * expected_reward + reward
            cumulative[state][action] += weight
            q_values[state][action] += (weight / cumulative[state][action]) * (expected_reward - q_values[state][action])
            if action != np.argmax(target_policy(state)):
                break
            weight = weight * 1.0 / behavior_policy(state)[action]

    return q_values


def mc_control_epsilon_greedy(env, episodes, discount_factor=1.0, epsilon=0.1):
    """Monte Carlo Control using e-greedy policy
    Paramters:
        env: OpenAI Gym-like environment
        episodes: number of episodes to play
        discount_factor: discount factor for future reward
        epsilon: the probability to select a random action
    Returns:
        q_values: final policy
    """
    returned_sum = defaultdict(float)
    returned_count = defaultdict(float)

    q_values = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_greedy_policy(q_values, epsilon, env.action_space.n)

    for index in range(1, episodes + 1):
        episode = []
        state = env.reset()
        for step in range(100):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        unique_sa_in_episode = set([(tuple(item[0]), item[1]) for item in episode])
        for state, action in unique_sa_in_episode:
            sa_pair = (state, action)
            first_visit_index = next(i for i, item in enumerate(episode) if item[0] == state and item[1] == action)
            expected_reward = sum([(discount_factor ** i) * item[2] for i, item in enumerate(episode[first_visit_index:])])
            returned_sum[sa_pair] += expected_reward
            returned_count[sa_pair] += 1.0
            q_values[state][action] = returned_sum[sa_pair] / returned_count[sa_pair]

    return q_values
