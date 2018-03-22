"""Temporal Difference solution for reinforcement learning. General update rule as follows:
    q[s, a] = q[s, a] + learning_rate * (td_target - q[s, a])
    td_target - q[s, a] is called TD Error
Unlike MC, which uses a full episode to update, TD use N-Step to update, especially, if we just use a
single step to update, it is call TD-0.
"""

import numpy as np
import itertools
from collections import namedtuple
from collections import defaultdict

# Definition
EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

def create_epsilon_greedy_policy(q_values, epsilon, num_actions):
    """e-greedy policy, explaration & exploitation
    Parameters:
        q_values: Q(s, a)
        epsilon: the probability to select a random action
        num_actions: number of total actions
    Return:
        a wrapper function, from state to action probabilities
    """
    def policy_fn(observation):
        actions = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(q_values[observation])
        actions[best_action] += (1.0 - epsilon)
        return actions

    return policy_fn


def sarsa(env, episodes, discount_factor=1.0, learning_rate=0.5, epsilon=0.1):
    """SARSA algorithm, on-policy td control
    td_target = R[t+1] + discount_factor * q[next_state, next_action]
    Parameters:
        env: OpenAI Gym-like environment
        episodes: number of episodes to play
        discount_factor: discount factor for future reward
        learning_rate: update learning rate
        epsilon: the probability to select a random action
    Returns:
        q_values: final Q function
        stats: EpisodeStats instance, indicating the improvement during training 
    """
    q_values = defaultdict(lambda: np.zeros(env.action_space.n))

    stats = EpisodeStats(
            episode_lengths=np.zeros(episodes),
            episode_rewards=np.zeros(episodes)
            )

    policy = create_epsilon_greedy_policy(q_values, epsilon, env.action_space.n)

    for index in range(episodes):
        state = env.reset()
        # here use e-greedy policy to produce the action
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        # take a step and update
        for t in itertools.count():
            # take a step
            next_state, reward, done, _ = env.step(action)
            # to find next_action, we also use the e-greedy policy
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            stats.episode_rewards[index] += reward
            stats.episode_lengths[index] += 1
            # update rule
            td_target = reward + discount_factor * q_values[next_state][next_action]
            td_error = td_target - q_values[state][action]
            q_values[state][action] += learning_rate * td_error

            if done:
                break

            state, action = next_state, next_action

    return q_values, stats


def q_learning(env, episodes, discount_factor=1.0, learning_rate=0.5, epsilon=0.1):
    """Q Learning algorithm, off-policy td control
    td_target = R[t+1] + discount_factor * max(Q[next_state]) 
    Parameters:
        env: OpenAI Gym-like environment
        episodes: number of episodes to play
        discount_factor: discount factor for future reward
        learning_rate: update learning rate
        epsilon: the probability to select a random action
    Returns:
        q_values: final Q function
        stats: EpisodeStats instance, indicating the improvement during training 
    """
    q_values = defaultdict(lambda: np.zeros(env.action_space.n))

    stats = EpisodeStats(
            episode_lengths=np.zeros(episodes),
            episode_rewards=np.zeros(episodes)
            )

    policy = create_epsilon_greedy_policy(q_values, epsilon, env.action_space.n)

    for index in range(episodes):
        state = env.reset()
        for t in itertools.count():
            # here we use e-greedy policy to produce the action
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            stats.episode_rewards[index] += reward
            stats.episode_lengths[index] += 1

            # update rule, notice this time we use a policy other than e-greedy, which is used to
            # choose an action to continue
            best_next_action = np.argmax(q_values[next_state])
            td_target = reward + discount_factor * q_values[next_state][best_next_action]
            td_error = td_target - q_values[state][action]
            q_values[state][action] += learning_rate * td_error

            if done:
                break
            state = next_state

    return q_values, stats
