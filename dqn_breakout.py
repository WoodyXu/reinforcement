"""
There are two tricks to make deep q learning training better and more stable:
    1. Experience Replay
    2. Off-policy and set a target network. The target network is used to estimate the td target.
There are also several variants, such as Double-Q Learning, Prioritized Experience Replay and so on.
"""
import gym
import itertools
import numpy as np
import sys
import os
import random
import tensorflow as tf
from collections import deque, namedtuple
import keras
import keras.backend as K
from keras.layers import Conv2D, Dense, Flatten, Input, BatchNormalization
from keras.optimizers import SGD
from keras.models import Model

sys.path.append("../")

from lib import plotting

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


class StateProcessor(object):
    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        return sess.run(self.output, {self.input_state: state})


class Estimator(object):
    def __init__(self, scope="estimator", summaries_dir=None):
        self.step = 0
        self.scope = scope
        self.summary_writer = None
        with tf.variable_scope(scope):
            self._build_model()
            if summaries_dir is not None:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        self.x = Input(shape=(84, 84, 4), dtype="float32", name="input")
        self.loss = tf.placeholder(dtype=tf.float32)

        conv1 = Conv2D(filters=32, kernel_size=8, strides=4, padding="same", activation="relu", name="conv_1")(self.x)
        conv1_bn = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=64, kernel_size=4, strides=2, padding="same", activation="relu", name="conv_2")(conv1_bn)
        conv2_bn = BatchNormalization()(conv2)
        conv3 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu", name="conv_3")(conv2_bn)

        flattened = Flatten(name="flatten_1")(conv3)
        fc1 = Dense(units=512, name="fc_1")(flattened)
        fc1_bn = BatchNormalization()(fc1)
        self.predictions = Dense(units=4, name="output")(fc1_bn)

        self.model = Model([self.x], [self.predictions])
        print "\n-----The model description is as follows:-----\n"
        self.model.summary()
        opt = SGD(lr=0.001)
        self.model.compile(optimizer=opt, loss="mean_squared_error")
        print "\n-----The End-----\n"

        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss)
            ]) 

    def predict(self, state):
        state = state.astype(np.float32) / 255.0
        return self.model.predict(state)

    def update(self, state, target):
        state = state.astype(np.float32) / 255.0
        self.step += 1
        loss = self.model.train_on_batch(state, target)
        feed_dict = {self.loss: loss}
        summaries = K.get_session().run(self.summaries, feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, self.step)
        return loss


def copy_model_parameters(estimator1, estimator2):
    """copy from estimator1 to estimator2
    """
    estimator2.model.set_weights(estimator1.model.get_weights())


def create_epsilon_greedy_policy(estimator, num_actions):
    def policy_fn(observation, epsilon):
        actions = np.ones(num_actions) * epsilon / num_actions
        best_action_idx = np.argmax(estimator.predict(np.expand_dims(observation, axis=0))[0])
        actions[best_action_idx] += (1.0 - epsilon)
        return actions

    return policy_fn

def deep_q_learning(env, q_estimator, target_estimator, episodes, state_processor, experiment_dir,
                    replay_memory_size=500000, replay_memory_init_size=50000,
                    update_target_estimator_every=10000, discount_factor=0.99, epsilon_start=1.0,
                    epsilon_end=0.1, epsilon_decay_steps=500000, batch_size=32, record_video_every=50):
    transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    replay_memory = deque()
    stats = EpisodeStats(
            episode_lengths=np.zeros(episodes),
            episode_rewards=np.zeros(episodes)
            )

    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy = create_epsilon_greedy_policy(q_estimator, 4)

    print "populate the replay memory"
    state = env.reset()
    state = state_processor.process(K.get_session(), state)
    state = np.stack([state] * 4, axis=2)
    for index in range(replay_memory_init_size):
        actions_prob = policy(state, epsilons[min(q_estimator.step, epsilon_decay_steps - 1)])
        action = np.random.choice(np.arange(len(actions_prob)), p=actions_prob)
        next_state, reward, done, _ = env.step(action)
        next_state = state_processor.process(K.get_session(), next_state)
        next_state = np.append(state[:,:, 1:], np.expand_dims(next_state, 2), axis=2)
        replay_memory.append(transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
            state = state_processor.process(K.get_session(), state)
            state = np.stack([state] * 4, axis=2)
        else:
            state = next_state
    print "finish initialize the replay memory"

    for index in range(episodes):
        state = env.reset()
        state = state_processor.process(K.get_session(), state)
        state = np.stack([state] * 4, axis=2)
        loss = None

        for step in itertools.count():
            epsilon = epsilons[min(q_estimator.step, epsilon_decay_steps - 1)]
            if q_estimator.step % update_target_estimator_every == 0:
                copy_model_parameters(q_estimator, target_estimator)

            print "Episode {}/{}, Step {}, Update {}, loss {}".format(index + 1, episodes, step + 1,
                    q_estimator.step + 1, loss)

            actions_prob = policy(state, epsilon)
            action = np.random.choice(np.arange(len(actions_prob)), p=actions_prob)
            next_state, reward, done, _ = env.step(action)
            next_state = state_processor.process(K.get_session(), next_state)
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)
            replay_memory.append(transition(state, action, reward, next_state, done))
            stats.episode_lengths[index] += 1
            stats.episode_rewards[index] += reward

            samples = random.sample(replay_memory, batch_size)

            states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = map(np.array, zip(*samples))

            # construct the targets, we only focus on the q value of the action we choose
            targets_batch = q_estimator.predict(states_batch)
            q_values_next = target_estimator.predict(next_states_batch)
            targets_batch[np.arange(len(actions_batch)), np.array(actions_batch).reshape(-1)] = \
                    np.array(rewards_batch).reshape(-1) + np.invert(dones_batch).astype(np.float32) * \
                    np.amax(q_values_next, axis=1).reshape(-1)

            loss = q_estimator.update(states_batch, targets_batch)

            if done:
                break
            state = next_state

    return stats

if __name__ == "__main__":
    env = gym.envs.make("Breakout-v0")
    experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))
    q_estimator = Estimator(scope="q_estimator", summaries_dir=experiment_dir)
    target_estimator = Estimator(scope="target_estimator")
    state_processor = StateProcessor()
    stats = deep_q_learning(env, q_estimator, target_estimator, 100, state_processor,
            experiment_dir)
    plotting.plot_episode_stats(stats)
    env.close()
