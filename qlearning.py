import tensorflow as tf
import keras
import keras.layers as kl

import numpy as np
import random

from collections import namedtuple


Experience = namedtuple("Experience", ["cur_state", "cur_action", "next_state", "reward"])


class QLearning:
  def __init__(self, game):
    self.game = game

    self.lr = 0.0001
    self.gamma = 0.9

    self.policy_net = DeepQNetwork(state_size=self.game.state_size, action_size=self.game.action_size)
    self.target_net = DeepQNetwork(state_size=self.game.state_size, action_size=self.game.action_size)
    self.optimizer = tf.optimizers.Adam(self.lr)
    self.batch_size = 16
    self.target_update = 25

    self.experiences = []
    self.max_exps = 200

    self.n_updates = 0
    self.total_loss = 0
    self.total_reward = 0

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

  def update_target_weights(self):
    for policy_var, target_var in zip(self.policy_net.trainable_variables, self.target_net.trainable_variables):
      target_var.assign(policy_var.numpy())

  def train(self, dt):
    cur_state = self.game.get_state()
    # choose action
    if np.random.rand() < 0.05:
      cur_action = np.random.randint(0, self.game.action_size, size=None)
    else:
      cur_action = np.argmax(self.policy_net(np.atleast_2d(cur_state)))

    reward = self.game.make_action(cur_action, dt)
    next_state = self.game.get_state()

    # if episode finished -> car is dead -> punish the ai
    if self.game.is_episode_finished():
      reward = -100

    self.total_reward += reward

    if self.n_updates % 30 == 0:
      print("REWARD: ", self.total_reward / 30)
      self.total_reward = 0

    # add to experiences
    self.experiences.append(Experience(cur_state, cur_action, next_state, reward))
    if len(self.experiences) > self.max_exps * 2:
      self.experiences = self.experiences[self.max_exps:]

    if len(self.experiences) >= self.batch_size * 4:
      sample_exps = random.sample(self.experiences, self.batch_size)
      batch = Experience(*zip(*sample_exps))
      cur_states, cur_actions, next_states, rewards = [np.asarray(batch[i]) for i in range(len(batch))]

      targets = rewards + self.gamma * np.max(self.policy_net(np.atleast_2d(next_states)), axis=1)
      targets = tf.convert_to_tensor(targets, dtype="float32")

      # calculate loss
      with tf.GradientTape() as tape:
        predictions = tf.math.reduce_sum(
          self.policy_net(np.atleast_2d(cur_states).astype("float32")) * tf.one_hot(cur_actions, self.game.action_size),
          axis=1)
        loss = tf.math.reduce_mean(tf.square(targets - predictions))

      # update weights
      weights = self.policy_net.trainable_weights
      gradients = tape.gradient(loss, weights)
      self.optimizer.apply_gradients(zip(gradients, weights))

      self.n_updates += 1
      self.total_loss += loss.numpy()
      if self.n_updates % 100 == 0:
        print("AVG LOSS OVER PAST 100 STEPS AFTER %d updates: %f" % (self.n_updates, loss.numpy() / 100))
        self.total_loss = 0

      if self.n_updates % self.target_update == 0:
        self.update_target_weights()

  def test(self):
    next_action = np.argmax(self.target_net(np.atleast_2d(self.game.get_state())))
    self.game.make_action(next_action)
    print(next_action)


class DeepQNetwork(keras.Model):
  def __init__(self, state_size, action_size):
    super(DeepQNetwork, self).__init__()
    self.state_size = state_size
    self.action_size = action_size

    self.input_layer = kl.InputLayer(input_shape=(state_size,))
    self.dense1 = kl.Dense(units=16, activation="tanh")
    self.dense2 = kl.Dense(units=16, activation="tanh")
    self.output_layer = kl.Dense(units=action_size, activation=None)

  @tf.function
  def call(self, inputs, **kwargs):
    x = self.input_layer(inputs)
    x = self.dense1(x)
    x = self.dense2(x)
    output = self.output_layer(x)
    return output
