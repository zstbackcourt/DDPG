# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
import numpy as np
import tensorflow as tf
from baselines.a2c.utils import fc
import tf_utils


N_H1 = 400
N_H2 = 300

class ActorNetwork(object):

    def __init__(self, state_size, action_size, lr, tau=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.tau = tau
        self.n_h1 = N_H1
        self.n_h2 = N_H2
        self.input_s, self.actor_variables, self.action_values = self._build_network("actor")
        self.input_s_target, self.actor_variables_target, self.action_values_target = self._build_network("actor_target")
        self.action_gradients = tf.placeholder(tf.float32, [None, self.action_size])
        self.actor_gradients = tf.gradients(self.action_values, self.actor_variables, -self.action_gradients)
        self.update_target_op = [self.actor_variables_target[i].assign(tf.multiply(self.actor_variables[i], self.tau) +
                                                                       tf.multiply(self.actor_variables_target[i], 1 - self.tau))
                                 for i in range(len(self.actor_variables))]
        self.optimize = self.optimizer.apply_gradients(zip(self.actor_gradients, self.actor_variables))

    def _build_network(self, name):
        input_s = tf.placeholder(tf.float32, [None, self.state_size])
        with tf.variable_scope(name):
            layer_1 = tf_utils.fc(input_s, self.n_h1, scope="fc1", activation_fn=tf.nn.relu,
                                  initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            layer_2 = tf_utils.fc(layer_1, self.n_h2, scope="fc2", activation_fn=tf.nn.relu,
                                  initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            action_values = tf_utils.fc(layer_2, self.action_size, scope="out", activation_fn=tf.nn.tanh,
                                        initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
        actor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return input_s, actor_variables, action_values

    def get_action(self, state, sess):
        return sess.run(self.action_values, feed_dict={self.input_s: state})

    def get_action_target(self, state, sess):
        return sess.run(self.action_values_target, feed_dict={self.input_s_target: state})

    def train(self, state, action_gradients, sess):
        sess.run(self.optimize, feed_dict={
            self.input_s: state,
            self.action_gradients: action_gradients
        })

    def update_target(self, sess):
        sess.run(self.update_target_op)

class CriticNetwork(object):

    def __init__(self, state_size, action_size, lr, tau=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.tau = tau

        self.n_h1 = N_H1
        self.n_h2 = N_H2

        self.input_s, self.action, self.critic_variables, self.q_value = self._build_network("critic")
        self.input_s_target, self.action_target, self.critic_variables_target, self.q_value_target = self._build_network(
            "critic_target")

        self.target = tf.placeholder(tf.float32, [None])
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.critic_variables])
        self.loss = tf.reduce_mean(tf.square(self.target - self.q_value)) + 0.01 * self.l2_loss
        self.optimize = self.optimizer.minimize(self.loss)
        self.update_target_op = [self.critic_variables_target[i].assign(
            tf.multiply(self.critic_variables[i], self.tau) + tf.multiply(self.critic_variables_target[i],
                                                                          1 - self.tau)) for i in
                                 range(len(self.critic_variables))]
        self.action_gradients = tf.gradients(self.q_value, self.action)

    def _build_network(self, name):
        input_s = tf.placeholder(tf.float32, [None, self.state_size])
        action = tf.placeholder(tf.float32, [None, self.action_size])
        with tf.variable_scope(name):
            layer_1 = tf_utils.fc(input_s, self.n_h1, scope="fc1", activation_fn=tf.nn.relu,
                                  initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            # tf.concat((layer_1, action), 1)
            layer_2 = tf_utils.fc(tf.concat((layer_1, action), 1), self.n_h2, scope="fc2", activation_fn=tf.nn.relu,
                                  initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            q_value = tf_utils.fc(layer_2, 1, scope="out", initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
        critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return input_s, action, critic_variables, tf.squeeze(q_value)

    def get_qvalue_target(self, state, action, sess):
        return sess.run(self.q_value_target, feed_dict={
            self.input_s_target: state,
            self.action_target: action
        })

    def get_gradients(self, state, action, sess):
        return sess.run(self.action_gradients, feed_dict={
            self.input_s: state,
            self.action: action
        })

    def train(self, state, action, target, sess):
        _, loss = sess.run([self.optimize, self.loss], feed_dict={
            self.input_s: state,
            self.action: action,
            self.target: target
        })
        return loss

    def update_target(self, sess):
        sess.run(self.update_target_op)