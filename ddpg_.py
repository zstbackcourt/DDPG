# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""

import numpy as np
import tensorflow as tf


class DDPG(object):


  def __init__(self, actor, critic, buffer, noise, gamma=0.99, action_bound=1):
    self.actor = actor
    self.critic = critic
    self.buffer = buffer
    self.noise = noise
    #self.total_steps = 0
    self.gamma = 0.99
    self.action_bound = action_bound


  def addDataToBuffer(self, s,a,r,ns,d):
    self.buffer.add(s, a, r, ns, d)


  def get_action(self, state, sess):
    state = np.reshape(state,[-1, self.actor.state_size])
    action = self.actor.get_action(state, sess) * self.action_bound
    return action


  def get_action_noise(self, state, sess, rate=1):
    state = np.reshape(state,[-1, self.actor.state_size])
    action = self.actor.get_action(state, sess) * self.action_bound
    action = action + self.noise.noise() * rate
    return action


  def learn_batch(self, sess,batch_size):
    # sample a random minibatch of N tranistions
    if self.buffer.size() < batch_size:
        return
    minibatch = self.buffer.get_batch(batch_size)
    if len(minibatch)==0:
      return
    """
    minibatch = self.buffer.get_batch(batch_size)
    state_batch = np.asarray([data[0] for data in minibatch])
    action_batch = np.asarray([data[1] for data in minibatch])
    reward_batch = np.asarray([data[2] for data in minibatch])
    next_state_batch = np.asarray([data[3] for data in minibatch])
    done_batch = np.asarray([data[4] for data in minibatch])
    action_batch = np.resize(action_batch, [batch_size, self.act_dim])
    next_action_batch = self.tactor.policy_fn(next_state_batch)
    q_value_batch = self.tcritic.value_fn(next_state_batch, next_action_batch)
    """

    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]
    done_batch = [data[4] for data in minibatch]

    # compute y_i (target q)
    # next_s = [s.next_step for s in batch]
    next_a_target = self.actor.get_action_target(next_state_batch, sess)
    next_q_target = self.critic.get_qvalue_target(next_state_batch, next_a_target, sess)
    y = np.array([data[2]+self.gamma*next_q_target[i]*(1-data[4]) for i,data in enumerate(minibatch)])
    # y = np.array([s.reward + self.gamma*next_q_target[i]*(1-s.done) for i,s in enumerate(batch)])
    y = y.reshape([len(minibatch)])

    # update ciritc by minimizing l2 loss
    # cur_s = [s.cur_step for s in batch] state_batch
    # a = [s.action for s in batch] action_batch
    l = self.critic.train(state_batch, action_batch, y, sess)

    # update actor policy with sampled gradient
    cur_a_pred = self.actor.get_action(state_batch, sess)
    a_gradients = self.critic.get_gradients(state_batch, cur_a_pred, sess)
    self.actor.train(state_batch, a_gradients[0], sess)

    # update target network:
    self.actor.update_target(sess)
    self.critic.update_target(sess)
    return l
