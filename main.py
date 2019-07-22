# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
import tensorflow as tf
import numpy as np
import argparse
from ddpg import DDPG
from models import ActorNetwork
from models import CriticNetwork
# from ddpg_simple.exp_replay import ExpReplay
# from ddpg_simple.exp_replay import Step
from ou import OUProcess
# from ddpg import DDPG
#from ddpg_simple.ddpg_ import DDPG
# from actor import ActorNetwork
# from critic import CriticNetwork
# from exp_replay import ExpReplay
# from exp_replay import Step
# from ou import OUProcess
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import sys
import gym
from gym import wrappers
#from ddpg_simple.SimpleEnv import SnakeEnv

NUM_EPISODES = 100000
# LOG_DIR=args.log_dir

ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001
MEM_SIZE = 1000000
STATE_SIZE = 3
ACTION_SIZE = 1
BATCH_SIZE = 64
MAX_STEPS = 200
FAIL_PENALTY = 0
ACTION_RANGE = 1
EVALUATE_EVERY = 10

# def train(agent,env,sess):
#     for i in range(NUM_EPISODES):
#         cur_state = env.reset()
#         cum_reward = 0
#         while True:
#             action = agent.get_action_noise(cur_state, sess, rate=(NUM_EPISODES - i) / NUM_EPISODES)[0]
#             action_list = action.tolist()
#             print(action_list)
#             true_action = action_list.index(max(action_list))
#             print(true_action)
#
#             # for t in range(MAX_STEPS):
#             #     # print(t)
#             #     action = agent.get_action(cur_state, sess)[0]
#             #     action_list = action.tolist()
#             #     print(action_list)
#             #     true_action = action_list.index(max(action_list))
#             #     print(true_action)
#             # else:
#             #     action = agent.get_action_noise(cur_state, sess, rate=(NUM_EPISODES - i) / NUM_EPISODES)[0]
#             #     action_list = action.tolist()
#             #     true_action = action_list.index(max(action_list))
#             #     #print(action.shape)
#             next_state, reward, done, info = env.step(true_action)
#             if done:
#                 print("done")
#                 cum_reward += reward
#                 agent.addDataToBuffer(s=cur_state,a=action,r=reward,ns=next_state,d=done)
#                 # print("Episode {} finished after {} timesteps, cum_reward: {}".format(i, t + 1, cum_reward))
#                 break
#             cum_reward += reward
#             cur_state = next_state
#         agent.learn_batch(sess,batch_size=BATCH_SIZE)
def train(agent, env, sess):
  for i in range(NUM_EPISODES):
    cur_state = env.reset()
    cum_reward = 0
    # tensorboard summary
    # summary_writer = tf.summary.FileWriter(LOG_DIR+'/train', graph=tf.get_default_graph())

    if (i % EVALUATE_EVERY) == 0:
      print ('====evaluation====')
    for t in range(MAX_STEPS):
        env.render()
        if (i % EVALUATE_EVERY) == 0:
        # env.render()
            action = agent.get_action(cur_state, sess)[0]
        else:
        # decaying noise
            action = agent.get_action_noise(cur_state, sess, rate=(NUM_EPISODES-i)/NUM_EPISODES)[0]
        next_state, reward, done, info = env.step(action)
        # print(next_state.shape,reward.shape,done)
        if done:
            cum_reward += reward
            agent.addDataToBuffer(s=cur_state,a=action,r=reward,ns=next_state,d=done)
            # agent.add_step(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
            print("Episode {} finished after {} timesteps, cum_reward: {}".format(i, t + 1, cum_reward))
        #summarize(cum_reward, i, summary_writer)
            break
        cum_reward += reward
        # agent.add_step(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
        agent.addDataToBuffer(s=cur_state, a=action, r=reward, ns=next_state, d=done)
        cur_state = next_state
        # if t == MAX_STEPS - 1:
        #     print("Episode {} finished after {} timesteps, cum_reward: {}".format(i, t + 1, cum_reward))
        #     print (action)
        # summarize(cum_reward, i, summary_writer)
        agent.learn_batch(sess,batch_size=BATCH_SIZE)


env = gym.make('Pendulum-v0')
# env = wrappers.Monitor(env, '/tmp/pendulum-experiment-0', force=True)

actor = ActorNetwork(state_size=STATE_SIZE, action_size=ACTION_SIZE, lr=ACTOR_LEARNING_RATE, tau=TAU)
critic = CriticNetwork(state_size=STATE_SIZE, action_size=ACTION_SIZE, lr=CRITIC_LEARNING_RATE, tau=TAU)
noise = OUProcess(ACTION_SIZE)
# exprep = ExpReplay(mem_size=MEM_SIZE, start_mem=10000, state_size=[STATE_SIZE], kth=-1, batch_size=BATCH_SIZE)
buffer = ReplayBuffer(buffer_size=MEM_SIZE)
sess = tf.Session()
#with tf.device('/{}:0'.format(DEVICE)):
# agent = DDPG(actor=actor, critic=critic, exprep=exprep, noise=noise, action_bound=env.action_space.high)
agent = DDPG(actor=actor,critic=critic,buffer=buffer,noise=noise,action_bound=env.action_space.high)
sess.run(tf.initialize_all_variables())

train(agent, env, sess)

# if __name__ == "__main__":
#     env = SnakeEnv(gameSpeed=5,train_model=False)
#     ob_dim = env.ob_dim
#     act_dim = env.act_dim
#     actor = ActorNetwork(state_size=ob_dim, action_size=act_dim, lr=ACTOR_LEARNING_RATE, tau=TAU)
#     critic = CriticNetwork(state_size=ob_dim, action_size=act_dim, lr=CRITIC_LEARNING_RATE, tau=TAU)
#     noise = OUProcess(act_dim)
#     buffer = ReplayBuffer(buffer_size=MEM_SIZE)
#     sess = tf.Session()
#     agent = DDPG(actor=actor, critic=critic, buffer=buffer, noise=noise, action_bound=1)
#     sess.run(tf.initialize_all_variables())
#     train(agent,env,sess)
#     # print(agent.action_bound)
#     #print(actor.state_size,actor.action_size)
#    # print(critic.state_size,critic.action_size)