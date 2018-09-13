# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 20:25:31 2018

@author: Meet
"""
import tensorflow as tf

class criticNeuralNetwork():
    def __init__(self, sess, ip_state_size, ip_action_size, h_1, h_2, lr, gamma):
        
        self.sess = sess
        self.input_state = tf.placeholder(shape=[None, ip_state_size], dtype=tf.float32, name="Input_state")
        self.input_action = tf.placeholder(shape=[None, ip_action_size], dtype=tf.float32, name="Input_action")
        self.Q_s = tf.placeholder(shape=[None], dtype=tf.float32, name="Q_next_state")
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32, name="Reward")
        self.gamma = gamma
        
        
        self.state_h1 = tf.layers.dense(self.input_state, h_1, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer())
        self.action_h1 = tf.layers.dense(self.input_action, h_1, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer())
        
        self.h_2 = tf.nn.relu(tf.add(self.state_h1, self.action_h1))
        self.h_3 = tf.layers.dense(self.h_2, h_2, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer())
        
        self.raw_Q = tf.layers.dense(self.h_3, 1, activation=None, kernel_initializer=tf.truncated_normal_initializer())
        
        self.critic_loss = tf.square(self.reward + self.gamma * self.Q_s - self.raw_Q)
        
        self.train_critic = tf.train.AdamOptimizer(lr).minimize(self.critic_loss)

    def get_Q(self, state, action):
        return self.sess.run(self.raw_Q, feed_dict={self.input_state: state,self.input_action: action})
    
    def get_TD_error(self, state, reward, action, state_next, action_next):
        Q_next_state = get_Q(state_next, action_next)
        td_error = self.sess.run(self.critic_loss, feed_dict={self.reward: reward, self.Q_s: Q_next_state, self.input_state: state, self.input_action: action})
        return tf.sqrt(td_error)
    
    def learn(self, state, reward, action, state_next, action_next):
        _ = self.sess.run(self.train_critic, feed_dict={self.reward: reward, self.Q_s: get_Q(state_next, action_next), self.input_state: state, self.input_action: action})    