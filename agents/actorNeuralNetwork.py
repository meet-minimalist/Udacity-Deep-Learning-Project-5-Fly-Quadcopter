# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 00:39:47 2018

@author: Meet
"""
import tensorflow as tf

class actorNerualNetwork():
    def __init__(self, sess, ip_state, h1, h2, op, lr, r_high, r_low):
        self.sess = sess
        self.rotor_speed_high = r_high
        self.rotor_speed_low = r_low
        
        self.inputs_ = tf.placeholder(shape=[None, ip_state], dtype=tf.float32, name="states")
        
        self.hidden_1 = tf.layers.dense(self.inputs_, h1, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer())
        self.hidden_2 = tf.layers.dense(self.hidden_1, h2, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer())
        
        # Used sigmoid activation to convert the output in range of [0, 1]
        # So that it can be scaled by range = rotor_speed.high - rotor_speed.low
        # and then rotor_speed.low will be added to obtain correct ranged outputs
        self.raw_output = tf.layers.dense(self.hidden_2, op, activation=tf.nn.sigmoid, kernel_initializer=tf.truncated_normal_initializer())
        
        self.real_op = self.raw_output*(self.rotor_speed_high - self.rotor_speed_low) + self.rotor_speed_low
        
        self.log_op = tf.log(self.raw_output)
        
        self.Q_s = tf.placeholder(shape=[None], dtype=tf.float32, name="Q_s")
        
        self.actor_loss = tf.reduce_mean(self.log_op*self.Q_s) 
        
        # here -ve sign is used to that we can minimize the loss
        # which is to maximize log multiplication we need to minimize the -ve value of it.
        
        self.train_actor = tf.train.AdamOptimizer(lr).minimize(self.actor_loss)
        
    def learn(self, s, q_s):
        _ = self.sess.run(self.train_actor, feed_dict={self.inputs_: s, self.Q_s: q_s})
        
    def choose_action(self, s):
        rotor_speeds = self.sess.run(self.real_op, feed_dict={self.inputs_: s})
        return rotor_speeds