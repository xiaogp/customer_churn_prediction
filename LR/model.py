# -*- coding: utf-8 -*-

import tensorflow as tf


#class LR(object):
#    def __init__(self, feature_size, loss_fuc="Cross_entropy", train_optimizer="Adam", learning_rate=0.01, 
#                 reg_type="l2_reg", reg_param=0.0, decaylearning_rate=0.9):
#        
#        self.input_x = tf.placeholder(tf.float32, [None, feature_size], name="input_x")
#        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
#        self.global_step = tf.Variable(0, name="global_step", trainable=False)
#        
#        with tf.name_scope("lr_layer"):
#            # 一阶系数
#            w = tf.get_variable(shape=[feature_size], initializer=tf.glorot_normal_initializer(), name="beta")
#            Y_first = tf.reduce_mean(tf.multiply(w, self.input_x), axis=1)
#            # 常数项
#            b = tf.Variable(tf.constant(0.0), dtype=tf.float32, name="bias")  
#            # output
#            self.output = tf.add(Y_first, b, name="output")
#            
#        with tf.name_scope("predict"):
#            self.logit = tf.nn.sigmoid(self.output, name="logit")
#            self.auc_score = tf.metrics.auc(self.input_y, self.logit)
#        
#        with tf.name_scope("loss"):
#            if reg_type == "l1_reg":
#                regularization = tf.contrib.layers.l1_regularizer(reg_param)(w)
#            
#            elif reg_type == "l2_reg":
#                regularization = reg_param * tf.nn.l2_loss(w)
#            
#            else:
#                regularization = reg_param * tf.nn.l2_loss(w)
#            
#            if loss_fuc == "Squared_error":
#                self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_y - self.output))) + regularization
#
#            elif loss_fuc == "Cross_entropy":
#                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#                    logits=tf.reshape(self.output, [-1]),
#                    labels=tf.reshape(self.input_y, [-1]))) + regularization
#                
#        with tf.name_scope("optimizer"):
#            if decaylearning_rate != 1:
#                learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 100, decaylearning_rate)
#            if train_optimizer == 'Adam':
#                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#            elif train_optimizer == 'Adagrad':
#                optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
#            elif train_optimizer == 'Momentum':
#                optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
#            
#            self.train_step = optimizer.minimize(self.loss, global_step=self.global_step)
#            
#        with tf.name_scope("summaries"):
#            tf.summary.scalar("loss", self.loss)
#            tf.summary.scalar("auc", self.auc_score[0])
#            tf.summary.histogram("beta", w)
#            tf.summary.histogram("bias", b)
#            self.summary_op = tf.summary.merge_all()
                

class LR(object):
    def __init__(self, batch, feature_size, loss_fuc="Cross_entropy", train_optimizer="Adam", 
                 learning_rate=0.01, reg_type="l2_reg", reg_param=0.0, decaylearning_rate=0.95):
        self.input_x, self.input_y = batch.get_next()
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        
        with tf.name_scope("lr_layer"):
            # 一阶系数
            w = tf.get_variable(shape=[feature_size], initializer=tf.glorot_normal_initializer(), name="beta")
            Y_first = tf.reduce_mean(tf.multiply(w, self.input_x), axis=1)
            # 常数项
            b = tf.Variable(tf.constant(0.0), dtype=tf.float32, name="bias")  
            # output
            self.output = tf.add(Y_first, b, name="output")
            
        with tf.name_scope("predict"):
            self.logit = tf.nn.sigmoid(self.output, name="logit")
            self.auc_score = tf.metrics.auc(self.input_y, self.logit)
        
        with tf.name_scope("loss"):
            if reg_type == "l1_reg":
                regularization = tf.contrib.layers.l1_regularizer(reg_param)(w)
            
            elif reg_type == "l2_reg":
                regularization = reg_param * tf.nn.l2_loss(w)
            
            else:
                regularization = reg_param * tf.nn.l2_loss(w)
            
            if loss_fuc == "Squared_error":
                self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_y - self.output))) + regularization

            elif loss_fuc == "Cross_entropy":
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.reshape(self.output, [-1]),
                    labels=tf.reshape(self.input_y, [-1]))) + regularization
                
        with tf.name_scope("optimizer"):
            if decaylearning_rate != 1:
                learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 100, decaylearning_rate)
            if train_optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            elif train_optimizer == 'Adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            elif train_optimizer == 'Momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
            
            self.train_step = optimizer.minimize(self.loss, global_step=self.global_step)
            
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("auc", self.auc_score[0])
            tf.summary.histogram("beta", w)
            tf.summary.histogram("bias", b)
            self.summary_op = tf.summary.merge_all()
            

            
                