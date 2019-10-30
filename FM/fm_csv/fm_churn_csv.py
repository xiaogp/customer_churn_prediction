import shutil
import random
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import *
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


class FM(object):
    def __init__(self, feature_size, fm_v_size=8, loss_fuc="Cross_entropy", train_optimizer="Adam", 
                 learning_rate=0.1, reg_type="l2_reg", reg_param_w=0.0, reg_param_v=0.0, decaylearning_rate=0.9):
        self.input_x = tf.placeholder(tf.float32, [None, feature_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        
        with tf.name_scope("fm_layers"):
            # 一阶系数
            FM_W = tf.get_variable(shape=[feature_size], initializer=tf.glorot_normal_initializer(), name="fm_beta1")
            # 二阶交互项,n×k
            FM_V = tf.get_variable(shape=[feature_size, fm_v_size], initializer=tf.glorot_normal_initializer, name="fm_beta2")
            # 常数项
            FM_B = tf.Variable(tf.constant(0.0), dtype=tf.float32, name="fm_bias")  # W0
            
            # 一阶相乘
            Y_first = tf.multiply(FM_W, self.input_x)
            
            # 二阶交互作用
            embeddings = tf.multiply(FM_V, tf.reshape(self.input_x, (-1, feature_size, 1)))  # None * V * X
            summed_features_emb = tf.reduce_sum(embeddings, 1)  # sum(v*x)
            summed_features_emb_square = tf.square(summed_features_emb)  # (sum(v*x))^2
            squared_features_emb = tf.square(embeddings)  # (v*x)^2
            squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # sum((v*x)^2)
            Y_second = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # 0.5*((sum(v*x))^2 - sum((v*x)^2))

            # 一阶 + 二阶 + 偏置
            FM_out_lay1 = tf.concat([Y_first, Y_second], axis=1)  # out = W * X + Vij * Vij* Xij
            y_out = tf.reduce_sum(FM_out_lay1, 1)
            y_d = tf.reshape(y_out, shape=[-1])  # out = out + bias
            y_bias = FM_B * tf.ones_like(y_d, dtype=tf.float32)  # Y_bias
            self.output = tf.add(y_out, y_bias, name='output')
        
        with tf.name_scope("predict"):
            self.logit = tf.nn.sigmoid(self.output, name='logit')
            self.auc_score = tf.metrics.auc(self.input_y, self.logit)
            
        with tf.name_scope("loss"):
            if reg_type == 'l1_reg':
                regularization = tf.contrib.layers.l1_regularizer(reg_param_w)(FM_W) + \
                                 tf.contrib.layers.l1_regularizer(reg_param_v)(FM_V)
            elif reg_type == 'l2_reg':
                regularization = reg_param_w * tf.nn.l2_loss(FM_W) + reg_param_v * tf.nn.l2_loss(FM_V)
            else:
                regularization = reg_param_w * tf.nn.l2_loss(FM_W) + reg_param_v * tf.nn.l2_loss(FM_V)
            if loss_fuc == 'Squared_error':
                self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_y - self.output), 
                                                    reduction_indices=[1])) + regularization
            elif loss_fuc == 'Cross_entropy':
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
            tf.summary.histogram("FM_W", FM_W)
            tf.summary.histogram("FM_V", FM_V)
            self.summary_op = tf.summary.merge_all()
            
            
def preprocessing():
    """对数据进行onehot和minmax处理，分割训练和测试样本，pickle序列化"""
    df = pd.read_csv("./churn2.csv", sep=",")
    train, test = train_test_split(df, test_size=0.2)
    
    onehot_transform = OneHotEncoder(categories="auto", handle_unknown="ignore")
    category_cols = train.columns.difference(["label", "USR_NUM_ID"])
    train_feature= onehot_transform.fit_transform(train[category_cols]).toarray()
    test_feature = onehot_transform.transform(test[category_cols]).toarray()
    
    print("{0:-^60}".format("data overview"))
    print("训练样本数", train_feature.shape[0])
    print("训练样本数", test_feature.shape[0])
    print("特征数", train_feature.shape[1])
    
    pickle.dump(onehot_transform, open("./encoder_churn.pkl", "wb"))
    pickle.dump((train_feature, train["label"].values), open("./train_churn.pkl", "wb"))
    pickle.dump((test_feature, test["label"].values), open("./test_churn.pkl", "wb"))


def get_batch(epoches, batch_size):
    """ 将数据转化为generator格式 """
    x_train, y_train = pickle.load(open("./train_churn.pkl", "rb"))
    data = list(zip(x_train, y_train))
    for epoch in range(epoches):
        random.shuffle(data)
        for batch in range(0, len(data), batch_size):
            if batch + batch_size < len(data):
                yield data[batch: (batch + batch_size)]


def train_main():
    FLAGS = tf.app.flags.FLAGS
    # model参数
    tf.app.flags.DEFINE_integer("feature_size", 149, "number of fields")
    tf.app.flags.DEFINE_integer("fm_v_size", 8, "number of implicit vector dimensions")
    tf.app.flags.DEFINE_string("loss_fuc", "Cross_entropy", "loss function")
    tf.app.flags.DEFINE_string("train_optimizer", "Adam", "optimizer method")
    tf.app.flags.DEFINE_float("learning_rate", 0.1, "initial learning rate")
    tf.app.flags.DEFINE_string("reg_type", "l2_reg", "regularization method")
    tf.app.flags.DEFINE_float("reg_param_w", 0.0, "first order beta regularization param")
    tf.app.flags.DEFINE_float("reg_param_v", 0.0, "second order beta regularization param")
    tf.app.flags.DEFINE_float("decaylearning_rate", 0.9, "decay learning rate param")
    # data参数
    tf.app.flags.DEFINE_integer("epoches", 100, "number of data repeat time")
    tf.app.flags.DEFINE_integer("batch_size", 2048, "number of train data each batch")
    print(FLAGS)
    tf.reset_default_graph()
    model = FM(FLAGS.feature_size, FLAGS.fm_v_size, FLAGS.loss_fuc, FLAGS.train_optimizer, 
               FLAGS.learning_rate, FLAGS.reg_type, FLAGS.reg_param_w, FLAGS.reg_param_v, 
               FLAGS.decaylearning_rate)

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
        sess.run(init_op)
        shutil.rmtree("./FM_churn.log", ignore_errors=True)
        writer = tf.summary.FileWriter("./FM_churn.log", sess.graph)
        batches = get_batch(FLAGS.epoches, FLAGS.batch_size)
        x_test, y_test = pickle.load(open("./test_churn.pkl", "rb"))
        
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            feed_dict = {model.input_x: x_batch, model.input_y: np.reshape(y_batch, [-1, 1])}
            _, step, loss_val, auc_val, merged  = sess.run([model.train_step, model.global_step, model.loss, model.auc_score, model.summary_op], feed_dict=feed_dict)
            writer.add_summary(merged, step)
            
            if step % 100 == 0:
                print("step:", step, "loss:", loss_val, "auc:", auc_val[0])
            
            if step % 1000 == 0:
                feed_dict = {model.input_x: x_test, model.input_y: np.reshape(y_test, [-1, 1])}
                _, loss_val, auc_val = sess.run([model.train_step, model.loss, model.auc_score], feed_dict=feed_dict)
                print("[evaluation]", "loss:", loss_val, "auc:", auc_val[0])
                print(" ")
        
        # 测试集评价
        feed_dict = {model.input_x: x_test, model.input_y: np.reshape(y_test, [-1, 1])}
        _, logit_val = sess.run([model.train_step,model.logit], feed_dict=feed_dict)
        
        print("accuracy:", accuracy_score(y_test, [round(x) for x in logit_val]))
        print("precision:", precision_score(y_test, [round(x) for x in logit_val]))
        print("reall:", recall_score(y_test, [round(x) for x in logit_val]))
        print("f1:", f1_score(y_test, [round(x) for x in logit_val]))
        print("auc:", roc_auc_score(y_test, logit_val))
        
        # 模型保存
        shutil.rmtree("FM_churn.pb", ignore_errors=True)
        builder = tf.saved_model.builder.SavedModelBuilder("FM_churn.pb")
        inputs = {'input_x': tf.saved_model.utils.build_tensor_info(model.input_x)}
        outputs = {'output': tf.saved_model.utils.build_tensor_info(model.logit)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], {'my_signature': signature})
        builder.save()
        
        
def predict_main():
    test = pd.read_csv("./churn2.csv").dropna()
    onehot_transform = pickle.load(open("./encoder_churn.pkl", "rb"))
    test_feature = onehot_transform.transform(test[test.columns.difference(["label", "USR_NUM_ID"])]).toarray()
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], "./FM_churn.pb")
        graph = tf.get_default_graph()
        # 获得tensor
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        logit = graph.get_tensor_by_name("predict/logit:0")
        predictions = sess.run(logit, feed_dict={input_x: test_feature})
    
    test["predictions"] = predictions
    result = test[["USR_NUM_ID", "label", "predictions"]]
    result.to_csv("./churn_FM_result.csv", index=False)

if __name__ == "__main__":
    preprocessing()
    train_main()
    predict_main()