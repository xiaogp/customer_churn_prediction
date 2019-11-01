import shutil

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import numpy as np
from sklearn.metrics import *
from sklearn.datasets import load_svmlight_file


class FM(object):
    def __init__(self, feature_size, fm_v_size=8, loss_fuc="Cross_entropy", train_optimizer="Adam", 
                 learning_rate=0.1, reg_type="l2_reg", reg_param_w=0.0, reg_param_v=0.0, decaylearning_rate=0.9):
        self.input_x = tf.sparse_placeholder(tf.float32, name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        
        with tf.name_scope("fm_layers"):
            # 一阶系数
            FM_W = tf.get_variable(shape=[feature_size, 1], initializer=tf.glorot_normal_initializer(), name="fm_beta1")
            # 二阶交互项,n×k
            FM_V = tf.get_variable(shape=[feature_size, fm_v_size], initializer=tf.glorot_normal_initializer, name="fm_beta2")
            # 常数项
            FM_B = tf.Variable(tf.constant(0.0), dtype=tf.float32, name="fm_bias")  # W0
                        
            # 二阶交互作用
            X_square = tf.SparseTensor(self.input_x.indices, tf.square(self.input_x.values), tf.to_int64(tf.shape(self.input_x)))
            xv = tf.square(tf.sparse_tensor_dense_matmul(self.input_x, FM_V))
            p = 0.5 * tf.reshape(tf.reduce_sum(xv - tf.sparse_tensor_dense_matmul(X_square, tf.square(FM_V)), 1), [-1, 1])  # output_dim=1
            
            # 一阶相乘
            xw = tf.sparse_tensor_dense_matmul(self.input_x, FM_W)
            self.output = tf.reshape(xw + FM_B + p, [-1])

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


def get_batch(epoches, batch_size):
    x_train, y_train = load_svmlight_file("./churn_train.svm", zero_based=True)
    for epoch in range(epoches):
        ind = np.arange(x_train.shape[0])
        np.random.shuffle(ind)
        x_shuf, y_shuf = x_train[ind], y_train[ind]
        for batch in range(0, x_train.shape[0], batch_size):
            if batch + batch_size < x_train.shape[0]:
                yield x_shuf[batch: (batch + batch_size)], y_shuf[batch: (batch + batch_size)]


def train_main():
    tf.reset_default_graph()
    model = FM(feature_size=186)
               
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
        sess.run(init_op)
        shutil.rmtree("./FM_churn.log", ignore_errors=True)
        writer = tf.summary.FileWriter("./FM_churn.log", sess.graph)
        
        def feed_step(batch, label, dtype="train"):
            indices = np.mat([batch.tocoo().row, batch.tocoo().col]).transpose()
            values = batch.tocoo().data
            shape = batch.tocoo().shape
            feed_dict = {
                model.input_x: tf.SparseTensorValue(indices, values, shape),
                model.input_y: np.reshape(label, [-1, 1])
            }
            if dtype == "train":
                train_operation = [model.train_step, model.global_step, model.loss, model.auc_score, model.summary_op]
                _, step, loss_val, auc_val, merged  = sess.run(train_operation, feed_dict=feed_dict)
                writer.add_summary(merged, step)
                return step, loss_val, auc_val
            
            elif dtype == "evaluate":
                loss_val, auc_val  = sess.run([model.loss, model.auc_score], feed_dict=feed_dict)
                return loss_val, auc_val
            
            elif dtype == "test":
                logit_val = sess.run([model.logit], feed_dict=feed_dict)[0]
                print("accuracy:", accuracy_score(label, [round(x) for x in logit_val]))
                print("precision:", precision_score(label, [round(x) for x in logit_val]))
                print("reall:", recall_score(label, [round(x) for x in logit_val]))
                print("f1:", f1_score(label, [round(x) for x in logit_val]))
                print("auc:", roc_auc_score(label, logit_val))
                
        # 读取训练数据和测试数据
        batches = get_batch(10, 256)
        x_test, y_test = load_svmlight_file("./churn_test.svm", zero_based=True)
    
        for x_batch, y_batch in batches:
            step, loss_val, auc_val = feed_step(x_batch, y_batch, dtype="train")      
            if step % 100 == 0:
                print("step:", step, "loss:", loss_val, "auc:", auc_val[0])   
            if step % 1000 == 0:
                loss_val, auc_val = feed_step(x_test, y_test, dtype="evaluate")
                print("[evaluation] loss", loss_val, "auc:", auc_val[0], "\n")

        # 测试集评价
        logit_val = feed_step(x_test, y_test, dtype="test")
        
        # 模型保存
        shutil.rmtree("./FM_churn.pb", ignore_errors=True)
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
    x_test, y_test = load_svmlight_file("./churn_test.svm", zero_based=True)
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], "./FM_churn.pb")
        graph = tf.get_default_graph()
        # 获得tensor
        input_value = graph.get_operation_by_name("input_x/values").outputs[0]
        input_shape = graph.get_operation_by_name("input_x/shape").outputs[0]
        input_indices = graph.get_operation_by_name("input_x/indices").outputs[0]
        logit = graph.get_tensor_by_name("predict/logit:0")

        indices = np.mat([x_test.tocoo().row, x_test.tocoo().col]).transpose()
        values = x_test.tocoo().data
        shape = x_test.tocoo().shape
        feed_dict = {input_value: values, input_shape: shape, input_indices: indices}
        predictions = sess.run(logit, feed_dict=feed_dict)
                
        with open("./fm_svm_res.txt", "w") as f:
            for i in predictions:
                f.write(str(i) + "\n")
    

if __name__ == "__main__":
    train_main()
    predict_main()
