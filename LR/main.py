# -*- coding: utf-8 -*-


import shutil
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

from utils import load_yaml_config, parser, get_batch_dataset, get_dataset
from model import LR


config = load_yaml_config("./config.yml")
# data
train_tfrecord_path = config["data"]["train_tfrecord_path"]
test_tfrecord_path = config["data"]["test_tfrecord_path"]
capacity = config["data"]["capacity"]
batch_size = config["data"]["batch_size"]
field_num = config["data"]["field_num"]
test_data_count = config["data"]["test_data_count"]
# model
learning_rate = config["model"]["learning_rate"]
train_steps = config["model"]["train_steps"]
train_display_steps = config["model"]["train_display_steps"]
test_display_steps = config["model"]["test_display_steps"]
pb_path = config["model"]["pb_path"]


def train():
    tf.reset_default_graph()
    
    # 读取tfrecords
    train_dataset = get_batch_dataset(train_tfrecord_path, parser, capacity, batch_size)
    dev_dataset = get_dataset(test_tfrecord_path, parser, test_data_count)
    
    # 创建句柄指定训练集和测试集
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()
    
    model = LR(batch=iterator, feature_size=field_num, learning_rate=learning_rate)
    
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
        sess.run(init_op)
        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())
        
        for _ in range(train_steps):
            train_op = [model.train_step, model.global_step, model.loss, model.auc_score, model.summary_op]
            _, step, loss_val, auc_val, merged = sess.run(train_op, feed_dict={handle: train_handle})
            
            if step % train_display_steps == 0:
                print("step:", step, "loss:", loss_val, "auc:", auc_val[0])
            
            if step % test_display_steps == 0:
                loss_val, auc_val = sess.run([model.loss, model.auc_score], feed_dict={handle: dev_handle})
                print("[evaluation]", "loss:", loss_val, "auc:", auc_val[0])
                print(" ")
        
        # 测试集评价
        logit_val = sess.run(model.logit, feed_dict={handle: dev_handle})
        y_test = sess.run(dev_iterator.get_next())[1]
        print("accuracy:", accuracy_score(y_test, [round(x) for x in logit_val]))
        print("precision:", precision_score(y_test, [round(x) for x in logit_val]))
        print("reall:", recall_score(y_test, [round(x) for x in logit_val]))
        print("f1:", f1_score(y_test, [round(x) for x in logit_val]))
        print("auc:", roc_auc_score(y_test, logit_val))
        
        # 模型保存
        shutil.rmtree(pb_path, ignore_errors=True)
        builder = tf.saved_model.builder.SavedModelBuilder(pb_path)
        inputs = {'input_x': tf.saved_model.utils.build_tensor_info(model.input_x)}
        outputs = {'output': tf.saved_model.utils.build_tensor_info(model.logit)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], {'my_signature': signature})
        builder.save()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)
    train()


    
    
    