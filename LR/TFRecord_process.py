# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def trainsform_to_tfrecord(field_num, libsvm_file, tfrecord_file):
    def int_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    def string_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    with open(libsvm_file, "r", encoding="utf8") as f:
        for line in f:
            try:
                splits = line.strip().split(" ")
                label = float(splits[0])
                featindex = [int(x.split(":")[0]) for x in splits[1:]]
                feature = np.zeros(field_num)
                for index in featindex:
                    feature[index] = 1
            
                features = tf.train.Features(feature={
                        "feature": string_feature(np.array(feature).astype(np.float32).tostring()), 
                        "label": float_feature(label)})
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
            except Exception as e:
                print(line, e)
    writer.close()


if __name__ == "__main__":
    from utils import load_yaml_config, parser, get_batch_dataset
    # 导入配置文件
    config = load_yaml_config("./config.yml")
    # 文件路径
    train_svm_path = config["data"]["train_svm_path"]
    test_svm_path = config["data"]["test_svm_path"]
    train_tfrecord_path = config["data"]["train_tfrecord_path"]
    test_tfrecord_path = config["data"]["test_tfrecord_path"]
    # 数据参数
    field_num = config["data"]["field_num"]
    capacity = config["data"]["capacity"]
    batch_size = config["data"]["batch_size"]
    # 训练集tfrecord
    trainsform_to_tfrecord(field_num, train_svm_path, train_tfrecord_path)
    # 测试集tfrecord
    trainsform_to_tfrecord(field_num, test_svm_path, test_tfrecord_path)
    # 解析数据
    dataset = get_batch_dataset(train_tfrecord_path, parser, capacity, batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        print(sess.run(next_element))



