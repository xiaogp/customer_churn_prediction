# -*- coding: utf-8 -*-

import yaml

import tensorflow as tf

def load_yaml_config(yaml_file):
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def parser(example):
    dicts = {
        'feature': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'label': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    }

    parsed_example = tf.parse_single_example(example, dicts)
    feature = tf.decode_raw(parsed_example['feature'], tf.float32)
    label = tf.cast(parsed_example['label'],tf.float32)

    return feature, label


def get_batch_dataset(tfrecord_file, parser, capacity=100, batch_size=32):
    dataset = tf.data.TFRecordDataset(tfrecord_file).map(parser).shuffle(capacity).repeat().batch(batch_size)
    
    return dataset

def get_dataset(tfrecord_file, parser, batch_size):
    dataset = tf.data.TFRecordDataset(tfrecord_file).map(parser).repeat().batch(batch_size)
    
    return dataset


def total_sample(file_name):
    sample_nums = 0
    for record in tf.python_io.tf_record_iterator(file_name):
        sample_nums += 1
        
    return sample_nums
