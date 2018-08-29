#!/usr/bin/python

# Create dataset for toy linear regression problems
# Convert dataset to TFRecords format

import argparse
import os
import sys

import tensorflow as tf
import numpy as np

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def generate_dataset(size):
    x = np.random.rand(1, size)
    noise = np.random.rand(1, size) * 0.1
    y = x + noise

    return x[0], y[0]


def convert_to(size, name):
    x, y = generate_dataset(size)

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(size):
        example = tf.train.Example(features=tf.train.Features(feature={
            'x': _float_feature(x[index]),
            'y': _float_feature(y[index])}))
        writer.write(example.SerializeToString())
    writer.close()


def main(unused_argv):
    convert_to(FLAGS.train_size, 'train')
    convert_to(FLAGS.validation_size, 'validation')
    convert_to(FLAGS.test_size, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='/tmp/data',
        help='Directory to generate data set and convert to'
    )
    parser.add_argument(
        '--train_size',
        type=float,
        default=1000,
        help='Number of training samples'
    )
    parser.add_argument(
        '--validation_size',
        type=float,
        default=100,
        help='Number of validation samples'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=100,
        help='Number of test samples'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)




