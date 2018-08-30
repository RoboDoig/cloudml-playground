import tensorflow as tf
import argparse
import os
import sys

from trainer import model

FLAGS = None


def main(_):
    model.train(os.path.join(FLAGS.data_dir, 'train.tfrecords'),
                batch_size=FLAGS.batch_size,
                num_epochs=FLAGS.num_epochs,
                learning_rate=FLAGS.learning_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='Bucket to find training data',
        required=True
    )
    parser.add_argument(
        '--batch_size',
        type=float,
        default=100,
        help='Specify batch size to use'
    )
    parser.add_argument(
        '--num_epochs',
        type=float,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=100,
        help='Specify gradient descent learning rate'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)