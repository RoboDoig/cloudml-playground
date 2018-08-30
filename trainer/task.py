import tensorflow as tf
import argparse
import os
import sys

from trainer import model

FLAGS = None


def main():
    model.train()


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

    args = parser.parse_args()
    model.DATA_DIR = os.path.join(args.data_dir, 'train.tfrecords')
    model.BATCH_SIZE = args.batch_size
    model.NUM_EPOCHS = args.num_epochs
    model.LEARNING_RATE = args.learning_rate

    main()
