import tensorflow as tf
import numpy
rng = numpy.random

tf.logging.set_verbosity(tf.logging.INFO)
from tensorflow.contrib.learn.python.learn.estimators.model_fn import ModeKeys as Modes


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'x': tf.FixedLenFeature([], tf.float32),
            'y': tf.FixedLenFeature([], tf.float32)
        })

    x = tf.cast(features['x'], tf.float32)
    y = tf.cast(features['y'], tf.float32)

    return x, y


def input_fn(filename, batch_size=100):
    filename_queue = tf.train.string_input_producer([filename])

    x, y = read_and_decode(filename_queue)
    x, y = tf.train.batch(
        [x, y], batch_size=batch_size,
        capacity=1000 + 3 * batch_size
    )

    return {'inputs': x}, y


def get_input_fn(filename, batch_size=100):
    return lambda: input_fn(filename, batch_size)


def regression_model(features, labels, mode):
    # Input layer
    x = features['inputs']
    y = labels

    # Model weights
    W = tf.Variable(rng.randn(), name='weight')
    b = tf.Variable(rng.randn(), name='bias')

    # Linear model
    pred = tf.add(tf.multiply(x, W), b)

    # Train
    loss = tf.reduce_sum(tf.pow(pred-y, 2))/(2*x.shape[0])
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def build_estimator(model_dir):
    return tf.estimator.Estimator(
        model_fn=regression_model,
        model_dir=model_dir,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=60)
    )


def serving_input_fn():
    inputs = {'inputs': tf.placeholder(tf.float32)}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
