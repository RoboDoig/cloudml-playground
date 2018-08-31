import tensorflow as tf
import numpy

rng = numpy.random

# tf.logging.set_verbosity(tf.logging.INFO)


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'x': tf.FixedLenFeature([], tf.float32),
            'y': tf.FixedLenFeature([], tf.float32)
        })

    x = tf.cast(features['x'], tf.float32)
    y = tf.cast(features['y'], tf.float32)

    return x, y


def inputs(filename, batch_size, num_epochs):
    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)

        dataset = dataset.map(decode)

        dataset = dataset.shuffle(1000 + 3 * batch_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def train(filename, log_dir, batch_size=100, num_epochs=10, learning_rate=0.1):

    # Inputs
    x_batch, y_batch = inputs(filename, batch_size=batch_size, num_epochs=num_epochs)

    # Model weights initialisation
    W = tf.Variable(rng.randn(), name="weight")
    b = tf.Variable(rng.randn(), name="bias")

    # Linear model
    pred = tf.add(tf.multiply(x_batch, W), b)

    # Mean squared error cost function
    cost = tf.reduce_sum(tf.pow(pred-y_batch, 2))/(2*batch_size)
    train_cost_summary = tf.summary.scalar("train_cost", cost)
    train_W_summary = tf.summary.scalar("weight", W)
    train_b_summary = tf.summary.scalar("bias", b)

    # Optimization by gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initialize variables
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # create a log writer
    writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
    print('Writing summary to:', log_dir)

    with tf.Session() as sess:
        sess.run(init_op)

        try:
            step = 0
            while True:
                _, cost_val, x, y, cost_summary, W_summary, b_summary = sess.run([optimizer,
                                                                                  cost,
                                                                                  x_batch,
                                                                                  y_batch,
                                                                                  train_cost_summary,
                                                                                  train_W_summary,
                                                                                  train_b_summary])

                # writer.add_summary(cost_summary, step)
                # writer.add_summary(W_summary, step)
                # writer.add_summary(b_summary, step)

                # c = sess.run(cost_val)
                summary = tf.Summary(value=[tf.Summary.Value(tag='train', simple_value=cost_val)])
                writer.add_summary(summary, step)

                step += 1

        except tf.errors.OutOfRangeError:
            print('Done training')

        print('Final gradient: ', sess.run(W))
