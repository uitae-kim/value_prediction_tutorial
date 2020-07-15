import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


def create_checkpoint_dir(path):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    if not os.path.exists(f'checkpoints/{path}'):
        os.makedirs(f'checkpoints/{path}')


def model(layer, num_features):
    # session
    net = tf.Session()

    # saver
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1,
                           save_relative_paths=True)

    X = tf.placeholder(dtype=tf.float32, shape=[None, num_features])
    y = tf.placeholder(dtype=tf.float32, shape=[None])

    # init theta
    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(
        mode='fan_avg',
        distribution='uniform',
        scale=sigma
    )
    bias_initializer = tf.zeros_initializer()

    W_hidden = []
    bias_hidden = []

    for i in range(len(layer)):
        if i == 0:
            _in = num_features
        else:
            _in = layer[i - 1]
        _out = layer[i]

        W_hidden.append(tf.Variable(weight_initializer([_in, _out])))
        bias_hidden.append(tf.Variable(bias_initializer([_out])))

    W_out = tf.Variable(weight_initializer([layer[-1], 1]))
    bias_out = tf.Variable(bias_initializer([1]))

    # layers
    hidden = []
    for i in range(len(layer)):
        if i == 0:
            _in = X
        else:
            _in = hidden[i - 1]
        _W_hidden = W_hidden[i]
        _bias = bias_hidden[i]
        hidden.append(tf.nn.relu(tf.add(tf.matmul(_in, _W_hidden), _bias)))

    out = tf.transpose(tf.add(tf.matmul(hidden[-1], W_out), bias_out))

    # cost function
    mse = tf.reduce_mean(tf.squared_difference(out, y))

    # optimizer
    adam = tf.train.AdamOptimizer().minimize(mse)

    net.run(tf.global_variables_initializer())

    network = lambda data_x, data_y: net.run(adam, feed_dict={X: data_x, y: data_y})
    error = lambda data_x, data_y: net.run(mse, feed_dict={X: data_x, y: data_y})
    output = lambda data_x: net.run(out, feed_dict={X: data_x})
    save = lambda step: saver.save(net, f'./checkpoints/{layer[0]}', step)

    return network, error, output, save


def run(network, error, output, save, X_train, y_train, X_cv, y_cv, X_test, y_test, batch_size=8, num_epochs=100):
    # fit
    mse_train = []
    mse_cv = []
    mse_test = []

    pred = y_test.transpose()

    for epoch in range(num_epochs):
        shuffle = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle]
        y_train = y_train[shuffle]

        for i in range(0, X_train.shape[0] // batch_size):
            start = i * batch_size
            batch_X = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]

            network(batch_X, batch_y)

            if np.mod(i, 10) == 0:
                mse_train.append(error(X_train, y_train))
                mse_cv.append(error(X_cv, y_cv))
                mse_test.append(error(X_test, y_test))
                print(f'Train Error: {mse_train[-1]} / Cross Validation Error: {mse_cv[-1]} / Test Error: {mse_test[-1]}')

                pred = output(X_test)

        if (epoch + 1) % 25 == 0:
            save(epoch)

    return pred.transpose()
