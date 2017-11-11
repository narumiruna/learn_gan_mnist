import tensorflow as tf


def conv2d(input_,
           filter_length,
           out_channels,
           strides=[1, 1, 1, 1],
           padding='SAME',
           activation=None,
           name='conv2'):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=[filter_length, filter_length, input_.get_shape()[-1], out_channels], initializer=tf.random_normal_initializer())
        biases = tf.get_variable('biases', shape=[out_channels], initializer=tf.random_normal_initializer(0.0))
        conv2d = tf.nn.conv2d(input_, weights, strides, padding)

        if activation:
            return activation(conv2d + biases)
        else:
            return conv2d + biases


def conv2d_transpose(value,
                     batch_size,
                     height,
                     width,
                     filter_length,
                     out_channels,
                     strides=[1, 1, 1, 1],
                     activation=None,
                     name='conv2d_transpose'):
    with tf.variable_scope(name):

        if activation:
            value = activation(value)

        weights = tf.get_variable('weights', shape=[filter_length, filter_length, out_channels, value.get_shape()[-1]], initializer=tf.random_normal_initializer())
        biases = tf.get_variable('biases', shape=[out_channels])

        output_shape = [batch_size, height, width, out_channels]
        conv2d_trans = tf.nn.conv2d_transpose(value, weights, output_shape, strides)

        return conv2d_trans + biases


def dense(input_, n_neurons, activation=None, name='dense'):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=[input_.get_shape()[-1], n_neurons], initializer=tf.random_normal_initializer())
        biases = tf.get_variable('biases', shape=[n_neurons])

        if activation:
            return activation(input_ @ weights + biases)
        else:
            return input_ @ weights + biases
