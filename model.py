import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from ops import dense, conv2d, conv2d_transpose


batch_size = 64


keep_prob = tf.placeholder(dtype=tf.float32)
z = tf.placeholder(dtype=tf.float32, shape=[None, 100])
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])


def build_generator(input_):
    with tf.variable_scope('generator'):
        dense_1 = dense(input_, 7 * 7 * 64, activation=tf.nn.relu, name='dense_1')
        drop_1 = tf.nn.dropout(dense_1, keep_prob)

        reshape_1 = tf.reshape(drop_1, shape=[-1, 7, 7, 64])

        deconv_1 = conv2d_transpose(reshape_1, batch_size, 14, 14, 5, 32, strides=[1, 2, 2, 1], activation=tf.nn.relu, name='deconv_1')
        drop_2 = tf.nn.dropout(deconv_1, keep_prob)

        deconv_2 = conv2d_transpose(drop_2, batch_size, 28, 28, 5, 1, strides=[1, 2, 2, 1], activation=tf.nn.relu, name='deconv_2')
        drop_3 = tf.nn.dropout(deconv_2, keep_prob)

        return drop_3


def build_discriminator(input_):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        conv_1 = conv2d(input_, 5, 32, activation=tf.nn.relu, name='conv_1')
        pool_1 = tf.nn.avg_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        drop_1 = tf.nn.dropout(pool_1, keep_prob)

        conv_2 = conv2d(drop_1, 5, 64, activation=tf.nn.relu, name='conv_2')
        pool_2 = tf.nn.avg_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        drop_2 = tf.nn.dropout(pool_2, keep_prob)

        flatten_1 = tf.reshape(drop_2, shape=[-1, 7 * 7 * 64])
        dense_1 = dense(flatten_1, 1024, activation=tf.nn.relu, name='dense_1')
        drop_3 = tf.nn.dropout(dense_1, keep_prob)

        dense_2 = dense(drop_3, 1, name='dense_2')
        return dense_2

g = build_generator(z)

d_x = build_discriminator(x)
d_z = build_discriminator(g)


d_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
g_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')


loss_d = - d_x + d_z
loss_g = - d_z


d_minimizer = tf.train.RMSPropOptimizer(0.00005).minimize(loss_d, var_list=d_var)
g_minimizer = tf.train.RMSPropOptimizer(0.00005).minimize(loss_g, var_list=g_var)


mnist = input_data.read_data_sets('MNIST')

print(mnist.train)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):

        for j in range(10):
            # sess.run(d_minimizer, feed_dict={x, z, keep_prob:0.5})
            pass
