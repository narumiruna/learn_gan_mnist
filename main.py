import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128


class Dataset:
    def __init__(self, number):
        mnist = input_data.read_data_sets('data')
        self.data = mnist.train.images[mnist.train.labels == number]
        self.n_samples, _ = self.data.shape
        self.index = 0

    def next_batch(self, batch_size):
        if self.index * batch_size + batch_size > self.n_samples:
            self.index = 0
        start = self.index * batch_size
        self.index += 1
        return self.data[start:start + batch_size]


def build_generator(z):
    with tf.variable_scope('generator'):
        z = tf.layers.dense(z, 128, name='g_d_1')
        z = tf.nn.leaky_relu(z)
        z = tf.layers.dense(z, 256, name='g_d_2')
        z = tf.nn.leaky_relu(z)
        z = tf.layers.dense(z, 28 * 28, name='g_d_3')
        z = tf.nn.sigmoid(z)
        z = tf.reshape(z, [-1, 28, 28, 1])
    return z


def build_discriminator(x):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        x = tf.reshape(x, [-1, 28 * 28])
        x = tf.layers.dense(x, 256, name='d_d_1')
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dense(x, 128, name='d_d_2')
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dense(x, 1, name='d_d_3')
    return x


def train():
    z = tf.placeholder(dtype=tf.float32, shape=[None, 100])
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])

    g_fake = build_generator(z)
    d_real = build_discriminator(x)
    d_fake = build_discriminator(g_fake)

    loss_d = -d_real + d_fake
    loss_g = -d_fake

    g_vars = tf.global_variables('generator')
    d_vars = tf.global_variables('discriminator')

    learning_rate = 5e-4
    train_g = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_g, var_list=g_vars)
    train_d = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_d, var_list=d_vars)

    c = 0.01
    clip = [v.assign(tf.clip_by_value(v, -c, c)) for v in d_vars]

    mnist_7 = Dataset(7)

    n_steps = 5001

    os.makedirs('img', exist_ok=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(n_steps):
            batch_x = np.reshape(mnist_7.next_batch(batch_size), [-1, 28, 28, 1])
            batch_z = np.random.normal(size=[batch_size, 100])

            feed_dict = {x: batch_x, z: batch_z}
            sess.run([train_d, clip], feed_dict)
            
            batch_z = np.random.normal(size=[batch_size, 100])
            feed_dict = {z: batch_z}
            sess.run(train_g, feed_dict)

            if i % 1000 == 0:
                img = sess.run(g_fake, feed_dict={z: np.random.normal(size=[1, 100])})
                plt.imshow(np.reshape(img[0], [28, 28]), cmap='gray')
                plt.savefig('img/{}.jpg'.format(str(int(i / 1000)).zfill(3)))


if __name__ == '__main__':
    train()
