#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

bitW = 8
bitA = 8

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

sess = tf.InteractiveSession()

X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)


def vgg(x, y_, reuse, is_train, bitW, bitA):
    with tf.variable_scope("vgg", reuse=reuse):
        net_in = InputLayer(x, name='input')
        # conv1
        net = QuanConv2d(net_in, 64, (1,1), (1,1), padding='SAME', act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_conv1_1')
        net = QuanConv2d(net, 64, (3,3), (1,1), padding='SAME', act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_conv1_2')
        net = PoolLayer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool, name='vgg_pool1')
        # conv2
        net = QuanConv2d(net, 128, (3,3), (1,1), padding='SAME', act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_conv2_1')
        net = QuanConv2d(net, 128, (3,3), (1,1), padding='SAME', act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_conv2_2')
        net = PoolLayer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool, name='vgg_pool2')
        # conv3
        net = QuanConv2d(net, 256, (3,3), (1,1), padding='SAME', act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_conv3_1')
        net = QuanConv2d(net, 256, (3,3), (1,1), padding='SAME', act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_conv3_2')
        net = QuanConv2d(net, 256, (3,3), (1,1), padding='SAME', act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_conv3_3')
        net = QuanConv2d(net, 256, (3,3), (1,1), padding='SAME', act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_conv3_4')
        net = PoolLayer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool, name='vgg_pool3')
        # conv4
        net = QuanConv2d(net, 512, (3,3), (1,1), padding='SAME', act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_conv4_1')
        net = QuanConv2d(net, 512, (3,3), (1,1), padding='SAME', act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_conv4_2')
        net = QuanConv2d(net, 512, (3,3), (1,1), padding='SAME', act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_conv4_3')
        net = QuanConv2d(net, 512, (3,3), (1,1), padding='SAME', act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_conv4_4')
        net = PoolLayer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool, name='vgg_pool4')
        # conv5
        net = QuanConv2d(net, 512, (3,3), (1,1), padding='SAME', act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_conv5_1')
        net = QuanConv2d(net, 512, (3,3), (1,1), padding='SAME', act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_conv5_2')
        net = QuanConv2d(net, 512, (3,3), (1,1), padding='SAME', act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_conv5_3')
        net = QuanConv2d(net, 512, (3,3), (1,1), padding='SAME', act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_conv5_4')
        net = PoolLayer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool, name='vgg_pool5')
        # fc 6~8
        net = FlattenLayer(net, name='flatten')
        net = QuanDenseLayer(net, 4096, act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_fc6')
        net = DenseLayer(net, 4096, act=tf.nn.relu, bitW=bitW, bitA=bitA, name='vgg_fc7')
        net = DenseLayer(net, 1000, act=None, bitW=bitW, bitA=bitA, name='vgg_fc8')

        ce = tl.cost.cross_entropy(y, y_, name='cost')
        L2 = 0
        for p in tl.layers.get_variables_with_name('relu/W', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
        cost = ce + L2

        # correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y), 1), y_)
        correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int64), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return net, cost, acc



def distort_fn(x, is_train=False):
    x = tl.prepro.crop(x, 24, 24, is_random=is_train)
    if is_train:
        x = tl.prepro.flip_axis(x, axis=1, is_random=True)
        x = tl.prepro.brightness(x, gamma=0.1, gain=1, is_random=True)
    x = (x - np.mean(x)) / max(np.std(x), 1e-5)  # avoid values divided by 0
    return x


x = tf.placeholder(dtype=tf.float32, shape=[None, 24, 24, 3], name='x')
y_ = tf.placeholder(dtype=tf.int64, shape=[None], name='y_')

network, cost, _ = vgg(x, y_, False, True, bitW=bitW, bitA=bitA)
_, cost_test, acc = vgg(x, y_, True, False, bitW=bitW, bitA=bitA)

# train
n_epoch = 50000
learning_rate = 0.0001 #0.01 for alex
print_freq = 1
batch_size = 128

train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                  use_locking=False).minimize(cost)

sess.run(tf.global_variables_initializer())

network.print_params(False)
network.print_layers()

print('   learning_rate: %f' % learning_rate)
print('   batch_size: %d' % batch_size)
print('   bitW: %d,   bitA: %d' % (bitW, bitA))

for epoch in range(n_epoch):
    start_time = time.time()
    for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        X_train_a = tl.prepro.threading_data(X_train_a, fn=distort_fn, is_train=True)  # data augmentation for training
        sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        test_loss, test_acc, n_batch = 0, 0, 0
        for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=False):
            X_test_a = tl.prepro.threading_data(X_test_a, fn=distort_fn, is_train=False)  # central crop
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_test_a, y_: y_test_a})
            test_loss += err
            test_acc += ac
            n_batch += 1
        print("   test loss: %f" % (test_loss / n_batch))
        print("   test acc: %f" % (test_acc / n_batch))
