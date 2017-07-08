# -*- coding: utf-8 -*-

"""
Author: Vic Chan
Date: 7/8/2017
Content: VGG Implementation
"""

import tensorflow as tf
import data as dataset

learning_rate = 0.001
training_iterations = 50000
batch_size = 128
show_step = 10

image_size = 32 * 32 * 3
image_classes = 100
test_step = 100

image = tf.placeholder(float, [None, image_size], name='image')
label = tf.placeholder(float, [None, image_classes], name='label')


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def conv2d(name, x, filter_size, filter_nums, channels, b, stride=1):
    x = tf.nn.conv2d(x,
                     filter=init_weights([filter_size, filter_size, channels, filter_nums]),
                     strides=[1, stride, stride, 1],
                     padding='SAME',
                     name=name)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool(name, x, pool_size, stride):
    return tf.nn.max_pool(x,
                          ksize=[1, pool_size, pool_size, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME',
                          name=name)


def init_bias(dim):
    return tf.Variable(tf.random_normal(dim))

def fc(name, x, inputSize, outputSize):
    weights = init_weights([inputSize, outputSize])
    bias = init_bias([outputSize])
    fc1 = tf.matmul(x, weights)
    return tf.nn.relu(tf.add(fc1, bias))


def vgg(x):
    x = tf.reshape(x, shape=[-1, 32, 32, 3])

    conv_1 = conv2d('conv_1', x, 3, 64, 3, init_weights([64]))
    conv_2 = conv2d('conv_2', conv_1, 3, 64, 64, init_weights([64]))

    pool_1 = maxpool('pool_1', conv_2, 2, 2)

    conv_3 = conv2d('conv_3', pool_1, 3, 128, 64, init_weights([128]))
    conv_4 = conv2d('conv_4', conv_3, 3, 128, 128, init_weights([128]))

    pool_2 = maxpool('pool_2', conv_4, 2, 2)

    conv_5 = conv2d('conv_5', pool_2, 3, 256, 128, init_weights([256]))
    conv_6 = conv2d('conv_6', conv_5, 3, 256, 256, init_weights([256]))
    conv_7 = conv2d('conv_7', conv_6, 3, 256, 256, init_weights([256]))

    pool_3 = maxpool('pool_3', conv_7, 2, 2)

    conv_8 = conv2d('conv_8', pool_3, 3, 512, 256, init_weights([512]))
    conv_9 = conv2d('conv_9', conv_8, 3, 512, 512, init_weights([512]))
    conv_10 = conv2d('conv_10', conv_9, 3, 512, 512, init_weights([512]))

    pool_4 = maxpool('pool_4', conv_10, 2, 2)

    conv_8 = conv2d('conv_8', pool_4, 3, 512, 256, init_weights([512]))
    conv_9 = conv2d('conv_9', conv_8, 3, 512, 512, init_weights([512]))
    conv_10 = conv2d('conv_10', conv_9, 3, 512, 512, init_weights([512]))

    pool_5 = maxpool('pool_4', conv_10, 2, 2)

    x = tf.reshape(pool_5, [-1])
    fc_1 = fc('fc_1', x, x.shape[0], 4096)
    fc_2 = fc('fc_2', fc_1, 4096, 4096)
    fc_3 = fc('fc_3', fc_2, 4096, 1000)
    weights = init_weights([1000, 100])
    bias = init_bias([100])
    output = tf.matmul(x, weights)
    return output


pred = vgg(image)


# loss + optimize
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(image, label))
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()


path = '~/Dev/DeepLearning/Paddle/DeepLearningWithPaddle/GoogLeNet/data'
cifar = dataset.CIFAR(path + 'train', path + 'test')

with tf.Session() as sess:
    sess.run(init)

    step = 1

    while step * batch_size < training_iterations:
        batch_images, batch_labels = cifar.next_batch(batch_size)
        sess.run(optimizer, feed_dict={'image': batch_images, 'label': batch_labels})
        if step % show_step == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={'image': batch_images, 'label': batch_labels})
            print("[Iter %s] LOSS=%s Train Accuracy=%s\n" % (step * batch_size, cost, accuracy))
        if step % test_step == 0:
            batch_test_images, batch_test_labels = cifar.test_batch(128)
            print("[Testing Accuracy]: %s" % (sess.run(accuracy, feed_dict={'image': batch_test_images, 'label': batch_test_labels})))
        step += 1

    print("Optimize Finished")


