# -*- coding: utf-8 -*-

"""
Author: Vic Chan
Date: 7/8/2017
Content: VGG Image Data (CIFAR - 100)
"""

"""
CIFAR-100
"""

import cPickle as pickle
import numpy as np
import os

# train 50000
# test 10000


class CIFAR(object):

    def __init__(self, path, test_path):
        self._current_index = 0
        self._test_index = 0

        with open(path, 'r') as f:
            datadict = pickle.load(f)
            X = datadict['data']
            Y = datadict['fine_labels']
            self.train_images = X.reshape(50000, 3072).astype("float")
            self.train_labels = np.array(Y)

        with open(test_path, 'r') as f:
            datadict = pickle.load(f)
            X = datadict['data']
            Y = datadict['fine_labels']
            self.test_images = X.reshape(10000, 3072).astype("float")
            self.test_labels = np.array(Y)

    def dense_to_one_hot(self, labels_dense, num_classes=100):
        """Convert class labels from scalars to one-hot vectors."""
        arr = [0] * num_classes
        arr[labels_dense] = 1
        return np.array(arr)

    def next_batch(self, batch_size):
        start = self._current_index
        self._current_index += batch_size
        end = self._current_index
        if self._current_index >= len(self.train_images):
            self._current_index = 0
        return self.train_images[start:end]/255.0, np.array([self.dense_to_one_hot(x) for x in self.train_labels[start:end]])

    def test_batch(self, batch_size):
        start = self._test_index
        self._test_index += batch_size
        end = self._test_index
        return self.test_images[start:end]/255.0, np.array([self.dense_to_one_hot(x) for x in self.test_labels[start:end]])