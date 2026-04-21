import numpy as np
import tensorflow as tf


class Cifar10Data:
    def __init__(self):
        self.shuffled_train_index = 0
        self.shuffled_test_index = 0

    def load(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        self.train_images = x_train.astype(np.float32) / 255.0
        self.test_images  = x_test.astype(np.float32)  / 255.0

        self.train_labels = tf.keras.utils.to_categorical(y_train, 10).astype(np.float32)
        self.test_labels  = tf.keras.utils.to_categorical(y_test,  10).astype(np.float32)

        self.train_indices = np.arange(len(self.train_images))
        self.test_indices  = np.arange(len(self.test_images))
        np.random.shuffle(self.train_indices)
        np.random.shuffle(self.test_indices)
        self.shuffled_train_index = 0
        self.shuffled_test_index  = 0

    def next_train_batch(self, n):
        if self.shuffled_train_index + n > len(self.train_indices):
            np.random.shuffle(self.train_indices)
            self.shuffled_train_index = 0
        idx = self.train_indices[self.shuffled_train_index: self.shuffled_train_index + n]
        self.shuffled_train_index += n
        return {'xs': self.train_images[idx], 'labels': self.train_labels[idx]}

    def next_test_batch(self, n):
        if self.shuffled_test_index + n > len(self.test_indices):
            np.random.shuffle(self.test_indices)
            self.shuffled_test_index = 0
        idx = self.test_indices[self.shuffled_test_index: self.shuffled_test_index + n]
        self.shuffled_test_index += n
        return {'xs': self.test_images[idx], 'labels': self.test_labels[idx]}
