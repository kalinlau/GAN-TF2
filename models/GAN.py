# coding=utf-8
# Copyright 2022, Jialin Liu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

""" Vanilla GANs """

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import register_model

class Generator(keras.layers.Layer):
    """Generator
    Same architecture as infoGAN (https://arxiv.org/abs/1606.03657)
    Architecture: fc1024_bn-fc7x7x128_bn-(64)4dc2s_bn-(1)4dc2s_s
    """
    def __init__(self, name='generator', inputs_shape=(128, ), **kwargs):
        super(Generator, self).__init__(name=name, **kwargs)
        self.fc1 = layers.Dense(1024, input_shape=inputs_shape, name='fc1')
        self.fc2 = layers.Dense(7 * 7 * 128, name='fc2')
        self.bn1 = layers.BatchNormalization(name='bn1')
        self.bn2 = layers.BatchNormalization(name='bn2')
        self.bn3 = layers.BatchNormalization(name='bn3')
        self.relu = keras.activations.relu
        self.dc1 = layers.Conv2DTranspose(64, 4, (2, 2), padding='same', name='dc1')
        self.dc2 = layers.Conv2DTranspose(1, 4, (2, 2), padding='same', name='dc1')

    def call(self, u, training=None):
        x = self.relu(self.bn1(self.fc1(u), training=training))
        x = self.relu(self.bn2(self.fc2(x), training=training))
        x = layers.Reshape((7, 7, 128))(x)
        x = self.relu(self.bn3(self.dc1(x)))
        x = self.dc2(x)

        return keras.activations.sigmoid(x)

class Discriminator(keras.layers.Layer):
    """Discriminator

    Same architecture as infoGAN (https://arxiv.org/abs/1606.03657)
    Architecture: (64)4c2s-(128)4c2s_bn-fc1024_bn-fc1_s
    """
    def __init__(self, name='discriminator', inputs_shape=(28, 28, 1), **kwargs):
        super(Discriminator, self).__init__(name=name, **kwargs)
        self.conv_1 = layers.Conv2D(64, 4, strides=(2, 2), input_shape=inputs_shape, padding='same', name='conv1')
        self.conv_2 = layers.Conv2D(128, 4, strides=(2, 2), padding='same', name='conv2')
        self.bn = layers.BatchNormalization()
        self.lrelu = layers.LeakyReLU(alpha=0.2)
        self.fc_1 = layers.Dense(1024)
        self.fc_2 = layers.Dense(1)

    def call(self, inputs, training=None):
        x = self.lrelu(self.conv_1(inputs))
        x = self.lrelu(self.bn(self.conv_2(x), training=training))
        x = layers.Flatten()(x)
        x = self.lrelu(self.bn(self.fc_1(x), training=training))
        x = self.fc_2(x)

        return keras.activations.sigmoid(x)

@register_model(name='gan_h')
class GAN_H(keras.Model):
    """Vanilla GAN implemented with Keras API"""
    def __init__(self) -> None:
        super().__init__()

    def build(self):
        pass

    def call(self):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}()'


if __name__ == '__main__':
    from absl import logging
    logging.set_verbosity(logging.DEBUG)
    inputs = keras.Input((128, ))
    x = layers.Dense(1024)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Dense(128 * 7 * 7)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Reshape((7, 7, 128))(x)
    x = layers.Conv2DTranspose(64, 4, (2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2DTranspose(1, 4, (2, 2), padding='same')(x)
    x = keras.activations.sigmoid(x)

    model = keras.Model(inputs, x)

    res = Generator()(tf.ones((64, 128)))
    logging.debug(res.shape)