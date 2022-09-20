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
import os

from absl import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .utils import register_model, save_image

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

    def call(self, u):
        x = self.relu(self.bn1(self.fc1(u)))
        x = self.relu(self.bn2(self.fc2(x)))
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
        self.conv_1 = layers.Conv2D(64, 4, strides=(2, 2), \
            input_shape=inputs_shape, padding='same', name='conv1')
        self.conv_2 = layers.Conv2D(128, 4, strides=(2, 2), \
            padding='same', name='conv2')
        self.bn1 = layers.BatchNormalization(name='bn1')
        self.bn2 = layers.BatchNormalization(name='bn2')
        self.lrelu = layers.LeakyReLU(alpha=0.1)
        self.fc_1 = layers.Dense(1024)
        self.fc_2 = layers.Dense(1)

    def call(self, inputs):
        x = self.lrelu(self.conv_1(inputs))
        x = self.lrelu(self.bn1(self.conv_2(x)))
        x = layers.Flatten()(x)
        x = self.lrelu(self.bn2(self.fc_1(x)))
        x = self.fc_2(x)

        return keras.activations.sigmoid(x)

@register_model(name='gan_h')
class GAN_H(keras.Model):
    """Vanilla GAN implemented with Keras API"""
    def __init__(self, workdir, latent_dim=74):
        super(GAN_H, self).__init__()
        self.discriminator = Discriminator()
        self.generator = Generator(inputs_shape=(latent_dim,))
        self.latent_dim = latent_dim
        self.callbacks = [
            SaveImageEpochEnd(workdir, latent_dim=74),
        ]

    def compile(self, d_optim, g_optim, loss_fn):
        super(GAN_H, self).compile()
        self.d_optimizer = d_optim
        self.g_optimizer = g_optim
        self.loss_fn = loss_fn

    def train_step(self, imgs):
        """Forward / Backward pass in one batch.

        Called by keras.Model.fit() automatically.
        """
        # Sample latent space to get noise
        batch_size = tf.shape(imgs)[0]
        z = tf.random.normal((batch_size, self.latent_dim))

        # Generate fake images
        fake_imgs = self.generator(z)

        # Generate input to discriminator
        pair_imgs = tf.concat([imgs, fake_imgs], axis=0)
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Add noise term to label. IMPORTANT TRICK
        # labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Compute backprop of discriminator
        with tf.GradientTape() as tape:
            label_p = self.discriminator(pair_imgs)
            d_loss = self.loss_fn(labels, label_p)

        d_grad = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(d_grad, self.discriminator.trainable_weights)
        )

        # Compute backprop of generator
        mis_z = tf.random.normal((batch_size, self.latent_dim))
        mis_labels = tf.zeros(shape=(batch_size, 1))

        with tf.GradientTape() as tape:
            label_p = self.discriminator(self.generator(mis_z))
            g_loss = self.loss_fn(mis_labels, label_p)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights)
        )

        return {'d_loss': d_loss, 'g_loss': g_loss}

    def test_step(self, inputs):
        return super(GAN_H, self).test_step(self, inputs)

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class SaveImageEpochEnd(keras.callbacks.Callback):
    def __init__(self, workdir, num_img=10, latent_dim=74):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.workdir = workdir

    def on_epoch_end(self, epoch, logs=None):
        z = tf.random.normal(shape=(self.num_img, self.latent_dim))
        g_image = self.model.generator(z)

        save_image(
            g_image.numpy(),
            os.path.join(self.workdir, f'epoch-{epoch}.png')
        )



if __name__ == '__main__':
    pass