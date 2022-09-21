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

""" Vanilla GANs: High-Level API """

import os
import sys

from absl import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .utils import register_model, save_image


@register_model()
class GAN_H(keras.Model):
    """Vanilla GAN implemented with Keras API"""
    def __init__(self, imgdir='./exps', logdir='./exps', ckptdir='./exps', latent_dim=62, **kwargs):
        super(GAN_H, self).__init__(**kwargs)
        # Generator
        # Same architecture with infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_Bn-FC7x7x128_Bn-(64)4dc2s_Bn-(1)4dc2s_Sigmoid
        self.generator= keras.Sequential(
            [
                keras.Input(shape=(latent_dim, )),
                layers.Dense(1024),
                layers.BatchNormalization(epsilon=1e-5,),
                layers.ReLU(),
                layers.Dense(7 * 7 * 128),
                layers.BatchNormalization(epsilon=1e-5,),
                layers.ReLU(),
                layers.Reshape((7, 7, 128)),
                layers.Conv2DTranspose(64, (4, 4), (2, 2), padding='same'),
                layers.BatchNormalization(epsilon=1e-5,),
                layers.ReLU(),
                layers.Conv2DTranspose(1, 4, 2, padding='same', activation='sigmoid'),
            ],
            name='generator',
        )

        # Discriminator
        # Same architecture with infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_Bn-FC1024_Bn-FC1_Sigmoid
        self.discriminator = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(64, 4, 2, padding='same', name='d_conv1'),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, 4, 2, padding='same', name='d_conv2'),
                layers.BatchNormalization(epsilon=1e-5, name='d_bn2'),
                layers.LeakyReLU(alpha=0.2),
                layers.Flatten(),
                layers.Dense(1024, name='d_fc3'),
                layers.BatchNormalization(epsilon=1e-5, name='d_bn3'),
                layers.LeakyReLU(alpha=0.2),
                layers.Dense(1, name='d_fc4', activation='sigmoid'),
            ],
            name='discriminator',
        )
        self.latent_dim = latent_dim
        self.callbacks = [
            SaveImageEpochEnd(imgdir, num_img=64, latent_dim=self.latent_dim),
            SaveModelWeights(ckptdir),
            keras.callbacks.TensorBoard(log_dir=logdir),
        ]

    def compile(self, d_optim, g_optim, loss_fn):
        super(GAN_H, self).compile()
        self.d_optimizer = d_optim
        self.g_optimizer = g_optim
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name='d_loss')

    @property
    def metrics(self):
        """Metrics update by model.fit()."""
        return [self.d_loss_metric]

    def train_step(self, imgs):
        """One forward / backward training step.

        Called by keras.Model.fit() automatically.
        """

        batch_size = tf.shape(imgs)[0]
        noise = tf.random.normal((batch_size, self.latent_dim))

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            g_imgs = self.generator(noise, training=True)

            d_real = self.discriminator(imgs, training=True)
            d_fake = self.discriminator(g_imgs, training=True)
            # g_imgs, d_fake, d_real = GAN.__call__((noise, imgs))

            g_loss = self.loss_fn(tf.ones_like(d_fake), d_fake)
            d_real_loss = self.loss_fn(tf.ones_like(d_real), d_real)
            d_fake_loss = self.loss_fn(tf.zeros_like(d_fake), d_fake)
            d_loss = d_real_loss + d_fake_loss

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_weights)
        )
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_weights)
        )

        self.d_loss_metric.update_state(d_loss)

        return {'d_loss': self.d_loss_metric.result()}

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

class SaveModelWeights(keras.callbacks.Callback):
    """Save model weights during training.

    Since there is no way currently to use Model.save() in TF2.9, save
    the weights instead. That's also the suggestion from
    https://stackoverflow.com/q/51806852/8069158
    """
    def __init__(self, ckptdir):
        self.ckptdir = ckptdir.rstrip('/')

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(
            f'{self.ckptdir}/epoch-{epoch}'
        )
