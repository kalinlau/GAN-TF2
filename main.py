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

""" Implement various GANs on MNIST with TF2.0 """

import os
import sys
from datetime import datetime

from absl import app, flags, logging
import tensorflow as tf

from models.utils import _MODELS, get_model
from models import GAN_H
from datasets import get_mnist

FLAGS = flags.FLAGS
FLAGS.alsologtostderr = True

flags.DEFINE_enum(
    name='model',
    default='GAN_H',
    enum_values=list(_MODELS.keys()),
    help='select models to use (default: GAN_H).')

flags.DEFINE_enum(
    name='mode',
    default='train',
    enum_values=['train', 'eval'],
    help='computation mode (Default: train).')

flags.DEFINE_integer(
    name='batch_size',
    default=64,
    lower_bound=4,
    help='Batch size to use (default: 64).'
)

flags.DEFINE_integer(
    name='epoch',
    default=25,
    lower_bound=1,
    help='Epochs to train the net.'
)


def main(argv):
    # argv passed by absl.
    del argv

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['NO_GCE_CHECK'] = 'true'

    timeshift = datetime.today().strftime('%Y%m%d-%H:%M:%S')
    workdir = f'{FLAGS.model}-{timeshift}'
    workdir = os.path.join('exps', workdir)

    logdir = os.path.join(workdir, 'logs/')
    imgdir = os.path.join(workdir, 'imgs/')
    ckptdir = os.path.join(workdir, 'ckpts/')

    tf.io.gfile.makedirs(logdir)
    tf.io.gfile.makedirs(imgdir)
    tf.io.gfile.makedirs(ckptdir)

    logging.set_verbosity(logging.DEBUG)

    with tf.io.gfile.GFile(os.path.join(logdir, 'logging.txt'), 'w') as gf:
        absl_handler = logging.get_absl_handler()
        absl_handler.python_handler.stream = gf
        logging.info(' '.join(sys.argv))

    # Data
    (ds_train, ds_test) = get_mnist(FLAGS.batch_size)

    # Model Compilation
    model = get_model(FLAGS.model)(
        imgdir=imgdir,
        logdir=logdir,
        ckptdir=ckptdir,
    )

    if FLAGS.mode == 'train':
        model.compile(
            d_optim=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5),
            g_optim=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5),
            loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        )

        model.fit(
            ds_train,
            epochs=FLAGS.epoch,
            callbacks=model.callbacks,
        )

    else:
        logging.debug('Not Implemented yet.')


if __name__ == '__main__':
    app.run(main)