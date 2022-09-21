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
from models import GAN
from datasets import get_mnist

FLAGS = flags.FLAGS
FLAGS.alsologtostderr = True

flags.DEFINE_enum(
    name='model',
    default='gan_h',
    enum_values=list(_MODELS.keys()),
    help='select models to use (default: gan_h).')

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

flags.DEFINE_enum(
    name='verb',
    default='debug',
    enum_values=['debug', 'info', 'warn', 'error', 'fatal'],
    help='Level of logging verbosity (default: debug).'
)


def main(argv):
    # argv passed by absl.
    del argv

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['NO_GCE_CHECK'] = 'true'
    tf.config.experimental.set_visible_devices([], 'GPU')

    timeshift = datetime.today().strftime('%Y%m%d%H%M%S')
    workdir = f'{FLAGS.model}-{timeshift}'

    workdir = os.path.join('exps', workdir)

    tf.io.gfile.makedirs(workdir)

    if FLAGS.verb == 'info':
        logging.set_verbosity(logging.INFO)
    elif FLAGS.verb == 'warn':
        logging.set_verbosity(logging.WARNING)
    elif FLAGS.verb == 'error':
        logging.set_verbosity(logging.ERROR)
    elif FLAGS.verb == 'fatal':
        logging.set_verbosity(logging.FATAL)
    else:
        logging.set_verbosity(logging.DEBUG)

    with tf.io.gfile.GFile(os.path.join(workdir, 'logs'), 'w') as gf:
        absl_handler = logging.get_absl_handler()
        absl_handler.python_handler.stream = gf
        logging.info(' '.join(sys.argv))

    # Data
    (ds_train, ds_test) = get_mnist(FLAGS.batch_size)

    # Model Compilation
    model = get_model(FLAGS.model)(workdir)

    if FLAGS.mode == 'train':
        model.compile(
            d_optim=tf.keras.optimizers.Adam(learning_rate=1e-4),
            g_optim=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss_fn=tf.keras.losses.BinaryCrossentropy(),
        )

        model.fit(
            ds_train.take(100),
            epochs=20,
            callbacks=model.callbacks,
        )
    else:
        logging.debug('Not Implemented yet.')


if __name__ == '__main__':
    app.run(main)