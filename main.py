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

from absl import app, flags, logging
import tensorflow as tf

from utils import _MODELS, get_model
from models import GAN

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


def main(argv):
    del argv

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['NO_GCE_CHECK'] = 'true'
    tf.config.experimental.set_visible_devices([], 'GPU')

    tf.io.gfile.makedirs('exps/' + FLAGS.model)

    with tf.io.gfile.GFile('exps/' + FLAGS.model + '/logs', 'w') as gf:
        absl_handler = logging.get_absl_handler()
        absl_handler.python_handler.stream = gf
        logging.info(' '.join(sys.argv))

    # Data

    # Model Compilation
    model = get_model(FLAGS.model)
    print(model)


if __name__ == '__main__':
    app.run(main)