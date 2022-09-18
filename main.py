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

from absl import app, flags
import tensorflow as tf

from utils import _MODELS
from models import GAN

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    name='model',
    default='GAN',
    enum_values=list(_MODELS.keys()),
    help='select models to use (default: GAN).')


def main(argv):
    del argv

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['NO_GCE_CHECK'] = 'true'


if __name__ == '__main__':
    app.run(main)