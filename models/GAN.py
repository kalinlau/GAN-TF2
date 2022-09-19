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

from utils import register_model

class G(keras.layers.Layer):
    """Generator"""
    def __init__(self):
        pass

    def call(self, x):
        pass

class D(keras.layers.Layer):
    """Discriminator"""
    def __init__(self):
        pass

    def call(self):
        return

@register_model()
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