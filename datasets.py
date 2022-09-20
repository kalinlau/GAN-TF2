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

"""Data Preprocessing"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils

gcs_utils._is_gcs_disabled = True

def get_mnist(batch_size=64):
    def norm_and_remove(img, label):
        """Normalize and remove labels

        Normalize pixel into [0, 1), and remove labels since generative model
        doesn't need it.

        Note:
            input signature of map_func is determined by the structure of each
            element in this dataset. Check:

            https://www.tensorflow.org/api_docs/python/tf/data/Dataset?hl=en#map
        """
        return tf.cast(img, tf.float32) / 255.0

    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        with_info=True,
        as_supervised=True,
        data_dir='./data',
        try_gcs=False,
    )

    ds_train = ds_train.map(norm_and_remove, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(norm_and_remove, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.cache()
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return (ds_train, ds_test)

if __name__ == '__main__':
    # from tensorflow_datasets.core.utils import gcs_utils
    # gcs_utils.gcs_dataset_info_files = lambda *args, **kwargs: None
    # gcs_utils.is_dataset_on_gcs = lambda *args, **kwargs: False

    (ds_train, ds_test) = get_mnist()