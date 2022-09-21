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

"""Utilities."""

import math
from PIL import Image
import numpy as np

_MODELS = {}

def register_model(cls=None, *, name=None):
    def _register(cls):
        if name is None:
            label = cls.__name__
        else:
            label = name
        if label not in _MODELS:
            _MODELS[label] = cls
        else:
            raise ValueError(f'{label} has been registered yet.')
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

def get_model(name):
    return _MODELS[name]

# https://github.com/yang-song/score_sde/blob/main/utils.py#L51
def save_image(ndarray, fp, nrow=8, padding=2, pad_value=0.3, format=None):
    """Make a grid of images and save it into an image file.

    Pixel values are assumed to be within [0, 1].

    Args:
        ndarray (array_like): 4D mini-batch images of shape (B x H x W x C).
        fp: A filename(string) or file object.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.
        format(Optional):  If omitted, the format to use is determined from the
            filename extension. If a file object was used instead of a filename,
            this parameter should always be used.
    """
    if not (isinstance(ndarray, np.ndarray) or (isinstance(ndarray, list) and all(isinstance(t, np.ndarray) for t in ndarray))):
        raise TypeError("array_like of tensors expected, got {}".format( type(ndarray)))

    ndarray = np.asarray(ndarray)

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = np.concatenate((ndarray, ndarray, ndarray), -1)

    # make the mini-batch of images into a grid
    nmaps = ndarray.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] + padding)
    num_channels = ndarray.shape[3]
    grid = np.full(
            (height * ymaps + padding, width * xmaps + padding, num_channels),
            pad_value
        ).astype(np.float32)

    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[y * height + padding:(y + 1) * height, x * width + padding:(x + 1) * width] = ndarray[k]
            k += 1

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = np.clip(grid * 255.0 + 0.5, 0, 255).astype(np.uint8)

    im = Image.fromarray(ndarr.copy())
    im.save(fp, format=format)



if __name__ == '__main__':
    pass