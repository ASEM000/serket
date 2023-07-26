# Copyright 2023 Serket authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp

import serket as sk
from serket.nn.utils import positive_int_cb, validate_spatial_ndim


class HistogramEqualization2D(sk.TreeClass):
    """Apply histogram equalization to 2D spatial array channel wise

    Args:
        bins: number of bins. Defaults to 256.

    References:
        - https://en.wikipedia.org/wiki/Histogram_equalization
        - http://www.janeriksolem.net/histogram-equalization-with-python-and.html
        - https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_hist
        - https://stackoverflow.com/questions/28518684/histogram-equalization-of-grayscale-images-with-numpy
    """

    def __init__(self, bins: int = 256):
        self.bins = positive_int_cb(bins)

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        bins_count = self.bins
        hist, bins = jnp.histogram(x.flatten(), bins_count, density=True)
        cdf = hist.cumsum()
        cdf = (bins_count - 1) * cdf / cdf[-1]
        return jnp.interp(x.flatten(), bins[:-1], cdf).reshape(x.shape)

    @property
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        return 2


class PixelShuffle2D(sk.TreeClass):
    """Rearrange elements in a tensor.

    Args:
        upscale_factor: factor to increase spatial resolution by. accepts a
            single integer or a tuple of length 2. defaults to 1.

    References:
        - https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upscale_factor: int | tuple[int, int] = 1):
        if isinstance(upscale_factor, int):
            if upscale_factor < 1:
                raise ValueError("upscale_factor must be >= 1")

            self.upscale_factor = (upscale_factor, upscale_factor)
            return

        if isinstance(upscale_factor, tuple):
            if len(upscale_factor) != 2:
                raise ValueError("upscale_factor must be a tuple of length 2")
            if upscale_factor[0] < 1 or upscale_factor[1] < 1:
                raise ValueError("upscale_factor must be >= 1")
            self.upscale_factor = upscale_factor
            return

        raise ValueError("upscale_factor must be an integer or tuple of length 2")

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        channels = x.shape[0]

        sr, sw = self.upscale_factor
        oc = channels // (sr * sw)

        if not (channels % (sr * sw)) == 0:
            raise ValueError(f"{channels=} not divisible by {sr*sw}.")

        ih, iw = x.shape[1], x.shape[2]
        x = jnp.reshape(x, (sr, sw, oc, ih, iw))
        x = jnp.transpose(x, (2, 3, 0, 4, 1))
        x = jnp.reshape(x, (oc, ih * sr, iw * sw))
        return x

    @property
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        return 2
