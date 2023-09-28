# Copyright 2023 serket authors
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

# grayscale

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp

import serket as sk
from serket._src.utils import CHWArray, validate_spatial_nd

# gray


def rgb_to_grayscale(image: CHWArray, weights: jax.Array | None = None) -> CHWArray:
    """Converts an RGB image to grayscale.

    Args:
        image: RGB image.
        weights: Weights for each channel.
    """
    c, _, _ = image.shape
    assert c == 3

    if weights is None:
        weights = jnp.array([76, 150, 29]) / (1 if image.dtype == jnp.uint8 else 255.0)

    rw, gw, bw = weights
    r, g, b = jnp.split(image, 3, axis=0)
    return rw * r + gw * g + bw * b


def grayscale_to_rgb(image: CHWArray) -> CHWArray:
    """Converts a single channel image to RGB."""
    c, _, _ = image.shape
    assert c == 1
    return jnp.concatenate([image, image, image], axis=0)


class RGBToGrayscale2D(sk.TreeClass):
    """Converts a channel-first RGB image to grayscale.

    .. image:: ../_static/rgbtograyscale.png

    Args:
        weights: Weights for each channel.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> rgb_image = jnp.ones([3, 5, 5])
        >>> layer = sk.image.RGBToGrayscale2D()
        >>> gray_image = layer(rgb_image)
        >>> gray_image.shape
        (1, 5, 5)
    """

    def __init__(self, weights: jax.Array | None = None):
        self.weights = weights

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, image: CHWArray) -> CHWArray:
        return rgb_to_grayscale(image, self.weights)

    @property
    def spatial_ndim(self) -> int:
        return 2


class GrayscaleToRGB2D(sk.TreeClass):
    """Converts a grayscale image to RGB.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> gray_image = jnp.ones([1, 5, 5])
        >>> layer = sk.image.GrayscaleToRGB2D()
        >>> rgb_image = layer(gray_image)
        >>> rgb_image.shape
        (3, 5, 5)
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, image: CHWArray) -> CHWArray:
        return grayscale_to_rgb(image)

    @property
    def spatial_ndim(self) -> int:
        return 2
