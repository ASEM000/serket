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

    .. image:: ../_static/rgbtograyscale2d.png

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


def rgb_to_hsv(image):
    """Convert an RGB image to HSV."""
    # https://kornia.readthedocs.io/en/latest/_modules/kornia/color/hsv.html#rgb_to_hsv

    eps = jnp.finfo(image.dtype).eps
    maxc = jnp.max(image, axis=0, keepdims=True)
    argmaxc = jnp.argmax(image, axis=0, keepdims=True)
    minc = jnp.min(image, axis=0, keepdims=True)
    diff = maxc - minc

    diff = jnp.where(diff == 0, 1, diff)
    rc, gc, bc = jnp.split((maxc - image), 3, axis=0)

    h1 = bc - gc
    h2 = (rc - bc) + 2.0 * diff
    h3 = (gc - rc) + 4.0 * diff

    h = jnp.stack((h1, h2, h3), axis=0) / (diff + eps)
    h = jnp.take_along_axis(h, argmaxc[None], axis=0).squeeze(0)
    h = (h / 6.0) % 1.0
    h = 2.0 * jnp.pi * h
    # saturation
    s = diff / (maxc + eps)
    # value
    v = maxc
    return jnp.concatenate((h, s, v), axis=0)


def hsv_to_rgb(image):
    """Convert an image from HSV to RGB."""
    # https://kornia.readthedocs.io/en/latest/_modules/kornia/color/hsv.html#rgb_to_hsv
    c, _, _ = image.shape
    assert c == 3

    h = image[0] / (2 * jnp.pi)
    s = image[1]
    v = image[2]

    hi = jnp.floor(h * 6) % 6
    f = ((h * 6) % 6) - hi
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    indices = jnp.stack([hi, hi + 6, hi + 12], axis=0).astype(jnp.int32)
    out = jnp.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q))
    out = jnp.take_along_axis(out, indices, axis=0)
    return out


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


class RGBToHSV2D(sk.TreeClass):
    """Converts an RGB image to HSV.

    .. image:: ../_static/rgbtohsv2d.png

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> rgb_image = jnp.ones([3, 5, 5])
        >>> layer = sk.image.RGBToHSV2D()
        >>> hsv_image = layer(rgb_image)
        >>> hsv_image.shape
        (3, 5, 5)

    Reference:
        - https://kornia.readthedocs.io/en/latest/_modules/kornia/color/hsv.html
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, image: CHWArray) -> CHWArray:
        return rgb_to_hsv(image)

    @property
    def spatial_ndim(self) -> int:
        return 2


class HSVToRGB2D(sk.TreeClass):
    """Converts an HSV image to RGB.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> hsv_image = jnp.ones([3, 5, 5])
        >>> layer = sk.image.HSVToRGB2D()
        >>> rgb_image = layer(hsv_image)
        >>> rgb_image.shape
        (3, 5, 5)

    Reference:
        - https://kornia.readthedocs.io/en/latest/_modules/kornia/color/hsv.html
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, image: CHWArray) -> CHWArray:
        return hsv_to_rgb(image)

    @property
    def spatial_ndim(self) -> int:
        return 2
