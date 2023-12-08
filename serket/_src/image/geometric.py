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

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.ndimage import map_coordinates

import serket as sk
from serket._src.custom_transform import tree_eval
from serket._src.nn.linear import Identity
from serket._src.utils import (
    CHWArray,
    HWArray,
    IsInstance,
    Range,
    validate_spatial_ndim,
)


def affine_2d(array: HWArray, matrix: HWArray) -> HWArray:
    h, w = array.shape
    center = jnp.array((h // 2, w // 2))
    coords = jnp.indices((h, w)).reshape(2, -1) - center.reshape(2, 1)
    coords = matrix @ coords + center.reshape(2, 1)
    return map_coordinates(array, coords, order=1).reshape((h, w))


def horizontal_shear_2d(image: HWArray, angle: float) -> HWArray:
    """shear rows by an angle in degrees"""
    shear = jnp.tan(jnp.deg2rad(angle))
    matrix = jnp.array([[1, 0], [shear, 1]])
    return affine_2d(image, matrix)


def random_horizontal_shear_2d(
    key: jax.Array,
    image: jax.Array,
    range: tuple[float, float],
) -> jax.Array:
    """shear rows by an angle in degrees"""
    minval, maxval = range
    angle = jr.uniform(key=key, shape=(), minval=minval, maxval=maxval)
    return horizontal_shear_2d(image, angle)


def vertical_shear_2d(
    image: HWArray,
    angle: float,
) -> HWArray:
    """shear cols by an angle in degrees"""
    shear = jnp.tan(jnp.deg2rad(angle))
    matrix = jnp.array([[1, shear], [0, 1]])
    return affine_2d(image, matrix)


def random_vertical_shear_2d(
    key: jax.Array,
    image: HWArray,
    range: tuple[float, float],
) -> HWArray:
    """shear cols by an angle in degrees"""
    minval, maxval = range
    angle = jr.uniform(key=key, shape=(), minval=minval, maxval=maxval)
    return vertical_shear_2d(image, angle)


def rotate_2d(image: HWArray, angle: float) -> HWArray:
    """rotate an image by an angle in degrees in CCW direction."""
    θ = jnp.deg2rad(-angle)
    matrix = jnp.array([[jnp.cos(θ), -jnp.sin(θ)], [jnp.sin(θ), jnp.cos(θ)]])
    return affine_2d(image, matrix)


def random_rotate_2d(
    key: jax.Array,
    image: HWArray,
    range: tuple[float, float],
) -> HWArray:
    minval, maxval = range
    angle = jr.uniform(key=key, shape=(), minval=minval, maxval=maxval)
    return rotate_2d(image, angle)


def perspective_transform_2d(image: HWArray, coeffs: jax.Array) -> HWArray:
    """Apply a perspective transform to an image."""

    rows, cols = image.shape
    y, x = jnp.meshgrid(jnp.arange(rows), jnp.arange(cols), indexing="ij")
    a, b, c, d, e, f, g, h = coeffs
    w = g * x + h * y + 1.0
    x_prime = (a * x + b * y + c) / w
    y_prime = (d * x + e * y + f) / w
    coords = [y_prime.ravel(), x_prime.ravel()]
    return map_coordinates(image, coords, order=1).reshape(rows, cols)


def random_perspective_2d(
    key: jax.Array,
    image: HWArray,
    scale: float,
) -> HWArray:
    """Applies a random perspective transform to a channel-first image"""
    _, _ = image.shape
    a = e = 1.0
    b = d = 0.0
    c = f = 0.0  # no translation
    g, h = jr.uniform(key, shape=(2,), minval=-1e-4, maxval=1e-4) * scale
    coeffs = jnp.array([a, b, c, d, e, f, g, h])
    return perspective_transform_2d(image, coeffs)


def horizontal_translate_2d(image: HWArray, shift: int) -> HWArray:
    """Translate an image horizontally by a pixel value."""
    _, _ = image.shape
    if shift > 0:
        return jnp.zeros_like(image).at[:, shift:].set(image[:, :-shift])
    if shift < 0:
        return jnp.zeros_like(image).at[:, :shift].set(image[:, -shift:])
    return image


def vertical_translate_2d(image: HWArray, shift: int) -> HWArray:
    """Translate an image vertically by a pixel value."""
    _, _ = image.shape
    if shift > 0:
        return jnp.zeros_like(image).at[shift:, :].set(image[:-shift, :])
    if shift < 0:
        return jnp.zeros_like(image).at[:shift, :].set(image[-shift:, :])
    return image


def random_horizontal_translate_2d(key: jax.Array, image: HWArray) -> HWArray:
    _, w = image.shape
    shift = jr.randint(key, shape=(), minval=-w, maxval=w)
    return horizontal_translate_2d(image, shift)


def random_vertical_translate_2d(key: jax.Array, image: HWArray) -> HWArray:
    h, _ = image.shape
    shift = jr.randint(key, shape=(), minval=-h, maxval=h)
    return vertical_translate_2d(image, shift)


def wave_transform_2d(image: HWArray, length: float, amplitude: float) -> HWArray:
    """Transform an image with a sinusoidal wave."""
    _, _ = image.shape
    eps = jnp.finfo(image.dtype).eps
    ny, nx = jnp.indices(image.shape)
    sinx = nx + amplitude * jnp.sin(ny / (length + eps))
    cosy = ny + amplitude * jnp.cos(nx / (length + eps))
    return map_coordinates(image, [cosy, sinx], order=1)


def random_wave_transform_2d(
    key: jax.Array,
    image: HWArray,
    length_range: tuple[float, float],
    amplitude_range: tuple[float, float],
) -> HWArray:
    """Transform an image with a sinusoidal wave.

    Args:
        key: a random key.
        image: a 2D image to transform.
        length_range: a tuple of min length and max length to randdomly choose from.
        amplitude_range: a tuple of min amplitude and max amplitude to randdomly choose from.
    """
    k1, k2 = jr.split(key)
    l0, l1 = length_range
    length = jr.uniform(k1, shape=(), minval=l0, maxval=l1)
    a0, a1 = amplitude_range
    amplitude = jr.uniform(k2, shape=(), minval=a0, maxval=a1)
    return wave_transform_2d(image, length, amplitude)


class Rotate2D(sk.TreeClass):
    """Rotate_2d a 2D image by an angle in dgrees in CCW direction

    .. image:: ../_static/rotate2d.png

    Args:
        angle: angle to rotate_2d in degrees counter-clockwise direction.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.image.Rotate2D(90)(x))
        [[[ 5 10 15 20 25]
          [ 4  9 14 19 24]
          [ 3  8 13 18 23]
          [ 2  7 12 17 22]
          [ 1  6 11 16 21]]]
    """

    def __init__(self, angle: float):
        self.angle = angle

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        angle = jax.lax.stop_gradient(self.angle)
        return jax.vmap(rotate_2d, in_axes=(0, None))(image, angle)

    spatial_ndim: int = 2


class RandomRotate2D(sk.TreeClass):
    """Rotate_2d a 2D image by an angle in dgrees in CCW direction

    Args:
        range: a tuple of min angle and max angle to randdomly choose from.

    Note:
        - Use :func:`tree_eval` to replace this layer with :class:`Identity` during
          evaluation.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 17).reshape(1, 4, 4)
        >>> layer = sk.image.RandomRotate2D((10, 30))
        >>> eval_layer = sk.tree_eval(layer)
        >>> print(eval_layer(x))
        [[[ 1  2  3  4]
          [ 5  6  7  8]
          [ 9 10 11 12]
          [13 14 15 16]]]

    Example:
        >>> import serket as sk
        >>> import jax
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.image.RandomRotate2D((10, 30))(x, key=jax.random.PRNGKey(0)))
        [[[ 1  2  4  7  4]
          [ 4  6  9 11 11]
          [ 8 10 13 16 18]
          [10 15 17 20 22]
          [ 8 19 22 18 11]]]
    """

    def __init__(self, range: tuple[float, float] = (0.0, 360.0)):
        if not (
            isinstance(range, tuple)
            and len(range) == 2
            and isinstance(range[0], (int, float))
            and isinstance(range[1], (int, float))
        ):
            raise ValueError(f"`{range=}` must be a tuple of 2 floats/ints ")

        self.range = range

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray, *, key: jax.Array) -> CHWArray:
        range = jax.lax.stop_gradient(self.range)
        return jax.vmap(random_rotate_2d, in_axes=(None, 0, None))(key, image, range)

    spatial_ndim: int = 2


class HorizontalShear2D(sk.TreeClass):
    """Shear an image horizontally

    .. image:: ../_static/horizontalshear2d.png

    Args:
        angle: angle to rotate_2d in degrees counter-clockwise direction.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.image.HorizontalShear2D(45)(x))
        [[[ 0  0  1  2  3]
          [ 0  6  7  8  9]
          [11 12 13 14 15]
          [17 18 19 20  0]
          [23 24 25  0  0]]]
    """

    def __init__(self, angle: float):
        self.angle = angle

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        angle = jax.lax.stop_gradient(self.angle)
        return jax.vmap(horizontal_shear_2d, in_axes=(0, None))(image, angle)

    spatial_ndim: int = 2


class RandomHorizontalShear2D(sk.TreeClass):
    """Shear an image horizontally with random angle choice.

    Args:
        range: a tuple of min angle and max angle to randdomly choose from.

    Note:
        - Use :func:`tree_eval` to replace this layer with :class:`Identity` during
          evaluation.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 17).reshape(1, 4, 4)
        >>> layer = sk.image.RandomHorizontalShear2D((45, 45))
        >>> eval_layer = sk.tree_eval(layer)
        >>> print(eval_layer(x))
        [[[ 1  2  3  4]
          [ 5  6  7  8]
          [ 9 10 11 12]
          [13 14 15 16]]]

    Example:
        >>> import serket as sk
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.image.RandomHorizontalShear2D((45, 45))(x, key=jr.PRNGKey(0)))
        [[[ 0  0  1  2  3]
          [ 0  6  7  8  9]
          [11 12 13 14 15]
          [17 18 19 20  0]
          [23 24 25  0  0]]]
    """

    def __init__(self, range: tuple[float, float] = (0.0, 90.0)):
        if not (
            isinstance(range, tuple)
            and len(range) == 2
            and isinstance(range[0], (int, float))
            and isinstance(range[1], (int, float))
        ):
            raise ValueError(f"`{range=}` must be a tuple of 2 floats")
        self.range = range

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray, *, key: jax.Array) -> CHWArray:
        angle = jax.lax.stop_gradient(self.range)
        in_axes = (None, 0, None)
        return jax.vmap(random_horizontal_shear_2d, in_axes=in_axes)(key, image, angle)

    spatial_ndim: int = 2


class VerticalShear2D(sk.TreeClass):
    """Shear an image vertically

    .. image:: ../_static/verticalshear2d.png

    Args:
        angle: angle to rotate_2d in degrees counter-clockwise direction.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.image.VerticalShear2D(45)(x))
        [[[ 0  0  3  9 15]
          [ 0  2  8 14 20]
          [ 1  7 13 19 25]
          [ 6 12 18 24  0]
          [11 17 23  0  0]]]
    """

    def __init__(self, angle: float):
        self.angle = angle

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: jax.Array) -> jax.Array:
        angle = jax.lax.stop_gradient(self.angle)
        return jax.vmap(vertical_shear_2d, in_axes=(0, None))(image, angle)

    spatial_ndim: int = 2


class RandomVerticalShear2D(sk.TreeClass):
    """Shear an image vertically with random angle choice.

    Args:
        range: a tuple of min angle and max angle to randdomly choose from.

    Note:
        - Use :func:`tree_eval` to replace this layer with :class:`Identity` during
          evaluation.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 17).reshape(1, 4, 4)
        >>> layer = sk.image.RandomVerticalShear2D((45, 45))
        >>> eval_layer = sk.tree_eval(layer)
        >>> print(eval_layer(x))
        [[[ 1  2  3  4]
          [ 5  6  7  8]
          [ 9 10 11 12]
          [13 14 15 16]]]

    Example:
        >>> import serket as sk
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.image.RandomVerticalShear2D((45, 45))(x, key=jr.PRNGKey(0)))
        [[[ 0  0  3  9 15]
          [ 0  2  8 14 20]
          [ 1  7 13 19 25]
          [ 6 12 18 24  0]
          [11 17 23  0  0]]]
    """

    def __init__(self, range: tuple[float, float] = (0.0, 90.0)):
        if not (
            isinstance(range, tuple)
            and len(range) == 2
            and isinstance(range[0], (int, float))
            and isinstance(range[1], (int, float))
        ):
            raise ValueError(f"`{range=}` must be a tuple of 2 floats")
        self.range = range

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray, *, key: jax.Array) -> CHWArray:
        angle = jax.lax.stop_gradient(self.range)
        in_axes = (None, 0, None)
        return jax.vmap(random_vertical_shear_2d, in_axes=in_axes)(key, image, angle)

    spatial_ndim: int = 2


class RandomPerspective2D(sk.TreeClass):
    """Applies a random perspective transform to a channel-first image.

    .. image:: ../_static/randomperspective2d.png

    Args:
        scale: the scale of the random perspective transform. Higher scale will
            lead to higher degree of perspective transform. default to 1.0. 0.0
            means no perspective transform.

    Note:
        - Use :func:`tree_eval` to replace this layer with :class:`Identity` during
          evaluation.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 17).reshape(1, 4, 4)
        >>> layer = sk.image.RandomPerspective2D(100)
        >>> eval_layer = sk.tree_eval(layer)
        >>> print(eval_layer(x))
        [[[ 1  2  3  4]
          [ 5  6  7  8]
          [ 9 10 11 12]
          [13 14 15 16]]]
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray, *, key: jax.Array) -> CHWArray:
        scale = jax.lax.stop_gradient(self.scale)
        args = (key, image, scale)
        return jax.vmap(random_perspective_2d, in_axes=(None, 0, None))(*args)

    spatial_ndim: int = 2


@sk.autoinit
class HorizontalTranslate2D(sk.TreeClass):
    """Translate an image horizontally by a pixel value.

    .. image:: ../_static/horizontaltranslate2d.png

    Args:
        shift: The number of pixels to shift the image by.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.image.HorizontalTranslate2D(2)(x))
        [[[ 0  0  1  2  3]
          [ 0  0  6  7  8]
          [ 0  0 11 12 13]
          [ 0  0 16 17 18]
          [ 0  0 21 22 23]]]
    """

    shift: int = sk.field(on_setattr=[IsInstance(int)])

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        return jax.vmap(horizontal_translate_2d, in_axes=(0, None))(image, self.shift)

    spatial_ndim: int = 2


@sk.autoinit
class VerticalTranslate2D(sk.TreeClass):
    """Translate an image vertically by a pixel value.

    .. image:: ../_static/verticaltranslate2d.png

    Args:
        shift: The number of pixels to shift the image by.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.image.VerticalTranslate2D(2)(x))
        [[[ 0  0  0  0  0]
          [ 0  0  0  0  0]
          [ 1  2  3  4  5]
          [ 6  7  8  9 10]
          [11 12 13 14 15]]]
    """

    shift: int = sk.field(on_setattr=[IsInstance(int)])

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        return jax.vmap(vertical_translate_2d, in_axes=(0, None))(image, self.shift)

    spatial_ndim: int = 2


@sk.autoinit
class RandomHorizontalTranslate2D(sk.TreeClass):
    """Translate an image horizontally by a random pixel value.

    Note:
        - Use :func:`tree_eval` to replace this layer with :class:`Identity` during
          evaluation.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 17).reshape(1, 4, 4)
        >>> layer = sk.image.RandomHorizontalTranslate2D()
        >>> eval_layer = sk.tree_eval(layer)
        >>> print(eval_layer(x))
        [[[ 1  2  3  4]
          [ 5  6  7  8]
          [ 9 10 11 12]
          [13 14 15 16]]]

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.image.RandomHorizontalTranslate2D()(x, key=jr.PRNGKey(0)))
        [[[ 4  5  0  0  0]
          [ 9 10  0  0  0]
          [14 15  0  0  0]
          [19 20  0  0  0]
          [24 25  0  0  0]]]
    """

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray, *, key: jax.Array) -> CHWArray:
        return jax.vmap(random_horizontal_translate_2d, in_axes=(None, 0))(key, image)

    spatial_ndim: int = 2


class RandomVerticalTranslate2D(sk.TreeClass):
    """Translate an image vertically by a random pixel value.

    Note:
        - Use :func:`tree_eval` to replace this layer with :class:`Identity` during
          evaluation.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> x = jnp.arange(1, 17).reshape(1, 4, 4)
        >>> layer = sk.image.RandomVerticalTranslate2D()
        >>> eval_layer = sk.tree_eval(layer)
        >>> print(eval_layer(x))
        [[[ 1  2  3  4]
          [ 5  6  7  8]
          [ 9 10 11 12]
          [13 14 15 16]]]

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.image.RandomVerticalTranslate2D()(x, key=jr.PRNGKey(0)))
        [[[16 17 18 19 20]
          [21 22 23 24 25]
          [ 0  0  0  0  0]
          [ 0  0  0  0  0]
          [ 0  0  0  0  0]]]
    """

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray, *, key: jax.Array) -> CHWArray:
        return jax.vmap(random_vertical_translate_2d, in_axes=(None, 0))(key, image)

    spatial_ndim: int = 2


class HorizontalFlip2D(sk.TreeClass):
    """Flip channels left to right.

    .. image:: ../_static/horizontalflip2d.png

    Examples:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> x = jnp.arange(1,10).reshape(1,3, 3)
        >>> print(x)
        [[[1 2 3]
          [4 5 6]
          [7 8 9]]]

        >>> print(sk.image.HorizontalFlip2D()(x))
        [[[3 2 1]
          [6 5 4]
          [9 8 7]]]

    Reference:
        - https://github.com/deepmind/dm_pix/blob/master/dm_pix/_src/augment.py
    """

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        return jax.vmap(lambda x: jnp.flip(x, axis=1))(image)

    spatial_ndim: int = 2


@sk.autoinit
class RandomHorizontalFlip2D(sk.TreeClass):
    """Flip channels left to right with a probability of `rate`.

    .. image:: ../_static/horizontalflip2d.png

    Args:
        rate: The probability of flipping the image.

    Note:
        use :func:`tree_eval` to replace this layer with :class:`Identity` during

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> key = jax.random.PRNGKey(0)
        >>> print(sk.image.RandomHorizontalFlip2D(rate=1.0)(x, key=key))
        [[[ 5  4  3  2  1]
          [10  9  8  7  6]
          [15 14 13 12 11]
          [20 19 18 17 16]
          [25 24 23 22 21]]]
    """

    rate: float = sk.field(on_setattr=[IsInstance(float), Range(0.0, 1.0)])

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray, *, key: jax.Array) -> CHWArray:
        rate = jax.lax.stop_gradient(self.rate)
        prop = jax.random.bernoulli(key, rate)
        return jnp.where(prop, jax.vmap(lambda x: jnp.flip(x, axis=1))(image), image)

    spatial_ndim: int = 2


class VerticalFlip2D(sk.TreeClass):
    """Flip channels up to down.

    .. image:: ../_static/verticalflip2d.png

    Examples:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> x = jnp.arange(1,10).reshape(1,3, 3)
        >>> print(x)
        [[[1 2 3]
          [4 5 6]
          [7 8 9]]]

        >>> print(sk.image.VerticalFlip2D()(x))
        [[[7 8 9]
          [4 5 6]
          [1 2 3]]]

    Reference:
        - https://github.com/deepmind/dm_pix/blob/master/dm_pix/_src/augment.py
    """

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        return jax.vmap(lambda x: jnp.flip(x, axis=0))(image)

    spatial_ndim: int = 2


@sk.autoinit
class RandomVerticalFlip2D(sk.TreeClass):
    """Flip channels up to down with a probability of `rate`.

    .. image:: ../_static/verticalflip2d.png

    Args:
        rate: The probability of flipping the image.

    Note:
        use :func:`tree_eval` to replace this layer with :class:`Identity` during

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> key = jax.random.PRNGKey(0)
        >>> print(sk.image.RandomVerticalFlip2D(rate=1.0)(x, key=key))
        [[[21 22 23 24 25]
          [16 17 18 19 20]
          [11 12 13 14 15]
          [ 6  7  8  9 10]
          [ 1  2  3  4  5]]]
    """

    rate: float = sk.field(on_setattr=[IsInstance(float), Range(0.0, 1.0)])

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray, *, key: jax.Array) -> CHWArray:
        rate = jax.lax.stop_gradient(self.rate)
        prop = jax.random.bernoulli(key, rate)
        return jnp.where(prop, jax.vmap(lambda x: jnp.flip(x, axis=0))(image), image)

    spatial_ndim: int = 2


class WaveTransform2D(sk.TreeClass):
    """Apply a wave transform to an image.

    .. image:: ../_static/wavetransform2d.png

    Args:
        length: The length of the wave.
        amplitude: The amplitude of the wave.
    """

    def __init__(self, length: int, amplitude: float):
        self.length = length
        self.amplitude = amplitude

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        in_axes = (0, None, None)
        length, amplitude = jax.lax.stop_gradient((self.length, self.amplitude))
        return jax.vmap(wave_transform_2d, in_axes=in_axes)(image, length, amplitude)

    spatial_ndim: int = 2


class RandomWaveTransform2D(sk.TreeClass):
    """Apply a random wave transform to an image.

    .. image:: ../_static/wavetransform2d.png

    Args:
        length_range: The range of the length of the wave.
        amplitude_range: The range of the amplitude of the wave.

    Note:
        - Use :func:`tree_eval` to replace this layer with :class:`Identity` during
          evaluation.
    """

    def __init__(
        self,
        length_range: tuple[float, float],
        amplitude_range: tuple[float, float],
    ):
        self.length_range = length_range
        self.amplitude_range = amplitude_range

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray, *, key: jax.Array) -> CHWArray:
        in_axes = (None, 0, None, None)
        L, A = jax.lax.stop_gradient((self.length_range, self.amplitude_range))
        return jax.vmap(random_wave_transform_2d, in_axes=in_axes)(key, image, L, A)

    spatial_ndim: int = 2


@tree_eval.def_eval(RandomRotate2D)
@tree_eval.def_eval(RandomHorizontalFlip2D)
@tree_eval.def_eval(RandomVerticalFlip2D)
@tree_eval.def_eval(RandomHorizontalShear2D)
@tree_eval.def_eval(RandomVerticalShear2D)
@tree_eval.def_eval(RandomPerspective2D)
@tree_eval.def_eval(RandomHorizontalTranslate2D)
@tree_eval.def_eval(RandomVerticalTranslate2D)
@tree_eval.def_eval(RandomWaveTransform2D)
def _(_):
    return Identity()
