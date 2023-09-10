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
from typing_extensions import Annotated

import serket as sk
from serket._src.custom_transform import tree_eval
from serket._src.nn.linear import Identity
from serket._src.utils import IsInstance, validate_spatial_nd


def affine_2d(
    array: Annotated[jax.Array, "HW"], matrix: Annotated[jax.Array, "HW"]
) -> Annotated[jax.Array, "HW"]:
    h, w = array.shape
    center = jnp.array((h // 2, w // 2))
    coords = jnp.indices((h, w)).reshape(2, -1) - center.reshape(2, 1)
    coords = matrix @ coords + center.reshape(2, 1)
    return map_coordinates(array, coords, order=1).reshape((h, w))


def horizontal_shear_2d(
    image: Annotated[jax.Array, "HW"], angle: float
) -> Annotated[jax.Array, "HW"]:
    """shear rows by an angle in degrees"""
    shear = jnp.tan(jnp.deg2rad(angle))
    matrix = jnp.array([[1, 0], [shear, 1]])
    return affine_2d(image, matrix)


def random_horizontal_shear_2d(
    image: jax.Array,
    angle_range: tuple[float, float],
    key: jax.random.KeyArray,
) -> jax.Array:
    """shear rows by an angle in degrees"""
    minval, maxval = angle_range
    angle = jr.uniform(key=key, shape=(), minval=minval, maxval=maxval)
    return horizontal_shear_2d(image, angle)


def vertical_shear_2d(
    image: Annotated[jax.Array, "HW"],
    angle: float,
) -> Annotated[jax.Array, "HW"]:
    """shear cols by an angle in degrees"""
    shear = jnp.tan(jnp.deg2rad(angle))
    matrix = jnp.array([[1, shear], [0, 1]])
    return affine_2d(image, matrix)


def random_vertical_shear_2d(
    image: Annotated[jax.Array, "HW"],
    angle_range: tuple[float, float],
    key: jax.random.KeyArray,
) -> Annotated[jax.Array, "HW"]:
    """shear cols by an angle in degrees"""
    minval, maxval = angle_range
    angle = jr.uniform(key=key, shape=(), minval=minval, maxval=maxval)
    return vertical_shear_2d(image, angle)


def rotate_2d(
    image: Annotated[jax.Array, "HW"], angle: float
) -> Annotated[jax.Array, "HW"]:
    """rotate an image by an angle in degrees in CCW direction."""
    θ = jnp.deg2rad(-angle)
    matrix = jnp.array([[jnp.cos(θ), -jnp.sin(θ)], [jnp.sin(θ), jnp.cos(θ)]])
    return affine_2d(image, matrix)


def random_rotate_2d(
    image: Annotated[jax.Array, "HW"],
    angle_range: tuple[float, float],
    key: jax.random.KeyArray,
) -> Annotated[jax.Array, "HW"]:
    minval, maxval = angle_range
    angle = jr.uniform(key=key, shape=(), minval=minval, maxval=maxval)
    return rotate_2d(image, angle)


def pixelate_2d(
    image: Annotated[jax.Array, "HW"],
    scale: int = 16,
) -> Annotated[jax.Array, "HW"]:
    """Return a pixelated image by downsizing and upsizing"""
    dtype = image.dtype
    h, w = image.shape
    image = image.astype(jnp.float32)
    image = jax.image.resize(image, (h // scale, w // scale), method="linear")
    image = jax.image.resize(image, (h, w), method="linear")
    image = image.astype(dtype)
    return image


def perspective_transform_2d(
    image: Annotated[jax.Array, "HW"],
    coeffs: jax.Array,
) -> Annotated[jax.Array, "HW"]:
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
    image: Annotated[jax.Array, "HW"],
    key: jax.random.KeyArray,
    scale: float = 1.0,
) -> Annotated[jax.Array, "HW"]:
    """Applies a random perspective transform to a channel-first image"""
    _, _ = image.shape
    a = e = 1.0
    b = d = 0.0
    c = f = 0.0  # no translation
    g, h = jr.uniform(key, shape=(2,), minval=-1e-4, maxval=1e-4) * scale
    coeffs = jnp.array([a, b, c, d, e, f, g, h])
    return perspective_transform_2d(image, coeffs)


def solarize_2d(
    image: Annotated[jax.Array, "HW"],
    threshold: float | int,
    max_val: float | int,
) -> Annotated[jax.Array, "HW"]:
    """Inverts all values above a given threshold."""
    _, _ = image.shape
    return jnp.where(image < threshold, image, max_val - image)


def horizontal_translate_2d(
    image: Annotated[jax.Array, "HW"],
    shift: int,
) -> Annotated[jax.Array, "HW"]:
    """Translate an image horizontally by a pixel value."""
    _, _ = image.shape
    if shift > 0:
        return jnp.zeros_like(image).at[:, shift:].set(image[:, :-shift])
    if shift < 0:
        return jnp.zeros_like(image).at[:, :shift].set(image[:, -shift:])
    return image


def vertical_translate_2d(
    image: Annotated[jax.Array, "HW"],
    shift: int,
) -> Annotated[jax.Array, "HW"]:
    """Translate an image vertically by a pixel value."""
    _, _ = image.shape
    if shift > 0:
        return jnp.zeros_like(image).at[shift:, :].set(image[:-shift, :])
    if shift < 0:
        return jnp.zeros_like(image).at[:shift, :].set(image[-shift:, :])
    return image


def random_horizontal_translate_2d(
    image: Annotated[jax.Array, "HW"], key: jr.KeyArray
) -> Annotated[jax.Array, "HW"]:
    _, w = image.shape
    shift = jr.randint(key, shape=(), minval=-w, maxval=w)
    return horizontal_translate_2d(image, shift)


def random_vertical_translate_2d(
    image: Annotated[jax.Array, "HW"],
    key: jr.KeyArray,
) -> Annotated[jax.Array, "HW"]:
    h, _ = image.shape
    shift = jr.randint(key, shape=(), minval=-h, maxval=h)
    return vertical_translate_2d(image, shift)


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

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        angle = jax.lax.stop_gradient(self.angle)
        return jax.vmap(rotate_2d, in_axes=(0, None))(x, angle)

    @property
    def spatial_ndim(self) -> int:
        return 2


class RandomRotate2D(sk.TreeClass):
    """Rotate_2d a 2D image by an angle in dgrees in CCW direction

    Args:
        angle_range: a tuple of min angle and max angle to randdomly choose from.

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

    def __init__(self, angle_range: tuple[float, float] = (0.0, 360.0)):
        if not (
            isinstance(angle_range, tuple)
            and len(angle_range) == 2
            and isinstance(angle_range[0], (int, float))
            and isinstance(angle_range[1], (int, float))
        ):
            raise ValueError(f"`{angle_range=}` must be a tuple of 2 floats/ints ")

        self.angle_range = angle_range

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(
        self,
        x: jax.Array,
        key: jax.random.KeyArray = jax.random.PRNGKey(0),
    ) -> jax.Array:
        angle_range = jax.lax.stop_gradient(self.angle_range)
        return jax.vmap(random_rotate_2d, in_axes=(0, None, None))(x, angle_range, key)

    @property
    def spatial_ndim(self) -> int:
        return 2


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

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        angle = jax.lax.stop_gradient(self.angle)
        return jax.vmap(horizontal_shear_2d, in_axes=(0, None))(x, angle)

    @property
    def spatial_ndim(self) -> int:
        return 2


class RandomHorizontalShear2D(sk.TreeClass):
    """Shear an image horizontally with random angle choice.

    Args:
        angle_range: a tuple of min angle and max angle to randdomly choose from.

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
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.image.RandomHorizontalShear2D((45, 45))(x))
        [[[ 0  0  1  2  3]
          [ 0  6  7  8  9]
          [11 12 13 14 15]
          [17 18 19 20  0]
          [23 24 25  0  0]]]
    """

    def __init__(self, angle_range: tuple[float, float] = (0.0, 90.0)):
        if not (
            isinstance(angle_range, tuple)
            and len(angle_range) == 2
            and isinstance(angle_range[0], (int, float))
            and isinstance(angle_range[1], (int, float))
        ):
            raise ValueError(f"`{angle_range=}` must be a tuple of 2 floats")
        self.angle_range = angle_range

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(
        self,
        x: jax.Array,
        key: jax.random.KeyArray = jax.random.PRNGKey(0),
    ) -> jax.Array:
        angle = jax.lax.stop_gradient(self.angle_range)
        in_axes = (0, None, None)
        return jax.vmap(random_horizontal_shear_2d, in_axes=in_axes)(x, angle, key)

    @property
    def spatial_ndim(self) -> int:
        return 2


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

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        angle = jax.lax.stop_gradient(self.angle)
        return jax.vmap(vertical_shear_2d, in_axes=(0, None))(x, angle)

    @property
    def spatial_ndim(self) -> int:
        return 2


class RandomVerticalShear2D(sk.TreeClass):
    """Shear an image vertically with random angle choice.

    Args:
        angle_range: a tuple of min angle and max angle to randdomly choose from.

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
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.image.RandomVerticalShear2D((45, 45))(x))
        [[[ 0  0  3  9 15]
          [ 0  2  8 14 20]
          [ 1  7 13 19 25]
          [ 6 12 18 24  0]
          [11 17 23  0  0]]]
    """

    def __init__(self, angle_range: tuple[float, float] = (0.0, 90.0)):
        if not (
            isinstance(angle_range, tuple)
            and len(angle_range) == 2
            and isinstance(angle_range[0], (int, float))
            and isinstance(angle_range[1], (int, float))
        ):
            raise ValueError(f"`{angle_range=}` must be a tuple of 2 floats")
        self.angle_range = angle_range

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(
        self,
        x: jax.Array,
        key: jax.random.KeyArray = jax.random.PRNGKey(0),
    ) -> jax.Array:
        angle = jax.lax.stop_gradient(self.angle_range)
        in_axes = (0, None, None)
        return jax.vmap(random_vertical_shear_2d, in_axes=in_axes)(x, angle, key)

    @property
    def spatial_ndim(self) -> int:
        return 2


class Pixelate2D(sk.TreeClass):
    """Pixelate an image by upsizing and downsizing an image

    .. image:: ../_static/pixelate2d.png

    Args:
        scale: the scale to which the image will be downsized before being upsized
            to the original shape. for example, ``scale=2`` means the image will
            be downsized to half of its size before being updsized to the original
            size. Higher scale will lead to higher degree of pixelation.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.image.Pixelate2D(2)(x))  # doctest: +SKIP
        [[[ 7  7  8  8  9]
          [ 8  8  9  9 10]
          [12 12 13 13 14]
          [16 16 17 17 18]
          [17 17 18 18 19]]]
    """

    def __init__(self, scale: int):
        if not isinstance(scale, int) and scale > 0:
            raise ValueError(f"{scale=} must be a positive int")
        self.scale = scale

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        scale = jax.lax.stop_gradient(self.scale)
        return jax.vmap(pixelate_2d, in_axes=(0, None))(x, scale)

    @property
    def spatial_ndim(self) -> int:
        return 2


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

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax
        >>> x, y = jnp.meshgrid(jnp.linspace(-1, 1, 30), jnp.linspace(-1, 1, 30))
        >>> d = jnp.sqrt(x**2 + y**2)
        >>> mask = d < 1
        >>> print(mask.astype(int))
        [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0]
         [0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0]
         [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0]
         [0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0]
         [0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0]
         [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
         [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]
         [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
         [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
         [0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0]
         [0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0]
         [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0]
         [0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0]
         [0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
        >>> layer = sk.image.RandomPerspective2D(100)
        >>> key = jax.random.PRNGKey(10)
        >>> out = layer(mask[None], key=key)[0]
        >>> print(out.astype(int))
        [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0]
         [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0]
         [0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0]
         [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0]
         [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0]
         [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0]
         [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        scale = jax.lax.stop_gradient(self.scale)
        return jax.vmap(random_perspective_2d, in_axes=(0, None, None))(x, key, scale)

    @property
    def spatial_ndim(self) -> int:
        return 2


@sk.autoinit
class Solarize2D(sk.TreeClass):
    """Inverts all values above a given threshold.

    .. image:: ../_static/solarize2d.png

    Args:
        threshold: The threshold value above which to invert.
        max_val: The maximum value of the image. e.g. 255 for uint8 images.
            1.0 for float images. default: 1.0

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> layer = sk.image.Solarize2D(threshold=10, max_val=25)
        >>> print(layer(x))
        [[[ 1  2  3  4  5]
          [ 6  7  8  9 15]
          [14 13 12 11 10]
          [ 9  8  7  6  5]
          [ 4  3  2  1  0]]]

    Reference:
        - https://github.com/tensorflow/models/blob/v2.13.1/official/vision/ops/augment.py#L804-L809
    """

    threshold: float
    max_val: float = 1.0

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        threshold, max_val = jax.lax.stop_gradient((self.threshold, self.max_val))
        return jax.vmap(solarize_2d, in_axes=(0, None, None))(x, threshold, max_val)

    @property
    def spatial_ndim(self) -> int:
        return 2


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

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.vmap(horizontal_translate_2d, in_axes=(0, None))(x, self.shift)

    @property
    def spatial_ndim(self) -> int:
        return 2


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

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.vmap(vertical_translate_2d, in_axes=(0, None))(x, self.shift)

    @property
    def spatial_ndim(self) -> int:
        return 2


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
        >>> print(sk.image.RandomHorizontalTranslate2D()(x))
        [[[ 4  5  0  0  0]
          [ 9 10  0  0  0]
          [14 15  0  0  0]
          [19 20  0  0  0]
          [24 25  0  0  0]]]
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(
        self,
        x: jax.Array,
        key: jr.KeyArray = jr.PRNGKey(0),
    ) -> jax.Array:
        return jax.vmap(random_horizontal_translate_2d, in_axes=(0, None))(x, key)

    @property
    def spatial_ndim(self) -> int:
        return 2


class RandomVerticalTranslate2D(sk.TreeClass):
    """Translate an image vertically by a random pixel value.

    Note:
        - Use :func:`tree_eval` to replace this layer with :class:`Identity` during
          evaluation.

        >>> import serket as sk
        >>> import jax.numpy as jnp
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
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.image.RandomVerticalTranslate2D()(x))
        [[[16 17 18 19 20]
          [21 22 23 24 25]
          [ 0  0  0  0  0]
          [ 0  0  0  0  0]
          [ 0  0  0  0  0]]]
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        return jax.vmap(random_vertical_translate_2d, in_axes=(0, None))(x, key)

    @property
    def spatial_ndim(self) -> int:
        return 2


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

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.vmap(lambda x: jnp.flip(x, axis=1))(x)

    @property
    def spatial_ndim(self) -> int:
        return 2


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

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.vmap(lambda x: jnp.flip(x, axis=0))(x)

    @property
    def spatial_ndim(self) -> int:
        return 2


@tree_eval.def_eval(RandomRotate2D)
@tree_eval.def_eval(RandomHorizontalShear2D)
@tree_eval.def_eval(RandomVerticalShear2D)
@tree_eval.def_eval(RandomPerspective2D)
@tree_eval.def_eval(RandomHorizontalTranslate2D)
@tree_eval.def_eval(RandomVerticalTranslate2D)
def random_image_transform(_):
    return Identity()
