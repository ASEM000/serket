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
from typing_extensions import Annotated

import serket as sk
from serket._src.custom_transform import tree_eval
from serket._src.nn.linear import Identity
from serket._src.utils import IsInstance, Range, validate_spatial_nd


def pixel_shuffle_3d(
    array: Annotated[jax.Array, "CHW"],
    upscale_factor: tuple[int, int],
) -> Annotated[jax.Array, "CHW"]:
    """Rearrange elements in a tensor."""
    channels, _, _ = array.shape

    sr, sw = upscale_factor
    oc = channels // (sr * sw)

    if not (channels % (sr * sw)) == 0:
        raise ValueError(f"{channels=} not divisible by {sr*sw}.")

    ih, iw = array.shape[1], array.shape[2]
    array = jnp.reshape(array, (sr, sw, oc, ih, iw))
    array = jnp.transpose(array, (2, 3, 0, 4, 1))
    array = jnp.reshape(array, (oc, ih * sr, iw * sw))
    return array


def solarize_2d(
    image: Annotated[jax.Array, "HW"],
    threshold: float | int,
    max_val: float | int,
) -> Annotated[jax.Array, "HW"]:
    """Inverts all values above a given threshold."""
    _, _ = image.shape
    return jnp.where(image < threshold, image, max_val - image)


def adjust_contrast_2d(image: Annotated[jax.Array, "HW"], contrast_factor: float):
    """Adjusts the contrast of an image by scaling the pixel values by a factor.

    Args:
        array: input array
        contrast_factor: contrast factor to adust the contrast by.
    """
    _, _ = image.shape
    μ = jnp.mean(image, keepdims=True)
    return (contrast_factor * (image - μ) + μ).astype(image.dtype)


def random_contrast_2d(
    array: Annotated[jax.Array, "HW"],
    contrast_range: tuple[float, float],
    key: jr.KeyArray = jr.PRNGKey(0),
) -> Annotated[jax.Array, "HW"]:
    """Randomly adjusts the contrast of an image by scaling the pixel values by a factor."""
    _, _ = array.shape
    minval, maxval = contrast_range
    contrast_factor = jr.uniform(key=key, shape=(), minval=minval, maxval=maxval)
    return adjust_contrast_2d(array, contrast_factor)


def pixelate_2d(
    image: Annotated[jax.Array, "HW"], scale: int = 16
) -> Annotated[jax.Array, "HW"]:
    """Return a pixelated image by downsizing and upsizing"""
    dtype = image.dtype
    h, w = image.shape
    image = image.astype(jnp.float32)
    image = jax.image.resize(image, (h // scale, w // scale), method="linear")
    image = jax.image.resize(image, (h, w), method="linear")
    image = image.astype(dtype)
    return image


@ft.partial(jax.jit, inline=True, static_argnums=1)
def jigsaw_2d(
    image: Annotated[jax.Array, "HW"],
    tiles: int = 1,
    key: jr.KeyArray = jr.PRNGKey(0),
) -> Annotated[jax.Array, "HW"]:
    """Jigsaw channel-first image"""
    height, width = image.shape
    tile_height = height // tiles
    tile_width = width // tiles
    image_ = image[: height - height % tiles, : width - width % tiles]
    image_ = image_.reshape(tiles, tile_height, tiles, tile_width)
    image_ = image_.transpose(0, 2, 1, 3)  # tiles, tiles, tile_height, tile_width
    image_ = image_.reshape(-1, tile_height, tile_width)
    indices = jr.permutation(key, len(image_))
    image_ = jax.vmap(lambda x: image_[x])(indices)
    image_ = image_.reshape(tiles, tiles, tile_height, tile_width)
    image_ = image_.transpose(0, 2, 1, 3)  # tiles, tiles, tile_height, tile_width
    image_ = image_.reshape(tiles * tile_height, tiles * tile_width)
    image = image.at[: height - height % tiles, : width - width % tiles].set(image_)
    return image


def posterize_2d(
    image: Annotated[jax.Array, "HW"],
    bits: int,
) -> Annotated[jax.Array, "HW"]:
    """Reduce the number of bits for each color channel.

    Args:
        image: The image to posterize.
        bits: The number of bits to keep for each channel (1-8).

    Reference:
        - https://github.com/tensorflow/models/blob/v2.13.1/official/vision/ops/augment.py#L859-L862
        - https://github.com/python-pillow/Pillow/blob/6651a3143621181d94cc92d36e1490721ef0b44f/src/PIL/ImageOps.py#L547
    """
    shift = 8 - bits
    return jnp.left_shift(jnp.right_shift(image, shift), shift)


class PixelShuffle2D(sk.TreeClass):
    """Rearrange elements in a tensor.

    .. image:: ../_static/pixelshuffle2d.png

    Args:
        upscale_factor: factor to increase spatial resolution by. accepts a
            single integer or a tuple of length 2. defaults to 1.

    Reference:
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

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        return pixel_shuffle_3d(x, self.upscale_factor)

    @property
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        return 2


@sk.autoinit
class AdjustContrast2D(sk.TreeClass):
    """Adjusts the contrast of an 2D input by scaling the pixel values by a factor.

    .. image:: ../_static/adjustcontrast2d.png

    Args:
        contrast_factor: contrast factor to adust the contrast by. Defaults to 1.0.

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/image/adjust_contrast
        - https://github.com/deepmind/dm_pix/blob/master/dm_pix/_src/augment.py
    """

    contrast_factor: float = 1.0

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        contrast_factor = jax.lax.stop_gradient(self.contrast_factor)
        return jax.vmap(adjust_contrast_2d, in_axes=(0, None))(x, contrast_factor)

    @property
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        return 2


class RandomContrast2D(sk.TreeClass):
    """Randomly adjusts the contrast of an 1D input by scaling the pixel values by a factor.

    Args:
        contrast_range: contrast range to adust the contrast by. Defaults to (0.5, 1).

    Note:
        - Use :func:`tree_eval` to replace this layer with :class:`Identity` during
          evaluation.

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/image/adjust_contrast
        - https://github.com/deepmind/dm_pix/blob/master/dm_pix/_src/augment.py
    """

    def __init__(self, contrast_range: tuple[float, float] = (0.5, 1)):
        if not (
            isinstance(contrast_range, tuple)
            and len(contrast_range) == 2
            and contrast_range[0] <= contrast_range[1]
        ):
            raise ValueError(
                "`contrast_range` must be a tuple of two floats, "
                "with the first one smaller than the second one."
            )

        self.contrast_range = contrast_range

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, *, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        contrast_range = jax.lax.stop_gradient(self.contrast_range)
        in_axes = (0, None, None)
        return jax.vmap(random_contrast_2d, in_axes=in_axes)(x, contrast_range, key)

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
class Posterize2D(sk.TreeClass):
    """Reduce the number of bits for each color channel.

    .. image:: ../_static/posterize2d.png

    Args:
        bits: The number of bits to keep for each channel (1-8).

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> layer = sk.image.Posterize2D(4)
        >>> x = jnp.arange(1, 51).reshape(2, 5, 5)
        >>> print(x)
        [[[ 1  2  3  4  5]
          [ 6  7  8  9 10]
          [11 12 13 14 15]
          [16 17 18 19 20]
          [21 22 23 24 25]]
         [[26 27 28 29 30]
          [31 32 33 34 35]
          [36 37 38 39 40]
          [41 42 43 44 45]
          [46 47 48 49 50]]]
        >>> print(layer(x))
        [[[ 0  0  0  0  0]
          [ 0  0  0  0  0]
          [ 0  0  0  0  0]
          [16 16 16 16 16]
          [16 16 16 16 16]]
         [[16 16 16 16 16]
          [16 32 32 32 32]
          [32 32 32 32 32]
          [32 32 32 32 32]
          [32 32 48 48 48]]]

    Reference:
        - https://www.tensorflow.org/api_docs/python/tfm/vision/augment/posterize
        - https://github.com/python-pillow/Pillow/blob/main/src/PIL/ImageOps.py#L547
    """

    bits: int = sk.field(on_setattr=[IsInstance(int), Range(1, 8)])

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        bits = jax.lax.stop_gradient(self.bits)
        return jax.vmap(posterize_2d, in_axes=(0, None))(x, bits)

    @property
    def spatial_ndim(self) -> int:
        return 2


@sk.autoinit
class JigSaw2D(sk.TreeClass):
    """Mixes up tiles of an image.

    .. image:: ../_static/jigsaw2d.png

    Args:
        tiles: number of tiles per side

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 17).reshape(1, 4, 4)
        >>> print(x)
        [[[ 1  2  3  4]
          [ 5  6  7  8]
          [ 9 10 11 12]
          [13 14 15 16]]]
        >>> print(sk.image.JigSaw2D(2)(x))
        [[[ 9 10  3  4]
          [13 14  7  8]
          [11 12  1  2]
          [15 16  5  6]]]

    Note:
        - Use :func:`tree_eval` to replace this layer with :class:`Identity` during
          evaluation.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 17).reshape(1, 4, 4)
        >>> layer = sk.image.JigSaw2D(2)
        >>> eval_layer = sk.tree_eval(layer)
        >>> print(eval_layer(x))
        [[[ 1  2  3  4]
          [ 5  6  7  8]
          [ 9 10 11 12]
          [13 14 15 16]]]

    Reference:
        - https://imgaug.readthedocs.io/en/latest/source/overview/geometric.html#jigsaw
    """

    tiles: int = sk.field(on_setattr=[IsInstance(int), Range(1)])

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        """Mixes up tiles of an image.

        Args:
            x: channel-first image (CHW)
            key: random key
        """
        return jax.vmap(jigsaw_2d, in_axes=(0, None, None))(x, self.tiles, key)

    @property
    def spatial_ndim(self) -> int:
        return 2


@tree_eval.def_eval(RandomContrast2D)
@tree_eval.def_eval(JigSaw2D)
def random_image_transform(_):
    return Identity()