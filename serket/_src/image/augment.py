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
from math import floor

import jax
import jax.numpy as jnp
import jax.random as jr

import serket as sk
from serket._src.custom_transform import tree_eval
from serket._src.image.color import hsv_to_rgb_3d, rgb_to_hsv_3d
from serket._src.nn.linear import Identity
from serket._src.utils import CHWArray, HWArray, IsInstance, Range, validate_spatial_nd


def pixel_shuffle_3d(array: CHWArray, upscale_factor: tuple[int, int]) -> CHWArray:
    """Rearrange elements in a tensor.

    Args:
        array: input array with shape (channels, height, width)
        upscale_factor: factor to increase spatial resolution by.

    Reference:
        - https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
    """
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
    image: HWArray,
    threshold: float | int,
    max_val: float | int,
) -> HWArray:
    """Inverts all values above a given threshold."""
    _, _ = image.shape
    return jnp.where(image < threshold, image, max_val - image)


def adjust_contrast_2d(image: HWArray, factor: float):
    """Adjusts the contrast of an image by scaling the pixel values by a factor.

    Args:
        array: input array \in [0, 1] with shape (height, width)
        factor: contrast factor to adust the contrast by.
    """
    _, _ = image.shape
    μ = jnp.mean(image, keepdims=True)
    return (factor * (image - μ) + μ).astype(image.dtype).clip(0.0, 1.0)


def random_contrast_2d(
    key: jax.Array,
    array: HWArray,
    range: tuple[float, float],
) -> HWArray:
    """Randomly adjusts the contrast of an image by scaling the pixel values by a factor."""
    _, _ = array.shape
    minval, maxval = range
    factor = jr.uniform(key=key, shape=(), minval=minval, maxval=maxval)
    return adjust_contrast_2d(array, factor)


def adjust_brightness_2d(image: HWArray, factor: float) -> HWArray:
    """Adjusts the brightness of an image by adding a value to the pixel values.

    Args:
        array: input array \in [0, 1] with shape (height, width)
        factor: brightness factor to adust the brightness by.
    """
    _, _ = image.shape
    return jnp.clip((image + factor).astype(image.dtype), 0.0, 1.0)


def random_brightness_2d(
    key: jax.Array,
    image: HWArray,
    range: tuple[float, float],
) -> HWArray:
    """Randomly adjusts the brightness of an image by adding a value to the pixel values."""
    _, _ = image.shape
    minval, maxval = range
    assert 0 <= minval <= maxval <= 1
    factor = jr.uniform(key=key, shape=(), minval=minval, maxval=maxval)
    return adjust_brightness_2d(image, factor)


def pixelate_2d(image: HWArray, scale: int = 16) -> HWArray:
    """Return a pixelated image by downsizing and upsizing"""
    dtype = image.dtype
    h, w = image.shape
    image = image.astype(jnp.float32)
    image = jax.image.resize(image, (h // scale, w // scale), method="linear")
    image = jax.image.resize(image, (h, w), method="linear")
    image = image.astype(dtype)
    return image


@ft.partial(jax.jit, inline=True, static_argnums=2)
def random_jigsaw_2d(key: jax.Array, image: HWArray, tiles: int) -> HWArray:
    """Jigsaw an image by mixing up tiles.

    Args:
        image: The image to jigsaw in shape of (height, width).
        tiles: The number of tiles per side.
        key: The random key to use for shuffling.
    """
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


def posterize_2d(image: HWArray, bits: int) -> HWArray:
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


def adjust_sigmoid_2d(
    image: HWArray,
    cutoff: float = 0.5,
    gain: float = 10,
    inv: bool = False,
) -> HWArray:
    """Adjust sigmoid correction on the input 2D image of range [0, 1]."""
    return jnp.where(
        inv,
        1 - 1 / (1 + jnp.exp(gain * (cutoff - image))),
        1 / (1 + jnp.exp(gain * (cutoff - image))),
    )


def adjust_log_2d(image: HWArray, gain: float = 1, inv: bool = False) -> HWArray:
    """Adjust log correction on the input 2D image of range [0, 1]."""
    return jnp.where(inv, (2**image - 1) * gain, jnp.log2(1 + image) * gain)


def adjust_hue_3d(image: CHWArray, factor: float) -> CHWArray:
    h, s, v = rgb_to_hsv_3d(image)
    divisor = 2 * jnp.pi
    h = jnp.fmod(h + factor, divisor)
    out = jnp.stack([h, s, v], axis=0)
    return hsv_to_rgb_3d(out)


def adust_saturation_3d(image: CHWArray, factor: float) -> CHWArray:
    h, s, v = rgb_to_hsv_3d(image)
    s = jnp.clip(s * factor, 0.0, 1.0)
    out = jnp.stack([h, s, v], axis=0)
    return hsv_to_rgb_3d(out)


def random_hue_3d(
    key: jax.Array, image: CHWArray, range: tuple[float, float]
) -> CHWArray:
    minval, maxval = range
    factor = jr.uniform(key=key, shape=(), minval=minval, maxval=maxval)
    return adjust_hue_3d(image, factor)


def random_saturation_3d(key: jax.Array, image: CHWArray, range: tuple[float, float]):
    minval, maxval = range
    factor = jr.uniform(key=key, shape=(), minval=minval, maxval=maxval)
    return adust_saturation_3d(image, factor)


def fourier_domain_adapt_2d(image: HWArray, styler: HWArray, beta: float):
    # adapted from https://github.com/albumentations-team/albumentations/
    # and original source from https://github.com/YanchaoYang/FDA
    dtype = image.dtype
    height, width = image.shape
    styler = jax.image.resize(styler, (height, width), method="bilinear")
    image_fft = jnp.fft.fft2(image, axes=(0, 1))
    styler_fft = jnp.fft.fft2(styler, axes=(0, 1))
    image_amp, image_phase = jnp.abs(image_fft), jnp.angle(image_fft)
    styler_amp = jnp.abs(styler_fft)
    image_amp = jnp.fft.fftshift(image_amp, axes=(0, 1))
    styler_amp = jnp.fft.fftshift(styler_amp, axes=(0, 1))
    border = int(floor(min((height, width)) * beta))
    center_y = int(floor(height / 2.0))
    center_x = int(floor(width / 2.0))
    y1, y2 = center_y - border, center_y + border + 1
    x1, x2 = center_x - border, center_x + border + 1
    image_amp = image_amp.at[y1:y2, x1:x2].set(styler_amp[y1:y2, x1:x2])
    image_amp = jnp.fft.ifftshift(image_amp, axes=(0, 1))
    image_out = jnp.fft.ifft2(image_amp * jnp.exp(1j * image_phase), axes=(0, 1))
    image_out = jnp.real(image_out)
    return image_out.astype(dtype)


class PixelShuffle2D(sk.TreeClass):
    """Rearrange elements in a tensor.

    .. image:: ../_static/pixelshuffle2d.png

    Args:
        upscale_factor: factor to increase spatial resolution by. accepts an integer.

    Reference:
        - https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upscale_factor: int = 1):
        if not isinstance(upscale_factor, int):
            raise TypeError("`upscale_factor` must be an integer.")

        self.upscale_factor = (upscale_factor, upscale_factor)

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, array: CHWArray) -> CHWArray:
        return pixel_shuffle_3d(array, self.upscale_factor)

    spatial_ndim: int = 2


@sk.autoinit
class AdjustContrast2D(sk.TreeClass):
    """Adjusts the contrast of an 2D input by scaling the pixel values by a factor.

    .. image:: ../_static/adjustcontrast2d.png

    Args:
        factor: contrast factor to adust the contrast by. Defaults to 1.0.

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/image/adjust_contrast
        - https://github.com/deepmind/dm_pix/blob/master/dm_pix/_src/augment.py
    """

    factor: float = 1.0

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: CHWArray) -> CHWArray:
        factor = jax.lax.stop_gradient(self.factor)
        return jax.vmap(adjust_contrast_2d, in_axes=(0, None))(x, factor)

    spatial_ndim: int = 2


class RandomContrast2D(sk.TreeClass):
    """Randomly adjusts the contrast of an 1D input by scaling the pixel values by a factor.

    Args:
        range: contrast range to adust the contrast by. Defaults to (0.5, 1).

    Note:
        - Use :func:`tree_eval` to replace this layer with :class:`Identity` during
          evaluation.

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/image/adjust_contrast
        - https://github.com/deepmind/dm_pix/blob/master/dm_pix/_src/augment.py
    """

    def __init__(self, range: tuple[float, float] = (0.5, 1)):
        if not (isinstance(range, tuple) and len(range) == 2 and range[0] <= range[1]):
            raise ValueError(
                "`range` must be a tuple of two floats, "
                "with the first one smaller than the second one."
            )

        self.range = range

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, array: CHWArray, *, key: jax.Array) -> CHWArray:
        range = jax.lax.stop_gradient(self.range)
        in_axes = (None, 0, None)
        return jax.vmap(random_contrast_2d, in_axes=in_axes)(key, array, range)

    spatial_ndim: int = 2


@sk.autoinit
class AdjustBrightness2D(sk.TreeClass):
    """Adjusts the brightness of an 2D input by adding a value to the pixel values.

    .. image:: ../_static/adjustbrightness2d.png

    Args:
        factor: brightness factor to adust the brightness by. Defaults to 1.0.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> x = jnp.arange(1, 17).reshape(1, 4, 4) / 16.0
        >>> print(sk.image.AdjustBrightness2D(0.5)(x))
        [[[0.5625 0.625  0.6875 0.75  ]
          [0.8125 0.875  0.9375 1.    ]
          [1.     1.     1.     1.    ]
          [1.     1.     1.     1.    ]]]
    """

    factor: float = sk.field(on_setattr=[IsInstance(float), Range(0, 1)])

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, array: CHWArray) -> CHWArray:
        factor = jax.lax.stop_gradient(self.factor)
        return jax.vmap(adjust_brightness_2d, in_axes=(0, None))(array, factor)

    spatial_ndim: int = 2


@sk.autoinit
class RandomBrightness2D(sk.TreeClass):
    """Randomly adjusts the brightness of an 2D input by adding a value to the pixel values.

    Args:
        range: brightness range to adust the brightness by. Defaults to (0.5, 1).

    Note:
        - Use :func:`tree_eval` to replace this layer with :class:`Identity` during
          evaluation.
    """

    range: tuple[float, float] = sk.field(on_setattr=[IsInstance(tuple)])

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, array: CHWArray, *, key: jax.Array) -> CHWArray:
        range = jax.lax.stop_gradient(self.range)
        in_axes = (None, 0, None)
        return jax.vmap(random_brightness_2d, in_axes=in_axes)(key, array, range)

    spatial_ndim: int = 2


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
    def __call__(self, array: CHWArray) -> CHWArray:
        return jax.vmap(pixelate_2d, in_axes=(0, None))(array, self.scale)

    spatial_ndim: int = 2


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
    def __call__(self, array: CHWArray) -> CHWArray:
        threshold, max_val = jax.lax.stop_gradient((self.threshold, self.max_val))
        in_axes = (0, None, None)
        args = (array, threshold, max_val)
        return jax.vmap(solarize_2d, in_axes=in_axes)(*args)

    spatial_ndim: int = 2


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
    def __call__(self, array: CHWArray) -> CHWArray:
        return jax.vmap(posterize_2d, in_axes=(0, None))(array, self.bits)

    spatial_ndim: int = 2


@sk.autoinit
class RandomJigSaw2D(sk.TreeClass):
    """Mixes up tiles of an image.

    .. image:: ../_static/jigsaw2d.png

    Args:
        tiles: number of tiles per side

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> x = jnp.arange(1, 17).reshape(1, 4, 4)
        >>> print(x)
        [[[ 1  2  3  4]
          [ 5  6  7  8]
          [ 9 10 11 12]
          [13 14 15 16]]]
        >>> print(sk.image.RandomJigSaw2D(2)(x, key=jr.PRNGKey(0)))
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
        >>> layer = sk.image.RandomJigSaw2D(2)
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
    def __call__(self, array: CHWArray, *, key: jax.Array) -> CHWArray:
        """Mixes up tiles of an image.

        Args:
            x: channel-first image (CHW)
            key: random key
        """
        in_axes = (None, 0, None)
        args = (key, array, self.tiles)
        return jax.vmap(random_jigsaw_2d, in_axes=in_axes)(*args)

    spatial_ndim: int = 2


class AdjustLog2D(sk.TreeClass):
    """Adjust log correction on the input 2D image of range [0, 1].

    Args:
        image: channel-first image in range [0, 1].
        gain: The gain factor. Default: 1.
        inv:  Whether to invert the log correction. Default: False.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 17).reshape(1, 4, 4) / 16.0
        >>> print(sk.image.AdjustLog2D()(x))  # doctest: +SKIP
        [[[0.08746284 0.16992499 0.24792752 0.32192808]
          [0.3923174  0.45943162 0.52356195 0.5849625 ]
          [0.64385617 0.7004397  0.75488746 0.8073549 ]
          [0.857981   0.9068906  0.9541963  1.        ]]]
    """

    def __init__(self, gain: float = 1, inv: bool = False):
        self.gain = gain
        self.inv = inv

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, array: CHWArray) -> CHWArray:
        in_axes = (0, None, None)
        gain = jax.lax.stop_gradient(self.gain)
        return jax.vmap(adjust_log_2d, in_axes=in_axes)(array, gain, self.inv)

    spatial_ndim: int = 2


class AdjustSigmoid2D(sk.TreeClass):
    """Adjust sigmoid correction on the input 2D image of range [0, 1].


    Args:
        image: channel-first image in range [0, 1].
        cutoff: The cutoff of sigmoid function.
        gain: The multiplier of sigmoid function.
        inv: If is set to True the function will return the inverse sigmoid correction.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 17).reshape(1, 4, 4) / 16.0
        >>> print(AdjustSigmoid2D()(x))  # doctest: +SKIP
        [[[0.01243165 0.02297737 0.04208773 0.07585818]
          [0.13296424 0.22270013 0.34864512 0.5       ]
          [0.6513549  0.7772999  0.86703575 0.9241418 ]
          [0.95791227 0.97702265 0.9875683  0.9933072 ]]]
    """

    def __init__(self, cutoff: float = 0.5, gain: float = 10, inv: bool = False):
        self.cutoff = cutoff
        self.gain = gain
        self.inv = inv

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, array: CHWArray) -> CHWArray:
        in_axes = (0, None, None, None)
        cutoff, gain = jax.lax.stop_gradient((self.cutoff, self.gain))
        args = (array, cutoff, gain, self.inv)
        return jax.vmap(adjust_sigmoid_2d, in_axes=in_axes)(*args)

    spatial_ndim: int = 2


class AdjustHue2D(sk.TreeClass):
    """Adjust hue of an RGB image.

    .. image:: ../_static/adjusthue2d.png

    Args:
        image: channel-first RGB image in range [0, 1].
        factor: The hue factor.
    """

    def __init__(self, factor: float):
        self.factor = factor

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, array: CHWArray) -> CHWArray:
        factor = jax.lax.stop_gradient(self.factor)
        return adjust_hue_3d(array, factor)

    spatial_ndim: int = 2


class RandomHue2D(sk.TreeClass):
    """Randomly adjust hue of an RGB image.

    Args:
        image: channel-first RGB image in range [0, 1].
        range: The hue range.

    Note:
        - Use :func:`tree_eval` to replace this layer with :class:`Identity` during
          evaluation.
    """

    def __init__(self, range: tuple[float, float]):
        self.range = range

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, array: CHWArray, *, key: jax.Array) -> CHWArray:
        range = jax.lax.stop_gradient(self.range)
        return random_hue_3d(key, array, range)

    spatial_ndim: int = 2


class AdjustSaturation2D(sk.TreeClass):
    """Adjust saturation of an RGB image.

    .. image:: ../_static/adjustsaturation2d.png

    Args:
        image: channel-first RGB image in range [0, 1].
        factor: The saturation factor.
    """

    def __init__(self, factor: float):
        self.factor = factor

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, array: CHWArray) -> CHWArray:
        factor = jax.lax.stop_gradient(self.factor)
        return adust_saturation_3d(array, factor)

    spatial_ndim: int = 2


class RandomSaturation2D(sk.TreeClass):
    """Randomly adjust saturation of an RGB image.

    Args:
        image: channel-first RGB image in range [0, 1].
        range: The saturation range.

    Note:
        - Use :func:`tree_eval` to replace this layer with :class:`Identity` during
          evaluation.
    """

    def __init__(self, range: tuple[float, float]):
        self.range = range

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: CHWArray, *, key: jax.Array) -> CHWArray:
        range = jax.lax.stop_gradient(self.range)
        return random_saturation_3d(key, x, range)

    spatial_ndim: int = 2


class FourierDomainAdapt2D(sk.TreeClass):
    """Domain adaptation via style transfer

    .. image:: ../_static/fourierdomainadapt2d.png

    Steps:
        1. Apply FFT to source and target images.
        2. Replace the low frequency part of the source amplitude with that from the target.
        3. Apply inverse FFT to the modified source spectrum.

    Args:
        beta: controls the size of the low frequency part to be replaced. accepts
            float.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> source = jnp.ones((1, 5, 5))
        >>> target = jnp.ones((1, 10, 10))
        >>> layer = sk.image.FourierDomainAdapt2D(0.5)
        >>> layer(source, target).shape
        (1, 5, 5)

    Reference:
        - https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf
        - https://github.com/albumentations-team/albumentations/
    """

    def __init__(self, beta: float):
        self.beta = beta

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim", argnum=0)
    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim", argnum=1)
    def __call__(self, image: CHWArray, target: CHWArray) -> CHWArray:
        """Fourier Domain Adaptation

        Args:
            image: channel-first source image.
            target: channel-first target image to adapt to.
        """
        beta = jax.lax.stop_gradient(self.beta)
        in_axes = (0, 0, None)
        args = (image, target, beta)
        return jax.vmap(fourier_domain_adapt_2d, in_axes=in_axes)(*args)

    spatial_ndim: int = 2


@tree_eval.def_eval(RandomBrightness2D)
@tree_eval.def_eval(RandomHue2D)
@tree_eval.def_eval(RandomSaturation2D)
@tree_eval.def_eval(RandomContrast2D)
@tree_eval.def_eval(RandomJigSaw2D)
def _(_) -> Identity:
    return Identity()
