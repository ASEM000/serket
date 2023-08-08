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

# import kernex as kex
import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax.scipy.ndimage import map_coordinates

import serket as sk
from serket.nn.convolution import DepthwiseConv2D, DepthwiseFFTConv2D
from serket.nn.custom_transform import tree_eval
from serket.nn.linear import Identity
from serket.nn.utils import (
    maybe_lazy_call,
    maybe_lazy_init,
    positive_int_cb,
    validate_axis_shape,
    validate_spatial_ndim,
)


def is_lazy_call(instance, *_, **__) -> bool:
    return getattr(instance, "in_features", False) is None


def is_lazy_init(_, in_features, *__, **___) -> bool:
    return in_features is None


def infer_in_features(_, x, *__, **___) -> int:
    return x.shape[0]


image_updates = dict(in_features=infer_in_features)


class AvgBlur2D(sk.TreeClass):
    """Average blur 2D layer

    Args:
        in_features: number of input channels.
        kernel_size: size of the convolving kernel.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.nn.AvgBlur2D(in_features=1, kernel_size=3)
        >>> print(layer(jnp.ones((1,5,5))))  # doctest: +SKIP
        [[[0.44444448 0.6666667  0.6666667  0.6666667  0.44444448]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.44444448 0.6666667  0.6666667  0.6666667  0.44444448]]]

    Note:
        :class:`.AvgBlur2D` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class Blur(sk.TreeClass):
        ...    l1: sk.nn.AvgBlur2D = sk.nn.AvgBlur2D(None, 3)
        ...    l2: sk.nn.AvgBlur2D = sk.nn.AvgBlur2D(None, 3)
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_blur = Blur()
        >>> # materialize the layer
        >>> _, materialized_blur = lazy_blur.at["__call__"](jnp.ones((5, 2, 2)))
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(self, in_features: int | None, kernel_size: int | tuple[int, int]):
        weight = jnp.ones(kernel_size)
        weight = weight / jnp.sum(weight)
        weight = weight[:, None]
        weight = jnp.repeat(weight[None, None], in_features, axis=0)

        self.conv1 = DepthwiseConv2D(
            in_features=in_features,
            kernel_size=(kernel_size, 1),
            padding="same",
            weight_init=lambda *_: weight,
            bias_init=None,
        )

        self.conv2 = DepthwiseConv2D(
            in_features=in_features,
            kernel_size=(1, kernel_size),
            padding="same",
            weight_init=lambda *_: jnp.moveaxis(weight, 2, 3),
            bias_init=None,
        )

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=image_updates)
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="conv1.in_features", axis=0)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return lax.stop_gradient(self.conv2(self.conv1(x)))

    @property
    def spatial_ndim(self) -> int:
        return 2


class GaussianBlur2D(sk.TreeClass):
    """Apply Gaussian blur to a channel-first image.

    Args:
        in_features: number of input features
        kernel_size: kernel size
        sigma: sigma. Defaults to 1.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.nn.GaussianBlur2D(in_features=1, kernel_size=3)
        >>> print(layer(jnp.ones((1,5,5))))  # doctest: +SKIP
        [[[0.5269764 0.7259314 0.7259314 0.7259314 0.5269764]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.5269764 0.7259314 0.7259314 0.7259314 0.5269764]]]

    Note:
        :class:`.GaussianBlur2D` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class Blur(sk.TreeClass):
        ...    l1: sk.nn.GaussianBlur2D = sk.nn.GaussianBlur2D(None, 3)
        ...    l2: sk.nn.GaussianBlur2D = sk.nn.GaussianBlur2D(None, 3)
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_blur = Blur()
        >>> # materialize the layer
        >>> _, materialized_blur = lazy_blur.at["__call__"](jnp.ones((5, 2, 2)))
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(self, in_features: int, kernel_size: int, *, sigma: float = 1.0):
        kernel_size = positive_int_cb(kernel_size)
        self.sigma = sigma

        x = jnp.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size)
        weight = jnp.exp(-0.5 * jnp.square(x) * jax.lax.rsqrt(self.sigma))

        weight = weight / jnp.sum(weight)
        weight = weight[:, None]

        weight = jnp.repeat(weight[None, None], in_features, axis=0)
        self.conv1 = DepthwiseFFTConv2D(
            in_features=in_features,
            kernel_size=(kernel_size, 1),
            padding="same",
            weight_init=lambda *_: weight,
            bias_init=None,
        )

        self.conv2 = DepthwiseFFTConv2D(
            in_features=in_features,
            kernel_size=(1, kernel_size),
            padding="same",
            weight_init=lambda *_: jnp.moveaxis(weight, 2, 3),
            bias_init=None,
        )

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=image_updates)
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="conv1.in_features", axis=0)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return lax.stop_gradient(self.conv1(self.conv2(x)))

    @property
    def spatial_ndim(self) -> int:
        return 2


class Filter2D(sk.TreeClass):
    """Apply 2D filter for each channel

    Args:
        in_features: number of input channels.
        kernel: kernel array.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.nn.Filter2D(in_features=1, kernel=jnp.ones((3,3)))
        >>> print(layer(jnp.ones((1,5,5))))
        [[[4. 6. 6. 6. 4.]
          [6. 9. 9. 9. 6.]
          [6. 9. 9. 9. 6.]
          [6. 9. 9. 9. 6.]
          [4. 6. 6. 6. 4.]]]
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(self, in_features: int, kernel: jax.Array):
        if not isinstance(kernel, jax.Array) or kernel.ndim != 2:
            raise ValueError("Expected `kernel` to be a 2D `ndarray` with shape (H, W)")

        in_features = positive_int_cb(in_features)
        weight = jnp.stack([kernel] * in_features, axis=0)
        weight = weight[:, None]

        self.conv = DepthwiseConv2D(
            in_features=in_features,
            kernel_size=kernel.shape,
            padding="same",
            weight_init=lambda *_: weight,
            bias_init=None,
        )

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=image_updates)
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="conv.in_features", axis=0)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return lax.stop_gradient(self.conv(x))

    @property
    def spatial_ndim(self) -> int:
        return 2


class FFTFilter2D(sk.TreeClass):
    """Apply 2D filter for each channel using FFT

    Args:
        in_features: number of input channels
        kernel: kernel array

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.nn.FFTFilter2D(in_features=1, kernel=jnp.ones((3,3)))
        >>> print(layer(jnp.ones((1,5,5))))  # doctest: +SKIP
        [[[4.0000005 6.0000005 6.000001  6.0000005 4.0000005]
          [6.0000005 9.        9.        9.        6.0000005]
          [6.0000005 9.        9.        9.        6.0000005]
          [6.0000005 9.        9.        9.        6.0000005]
          [4.        6.0000005 6.0000005 6.0000005 4.       ]]]
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(self, in_features: int, kernel: jax.Array):
        if not isinstance(kernel, jax.Array) or kernel.ndim != 2:
            raise ValueError("Expected `kernel` to be a 2D `ndarray` with shape (H, W)")

        in_features = positive_int_cb(in_features)
        weight = jnp.stack([kernel] * in_features, axis=0)
        weight = weight[:, None]

        self.conv = DepthwiseFFTConv2D(
            in_features=in_features,
            kernel_size=kernel.shape,
            padding="same",
            weight_init=lambda *_: weight,
            bias_init=None,
        )

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=image_updates)
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="conv.in_features", axis=0)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return lax.stop_gradient(self.conv(x))

    @property
    def spatial_ndim(self) -> int:
        return 2


class HistogramEqualization2D(sk.TreeClass):
    """Apply histogram equalization to 2D spatial array channel wise

    Args:
        bins: number of bins. Defaults to 256.

    Reference:
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


def adjust_contrast_nd(x: jax.Array, contrast_factor: float):
    """Adjusts the contrast of an image by scaling the pixel values by a factor."""
    μ = jnp.mean(x, axis=tuple(range(1, x.ndim)), keepdims=True)
    return (contrast_factor * (x - μ) + μ).astype(x.dtype)


def random_contrast_nd(
    x: jax.Array,
    contrast_range: tuple[float, float],
    key: jr.KeyArray = jr.PRNGKey(0),
) -> jax.Array:
    """Randomly adjusts the contrast of an image by scaling the pixel values by a factor."""
    minval, maxval = contrast_range
    contrast_factor = jr.uniform(key=key, shape=(), minval=minval, maxval=maxval)
    return adjust_contrast_nd(x, contrast_factor)


@sk.autoinit
class AdjustContrast2D(sk.TreeClass):
    """Adjusts the contrast of an 2D input by scaling the pixel values by a factor.

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/image/adjust_contrast
        - https://github.com/deepmind/dm_pix/blob/master/dm_pix/_src/augment.py
    """

    contrast_factor: float = 1.0

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return lax.stop_gradient(adjust_contrast_nd(x, self.contrast_factor))

    @property
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        return 2


class RandomContrast2D(sk.TreeClass):
    """Randomly adjusts the contrast of an 1D input by scaling the pixel values by a factor.

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

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, *, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        return random_contrast_nd(x, lax.stop_gradient(self.contrast_range), key=key)

    @property
    def spatial_ndim(self) -> int:
        return 2


def affine(image, matrix):
    _, h, w = image.shape
    center = jnp.array((h // 2, w // 2))
    coords = jnp.indices((h, w)).reshape(2, -1) - center.reshape(2, 1)
    coords = matrix @ coords + center.reshape(2, 1)

    def affine_channel(array):
        return map_coordinates(array, coords, order=1).reshape((h, w))

    return jax.vmap(affine_channel)(image)


def horizontal_shear(image: jax.Array, angle: float) -> jax.Array:
    """shear rows by an angle in degrees"""
    shear = jnp.tan(jnp.deg2rad(angle))
    matrix = jnp.array([[1, shear], [0, 1]])
    return affine(image, matrix)


def random_horizontal_shear(
    image: jax.Array,
    angle_range: tuple[float, float],
    key: jax.random.KeyArray,
) -> jax.Array:
    """shear rows by an angle in degrees"""
    minval, maxval = angle_range
    angle = jr.uniform(key=key, shape=(), minval=minval, maxval=maxval)
    return horizontal_shear(image, angle)


def vertical_shear(image: jax.Array, angle: float) -> jax.Array:
    """shear cols by an angle in degrees"""
    shear = jnp.tan(jnp.deg2rad(angle))
    matrix = jnp.array([[1, 0], [shear, 1]])
    return affine(image, matrix)


def random_vertical_shear(
    image: jax.Array,
    angle_range: tuple[float, float],
    key: jax.random.KeyArray,
) -> jax.Array:
    """shear cols by an angle in degrees"""
    minval, maxval = angle_range
    angle = jr.uniform(key=key, shape=(), minval=minval, maxval=maxval)
    return vertical_shear(image, angle)


def rotate(image: jax.Array, angle: float) -> jax.Array:
    """Rotate a channel-first image by an angle in degrees in CCW direction."""
    θ = jnp.deg2rad(angle)
    matrix = jnp.array([[jnp.cos(θ), -jnp.sin(θ)], [jnp.sin(θ), jnp.cos(θ)]])
    return affine(image, matrix)


def random_rotate(
    image: jax.Array,
    angle_range: tuple[float, float],
    key: jax.random.KeyArray,
) -> jax.Array:
    minval, maxval = angle_range
    angle = jr.uniform(key=key, shape=(), minval=minval, maxval=maxval)
    return rotate(image, angle)


class Rotate2D(sk.TreeClass):
    """Rotate a 2D image by an angle in dgrees in CCW direction

    Args:
        angle: angle to rotate in degrees counter-clockwise direction.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.nn.Rotate2D(30)(x))
        [[[ 2  5  3  2  1]
          [ 9 10  8  7  5]
          [16 15 13 11 10]
          [21 19 18 16 11]
          [ 6 18 23 21  5]]]
    """

    def __init__(self, angle: float):
        self.angle = angle

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return rotate(x, jax.lax.stop_gradient(self.angle))

    @property
    def spatial_ndim(self) -> int:
        return 2


class RandomRotate2D(sk.TreeClass):
    """Rotate a 2D image by an angle in dgrees in CCW direction

    Args:
        angle_range: a tuple of min angle and max angle to randdomly choose from.

    Example:
        >>> import serket as sk
        >>> import jax
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.nn.RandomRotate2D((10, 30))(x, key=jax.random.PRNGKey(0)))
        [[[ 2  4  3  3  2]
          [ 7  9  8  7  7]
          [14 14 13 12 12]
          [19 19 18 17 13]
          [10 18 23 22 10]]]
    """

    def __init__(self, angle_range: tuple[float, float]):
        if not (
            isinstance(angle_range, tuple)
            and len(angle_range) == 2
            and isinstance(angle_range[0], (int, float))
            and isinstance(angle_range[1], (int, float))
        ):
            raise ValueError(f"`{angle_range=}` must be a tuple of 2 floats/ints ")

        self.angle_range = angle_range

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(
        self,
        x: jax.Array,
        key: jax.random.KeyArray = jax.random.PRNGKey(0),
    ) -> jax.Array:
        return random_rotate(x, jax.lax.stop_gradient(self.angle_range), key)

    @property
    def spatial_ndim(self) -> int:
        return 2


class HorizontalShear2D(sk.TreeClass):
    """Shear an image horizontally

    Args:
        angle: angle to rotate in degrees counter-clockwise direction.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.nn.HorizontalShear2D(60)(x))
        [[[ 0  0  3 13 22]
          [ 0  1  8 18 13]
          [ 0  3 13 23  0]
          [ 1  8 18  6  0]
          [ 4 13 23  0  0]]]
    """

    def __init__(self, angle: float):
        self.angle = angle

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return horizontal_shear(x, jax.lax.stop_gradient(self.angle))

    @property
    def spatial_ndim(self) -> int:
        return 2


class RandomHorizontalShear2D(sk.TreeClass):
    """Shear an image horizontally with random angle choice.

    Args:
        angle_range: a tuple of min angle and max angle to randdomly choose from.

    Example:
        >>> import serket as sk
        >>> import jax
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.nn.RandomHorizontalShear2D((10, 60))(x, key=jax.random.PRNGKey(0)))
        [[[ 0  1  3  7 11]
          [ 1  4  8 12 16]
          [ 5  9 13 17 21]
          [10 14 18 22 20]
          [15 19 23 10  0]]]
    """

    def __init__(self, angle_range: tuple[float, float]):
        if not (
            isinstance(angle_range, tuple)
            and len(angle_range) == 2
            and isinstance(angle_range[0], (int, float))
            and isinstance(angle_range[1], (int, float))
        ):
            raise ValueError(f"`{angle_range=}` must be a tuple of 2 floats")
        self.angle_range = angle_range

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(
        self,
        x: jax.Array,
        key: jax.random.KeyArray = jax.random.PRNGKey(0),
    ) -> jax.Array:
        return random_horizontal_shear(x, jax.lax.stop_gradient(self.angle_range), key)

    @property
    def spatial_ndim(self) -> int:
        return 2


class VerticalShear2D(sk.TreeClass):
    """Shear an image vertically

    Args:
        angle: angle to rotate in degrees counter-clockwise direction.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.nn.VerticalShear2D(60)(x))
        [[[ 0  0  0  1  2]
          [ 0  2  6  7  8]
          [11 12 13 14 15]
          [18 19 20  5  0]
          [24 13  0  0  0]]]
    """

    def __init__(self, angle: float):
        self.angle = angle

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return vertical_shear(x, jax.lax.stop_gradient(self.angle))

    @property
    def spatial_ndim(self) -> int:
        return 2


class RandomVerticalShear2D(sk.TreeClass):
    """Shear an image vertically with random angle choice.

    Args:
        angle_range: a tuple of min angle and max angle to randdomly choose from.

    Example:
        >>> import serket as sk
        >>> import jax
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.nn.RandomVerticalShear2D((10, 60))(x, key=jax.random.PRNGKey(0)))
        [[[ 0  1  2  3  4]
          [ 2  6  7  8  9]
          [11 12 13 14 15]
          [17 18 19 20  8]
          [22 23 24 20  0]]]
    """

    def __init__(self, angle_range: tuple[float, float]):
        if not (
            isinstance(angle_range, tuple)
            and len(angle_range) == 2
            and isinstance(angle_range[0], (int, float))
            and isinstance(angle_range[1], (int, float))
        ):
            raise ValueError(f"`{angle_range=}` must be a tuple of 2 floats")
        self.angle_range = angle_range

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(
        self,
        x: jax.Array,
        key: jax.random.KeyArray = jax.random.PRNGKey(0),
    ) -> jax.Array:
        return random_vertical_shear(x, jax.lax.stop_gradient(self.angle_range), key)

    @property
    def spatial_ndim(self) -> int:
        return 2


def pixelate(image: jax.Array, scale: int = 16) -> jax.Array:
    """Return a pixelated image by downsizing and upsizing"""
    dtype = image.dtype
    c, h, w = image.shape
    image = image.astype(jnp.float32)
    image = jax.image.resize(image, (c, h // scale, w // scale), method="linear")
    image = jax.image.resize(image, (c, h, w), method="linear")
    image = image.astype(dtype)
    return image


class Pixelate2D(sk.TreeClass):
    """Pixelate an image by upsizing and downsizing an image

    Args:
        scale: the scale to which the image will be downsized before being upsized
            to the original shape. for example, ``scale=2`` means the image will
            be downsized to half of its size before being updsized to the original
            size. Higher scale will lead to higher degree of pixelation.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.nn.Pixelate2D(2)(x))  # doctest: +SKIP
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

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return pixelate(x, jax.lax.stop_gradient(self.scale))

    @property
    def spatial_ndim(self) -> int:
        return 2


def perspective_transform(image: jax.Array, coeffs: jax.Array) -> jax.Array:
    """Apply a perspective transform to an image."""

    _, rows, cols = image.shape
    y, x = jnp.meshgrid(jnp.arange(rows), jnp.arange(cols), indexing="ij")
    a, b, c, d, e, f, g, h = coeffs
    w = g * x + h * y + 1.0
    x_prime = (a * x + b * y + c) / w
    y_prime = (d * x + e * y + f) / w
    coords = [y_prime.ravel(), x_prime.ravel()]

    def transform_channel(image):
        return map_coordinates(image, coords, order=1).reshape(rows, cols)

    return jax.vmap(transform_channel)(image)


def random_perspective(
    image: jax.Array,
    key: jax.random.KeyArray,
    scale: float = 1.0,
) -> jax.Array:
    """Applies a random perspective transform to a channel-first image"""
    _, __, ___ = image.shape
    a = e = 1.0
    b = d = 0.0
    c = f = 0.0  # no translation
    g, h = jr.uniform(key, shape=(2,), minval=-1e-4, maxval=1e-4) * scale
    coeffs = jnp.array([a, b, c, d, e, f, g, h])
    return perspective_transform(image, coeffs)


class RandomPerspective2D(sk.TreeClass):
    """Applies a random perspective transform to a channel-first image.

    Args:
        scale: the scale of the random perspective transform. Higher scale will
            lead to higher degree of perspective transform. default to 1.0. 0.0
            means no perspective transform.

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
        >>> layer = sk.nn.RandomPerspective2D(100)
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

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        return random_perspective(x, key, jax.lax.stop_gradient(self.scale))

    @property
    def spatial_ndim(self) -> int:
        return 2


@tree_eval.def_eval(RandomContrast2D)
@tree_eval.def_eval(RandomRotate2D)
@tree_eval.def_eval(RandomHorizontalShear2D)
@tree_eval.def_eval(RandomVerticalShear2D)
@tree_eval.def_eval(RandomPerspective2D)
def random_image_transform(_):
    return Identity()
