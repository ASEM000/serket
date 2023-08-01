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


@tree_eval.def_eval(RandomContrast2D)
def tree_eval_random_contrast2d(_: RandomContrast2D):
    return Identity()
