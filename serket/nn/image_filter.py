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
from jax import lax

import serket as sk
from serket.nn.convolution import DepthwiseConv2D
from serket.nn.fft_convolution import DepthwiseFFTConv2D
from serket.nn.utils import positive_int_cb, validate_axis_shape, validate_spatial_ndim


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
    """

    def __init__(self, in_features: int, kernel_size: int | tuple[int, int]):
        weight = jnp.ones(kernel_size)
        weight = weight / jnp.sum(weight)
        weight = weight[:, None]
        weight = jnp.repeat(weight[None, None], in_features, axis=0)

        self.conv1 = DepthwiseConv2D(
            in_features=in_features,
            kernel_size=(kernel_size, 1),
            padding="same",
            weight_init_func=lambda *_: weight,
            bias_init_func=None,
        )

        self.conv2 = DepthwiseConv2D(
            in_features=in_features,
            kernel_size=(1, kernel_size),
            padding="same",
            weight_init_func=lambda *_: jnp.moveaxis(weight, 2, 3),
            bias_init_func=None,
        )

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
    """

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
            weight_init_func=lambda *_: weight,
            bias_init_func=None,
        )

        self.conv2 = DepthwiseConv2D(
            in_features=in_features,
            kernel_size=(1, kernel_size),
            padding="same",
            weight_init_func=lambda *_: jnp.moveaxis(weight, 2, 3),
            bias_init_func=None,
        )

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
            weight_init_func=lambda *_: weight,
            bias_init_func=None,
        )

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
            weight_init_func=lambda *_: weight,
            bias_init_func=None,
        )

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="conv.in_features", axis=0)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return lax.stop_gradient(self.conv(x))

    @property
    def spatial_ndim(self) -> int:
        return 2
