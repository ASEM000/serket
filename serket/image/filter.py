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
from typing_extensions import Annotated

import serket as sk
from serket.nn.convolution import fft_conv_general_dilated
from serket.nn.initialization import DType
from serket.utils import (
    generate_conv_dim_numbers,
    positive_int_cb,
    resolve_string_padding,
    validate_spatial_ndim,
)


def filter_2d(
    array: Annotated[jax.Array, "CHW"],
    weight: Annotated[jax.Array, "OIHW"],
) -> jax.Array:
    """Filtering wrapping ``jax.lax.conv_general_dilated``.

    Args:
        array: input array. shape is (in_features, *spatial).
        weight: convolutional kernel. shape is (out_features, in_features, *kernel).
    """
    assert array.ndim == 3

    ones = (1,) * (array.ndim - 1)
    x = jax.lax.conv_general_dilated(
        lhs=jnp.expand_dims(array, 0),
        rhs=weight,
        window_strides=ones,
        padding="SAME",
        rhs_dilation=ones,
        dimension_numbers=generate_conv_dim_numbers(array.ndim - 1),
        feature_group_count=array.shape[0],  # in_features
    )
    return jax.lax.stop_gradient_p.bind(jnp.squeeze(x, 0))


def fft_filter_2d(
    array: Annotated[jax.Array, "CHW"],
    weight: Annotated[jax.Array, "OIHW"],
) -> jax.Array:
    """Filtering wrapping ``serket`` ``fft_conv_general_dilated``

    Args:
        array: input array. shape is (in_features, *spatial).
        weight: convolutional kernel. shape is (out_features, in_features, *kernel).
    """
    assert array.ndim == 3

    ones = (1,) * (array.ndim - 1)

    padding = resolve_string_padding(
        in_dim=array.shape[1:],
        padding="SAME",
        kernel_size=weight.shape[2:],
        strides=ones,
    )

    x = fft_conv_general_dilated(
        lhs=jnp.expand_dims(array, 0),
        rhs=weight,
        strides=ones,
        padding=padding,
        dilation=ones,
        groups=array.shape[0],  # in_features
    )
    return jax.lax.stop_gradient_p.bind(jnp.squeeze(x, 0))


class AvgBlur2DBase(sk.TreeClass):
    def __init__(self, kernel_size: int, *, dtype: DType = jnp.float32):
        kernel_size = positive_int_cb(kernel_size)
        kernel = jnp.ones(kernel_size)
        kernel = kernel / jnp.sum(kernel)
        self.kernel = kernel[None, None, None].astype(dtype)

    @property
    def spatial_ndim(self) -> int:
        return 2


class AvgBlur2D(AvgBlur2DBase):
    """Average blur 2D layer.

    .. image:: ../_static/avgblur2d.png

    Args:
        kernel_size: size of the convolving kernel.
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.AvgBlur2D(kernel_size=3)
        >>> print(layer(jnp.ones((1,5,5))))  # doctest: +SKIP
        [[[0.44444448 0.6666667  0.6666667  0.6666667  0.44444448]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.44444448 0.6666667  0.6666667  0.6666667  0.44444448]]]
    """

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.expand_dims(x, 1)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, self.kernel)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, jnp.moveaxis(self.kernel, 2, 3))
        x = x[:, 0]
        return x


class FFTAvgBlur2D(AvgBlur2DBase):
    """Average blur 2D layer using FFT.

    .. image:: ../_static/avgblur2d.png

    Args:
        kernel_size: size of the convolving kernel.
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.FFTAvgBlur2D(kernel_size=3)
        >>> print(layer(jnp.ones((1,5,5))))  # doctest: +SKIP
        [[[0.44444448 0.6666667  0.6666667  0.6666667  0.44444448]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.44444448 0.6666667  0.6666667  0.6666667  0.44444448]]]
    """

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.expand_dims(x, 1)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, self.kernel)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, jnp.moveaxis(self.kernel, 2, 3))
        x = jnp.squeeze(x, 1)
        return x


class GaussianBlur2DBase(sk.TreeClass):
    def __init__(
        self,
        kernel_size: int,
        *,
        sigma: float = 1.0,
        dtype: DType = jnp.float32,
    ):
        self.kernel_size = positive_int_cb(kernel_size)
        self.sigma = sigma
        x = jnp.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size)
        kernel = jnp.exp(-0.5 * jnp.square(x) * jax.lax.rsqrt(self.sigma))
        kernel = kernel / jnp.sum(kernel)
        self.kernel = kernel[None, None, None].astype(dtype)

    @property
    def spatial_ndim(self) -> int:
        return 2


class GaussianBlur2D(GaussianBlur2DBase):
    """Apply Gaussian blur to a channel-first image.

    .. image:: ../_static/gaussianblur2d.png

    Args:
        kernel_size: kernel size
        sigma: sigma. Defaults to 1.
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.GaussianBlur2D(kernel_size=3)
        >>> print(layer(jnp.ones((1,5,5))))  # doctest: +SKIP
        [[[0.5269764 0.7259314 0.7259314 0.7259314 0.5269764]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.5269764 0.7259314 0.7259314 0.7259314 0.5269764]]]
    """

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.expand_dims(x, 1)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, self.kernel)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, jnp.moveaxis(self.kernel, 2, 3))
        x = jnp.squeeze(x, 1)
        return x


class FFTGaussianBlur2D(GaussianBlur2DBase):
    """Apply Gaussian blur to a channel-first image using FFT.

    .. image:: ../_static/gaussianblur2d.png

    Args:
        kernel_size: kernel size
        sigma: sigma. Defaults to 1.
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.FFTGaussianBlur2D(kernel_size=3)
        >>> print(layer(jnp.ones((1,5,5))))  # doctest: +SKIP
        [[[0.5269764 0.7259314 0.7259314 0.7259314 0.5269764]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.5269764 0.7259314 0.7259314 0.7259314 0.5269764]]]
    """

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.expand_dims(x, 1)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, self.kernel)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, jnp.moveaxis(self.kernel, 2, 3))
        x = jnp.squeeze(x, 1)
        return x


class Filter2D(sk.TreeClass):
    """Apply 2D filter for each channel

    .. image:: ../_static/filter2d.png

    Args:
        kernel: kernel array with shape (H, W).
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.Filter2D(kernel=jnp.ones((3,3)))
        >>> print(layer(jnp.ones((1,5,5))))
        [[[4. 6. 6. 6. 4.]
          [6. 9. 9. 9. 6.]
          [6. 9. 9. 9. 6.]
          [6. 9. 9. 9. 6.]
          [4. 6. 6. 6. 4.]]]
    """

    def __init__(
        self,
        kernel: jax.Array,
        *,
        dtype: DType = jnp.float32,
    ):
        if not isinstance(kernel, jax.Array) or kernel.ndim != 2:
            raise ValueError("Expected `kernel` to be a 2D `ndarray` with shape (H, W)")
        self.kernel = kernel[None, None].astype(dtype)

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.expand_dims(x, 1)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, self.kernel)
        x = jnp.squeeze(x, 1)
        return x

    @property
    def spatial_ndim(self) -> int:
        return 2


class FFTFilter2D(sk.TreeClass):
    """Apply 2D filter for each channel using FFT

    .. image:: ../_static/filter2d.png

    Args:
        kernel: kernel array
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.FFTFilter2D(kernel=jnp.ones((3,3)))
        >>> print(layer(jnp.ones((1,5,5))))  # doctest: +SKIP
        [[[4.0000005 6.0000005 6.000001  6.0000005 4.0000005]
          [6.0000005 9.        9.        9.        6.0000005]
          [6.0000005 9.        9.        9.        6.0000005]
          [6.0000005 9.        9.        9.        6.0000005]
          [4.        6.0000005 6.0000005 6.0000005 4.       ]]]
    """

    def __init__(
        self,
        kernel: jax.Array,
        *,
        dtype: DType = jnp.float32,
    ):
        if kernel.ndim != 2:
            raise ValueError("Expected `kernel` to be a 2D `ndarray` with shape (H, W)")

        self.kernel = kernel[None, None].astype(dtype)

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.expand_dims(x, 1)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, self.kernel)
        x = jnp.squeeze(x, 1)
        return x

    @property
    def spatial_ndim(self) -> int:
        return 2
