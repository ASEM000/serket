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
from typing_extensions import Annotated

import serket as sk
from serket._src.nn.convolution import fft_conv_general_dilated
from serket._src.nn.initialization import DType
from serket._src.utils import (
    canonicalize,
    generate_conv_dim_numbers,
    resolve_string_padding,
    validate_spatial_nd,
)


def filter_2d(
    array: Annotated[jax.Array, "HW"],
    weight: Annotated[jax.Array, "HW"],
) -> Annotated[jax.Array, "HW"]:
    """Filtering wrapping ``jax.lax.conv_general_dilated``.

    Args:
        array: 2D input array. shape is (row, col).
        weight: convolutional kernel. shape is (row, col).

    Note:
        - To filter 3D array, channel-wise use ``jax.vmap(filter_2d, in_axes=(0, None))``.
    """
    assert array.ndim == 2
    assert weight.ndim == 2

    array = jnp.expand_dims(array, 0)
    weight = jnp.expand_dims(weight, (0, 1))

    x = jax.lax.conv_general_dilated(
        lhs=jnp.expand_dims(array, 0),
        rhs=weight,
        window_strides=(1, 1),
        padding="SAME",
        rhs_dilation=(1, 1),
        dimension_numbers=generate_conv_dim_numbers(2),
        feature_group_count=array.shape[0],  # in_features
    )
    return jax.lax.stop_gradient_p.bind(jnp.squeeze(x, (0, 1)))


def fft_filter_2d(
    array: Annotated[jax.Array, "HW"],
    weight: Annotated[jax.Array, "HW"],
) -> Annotated[jax.Array, "HW"]:
    """Filtering wrapping ``serket`` ``fft_conv_general_dilated``

    Args:
        array: 2D input array. shape is (row, col).
        weight: convolutional kernel. shape is (row, col).

    Note:
        - To filter 3D array, channel-wise use ``jax.vmap(filter_2d, in_axes=(0, None))``.
    """
    assert array.ndim == 2
    assert weight.ndim == 2

    array = jnp.expand_dims(array, 0)
    weight = jnp.expand_dims(weight, (0, 1))

    padding = resolve_string_padding(
        in_dim=array.shape[1:],
        padding="SAME",
        kernel_size=weight.shape[2:],
        strides=(1, 1),
    )

    x = fft_conv_general_dilated(
        lhs=jnp.expand_dims(array, 0),
        rhs=weight,
        strides=(1, 1),
        padding=padding,
        dilation=(1, 1),
        groups=array.shape[0],  # in_features
    )
    return jax.lax.stop_gradient_p.bind(jnp.squeeze(x, (0, 1)))


def calculate_average_kernel(
    kernel_size: int,
    dtype: DType,
) -> Annotated[jax.Array, "HW"]:
    kernel = jnp.ones((kernel_size))
    kernel = kernel / jnp.sum(kernel)
    kernel = kernel.astype(dtype)
    kernel = jnp.expand_dims(kernel, 0)
    return kernel


class AvgBlur2DBase(sk.TreeClass):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        *,
        dtype: DType = jnp.float32,
    ):
        self.kernel_size = canonicalize(kernel_size, ndim=2, name="kernel_size")
        ky, kx = self.kernel_size
        self.kernel_x = calculate_average_kernel(kx, dtype)
        self.kernel_y = calculate_average_kernel(ky, dtype)

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
        >>> print(layer(jnp.ones((1, 5, 5))))  # doctest: +SKIP
        [[[0.44444448 0.6666667  0.6666667  0.6666667  0.44444448]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.44444448 0.6666667  0.6666667  0.6666667  0.44444448]]]
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        kernel_x, kernel_y = jax.lax.stop_gradient((self.kernel_x, self.kernel_y))
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, kernel_x)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, kernel_y.T)
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
        >>> print(layer(jnp.ones((1, 5, 5))))  # doctest: +SKIP
        [[[0.44444448 0.6666667  0.6666667  0.6666667  0.44444448]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.44444448 0.6666667  0.6666667  0.6666667  0.44444448]]]
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        kernel_x, kernel_y = jax.lax.stop_gradient((self.kernel_x, self.kernel_y))
        x = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel_x)
        x = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel_y.T)
        return x


def calculate_gaussian_kernel(
    kernel_size: int,
    sigma: float,
    dtype: DType,
) -> Annotated[jax.Array, "HW"]:
    x = jnp.arange(kernel_size) - kernel_size // 2
    x = x + 0.5 if kernel_size % 2 == 0 else x
    kernel = jnp.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / jnp.sum(kernel)
    kernel = kernel.astype(dtype)
    kernel = jnp.expand_dims(kernel, (0))
    return kernel


class GaussianBlur2DBase(sk.TreeClass):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        *,
        sigma: float | tuple[float, float] = 1.0,
        dtype: DType = jnp.float32,
    ):
        self.kernel_size = canonicalize(kernel_size, ndim=2, name="kernel_size")
        self.sigma = canonicalize(sigma, ndim=2, name="sigma")
        ky, kx = self.kernel_size
        sigma_y, sigma_x = self.sigma
        self.kernel_x = calculate_gaussian_kernel(kx, sigma_x, dtype)
        self.kernel_y = calculate_gaussian_kernel(ky, sigma_y, dtype)

    @property
    def spatial_ndim(self) -> int:
        return 2


class GaussianBlur2D(GaussianBlur2DBase):
    """Apply Gaussian blur to a channel-first image.

    .. image:: ../_static/gaussianblur2d.png

    Args:
        kernel_size: kernel size. accepts int or tuple of two ints.
        sigma: sigma. Defaults to 1. accepts float or tuple of two floats.
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.GaussianBlur2D(kernel_size=3)
        >>> print(layer(jnp.ones((1, 5, 5))))  # doctest: +SKIP
        [[[0.5269764 0.7259314 0.7259314 0.7259314 0.5269764]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.5269764 0.7259314 0.7259314 0.7259314 0.5269764]]]
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        kernel_x, kernel_y = jax.lax.stop_gradient((self.kernel_x, self.kernel_y))
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, kernel_x)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, kernel_y.T)
        return x


class FFTGaussianBlur2D(GaussianBlur2DBase):
    """Apply Gaussian blur to a channel-first image using FFT.

    .. image:: ../_static/gaussianblur2d.png

    Args:
        kernel_size: kernel size. accepts int or tuple of two ints.
        sigma: sigma. Defaults to 1. accepts float or tuple of two floats.
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.FFTGaussianBlur2D(kernel_size=3)
        >>> print(layer(jnp.ones((1, 5, 5))))  # doctest: +SKIP
        [[[0.5269764 0.7259314 0.7259314 0.7259314 0.5269764]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.5269764 0.7259314 0.7259314 0.7259314 0.5269764]]]
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        kernel_x, kernel_y = jax.lax.stop_gradient((self.kernel_x, self.kernel_y))
        x = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel_x)
        x = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel_y.T)
        return x


class UnsharpMask2D(GaussianBlur2DBase):
    """Apply unsharp mask to a channel-first image.

    .. image:: ../_static/unsharpmask2d.png

    Args:
        kernel_size: kernel size. accepts int or tuple of two ints.
        sigma: sigma. Defaults to 1. accepts float or tuple of two floats.
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.UnsharpMask2D(kernel_size=3)
        >>> print(layer(jnp.ones((1, 5, 5))))  # doctest: +SKIP
        [[[1.4730237 1.2740686 1.2740686 1.2740686 1.4730237]
          [1.2740686 1.        1.        1.        1.2740686]
          [1.2740686 1.        1.        1.        1.2740686]
          [1.2740686 1.        1.        1.        1.2740686]
          [1.4730237 1.2740686 1.2740686 1.2740686 1.4730237]]]
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        kernel_x, kernel_y = jax.lax.stop_gradient((self.kernel_x, self.kernel_y))
        blur = jax.vmap(filter_2d, in_axes=(0, None))(x, kernel_x)
        blur = jax.vmap(filter_2d, in_axes=(0, None))(blur, kernel_y.T)
        return x + (x - blur)


class FFTUnsharpMask2D(GaussianBlur2DBase):
    """Apply unsharp mask to a channel-first image using FFT.

    .. image:: ../_static/unsharpmask2d.png

    Args:
        kernel_size: kernel size. accepts int or tuple of two ints.
        sigma: sigma. Defaults to 1. accepts float or tuple of two floats.
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.FFTUnsharpMask2D(kernel_size=3)
        >>> print(layer(jnp.ones((1, 5, 5))))  # doctest: +SKIP
        [[[1.4730237 1.2740686 1.2740686 1.2740686 1.4730237]
          [1.2740686 1.        1.        1.        1.2740686]
          [1.2740686 1.        1.        1.        1.2740686]
          [1.2740686 1.        1.        1.        1.2740686]
          [1.4730237 1.2740686 1.2740686 1.2740686 1.4730237]]]
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        kernel_x, kernel_y = jax.lax.stop_gradient((self.kernel_x, self.kernel_y))
        blur = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel_x)
        blur = jax.vmap(fft_filter_2d, in_axes=(0, None))(blur, kernel_y.T)
        return x + (x - blur)


def calculate_box_kernel(kernel_size: int, dtype: DType) -> Annotated[jax.Array, "HW"]:
    kernel = jnp.ones((kernel_size))
    kernel = kernel.astype(dtype)
    kernel = jnp.expand_dims(kernel, 0)
    return kernel / kernel_size


class BoxBlur2DBase(sk.TreeClass):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        *,
        dtype: DType = jnp.float32,
    ):
        self.kernel_size = canonicalize(kernel_size, ndim=2, name="kernel_size")
        ky, kx = self.kernel_size
        self.kernel_x = calculate_box_kernel(kx, dtype)
        self.kernel_y = calculate_box_kernel(ky, dtype)

    @property
    def spatial_ndim(self) -> int:
        return 2


class BoxBlur2D(BoxBlur2DBase):
    """Box blur 2D layer.

    .. image:: ../_static/boxblur2d.png

    Args:
        kernel_size: size of the convolving kernel. Accepts int or tuple of two ints.
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.BoxBlur2D((3, 5))
        >>> print(layer(jnp.ones((1, 5, 5))))  # doctest: +SKIP
        [[[0.40000004 0.53333336 0.6666667  0.53333336 0.40000004]
          [0.6        0.8        1.         0.8        0.6       ]
          [0.6        0.8        1.         0.8        0.6       ]
          [0.6        0.8        1.         0.8        0.6       ]
          [0.40000004 0.53333336 0.6666667  0.53333336 0.40000004]]]
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        kernel_x, kernel_y = jax.lax.stop_gradient((self.kernel_x, self.kernel_y))
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, kernel_x)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, kernel_y.T)
        return x


class FFTBoxBlur2D(BoxBlur2DBase):
    """Box blur 2D layer using FFT.

    .. image:: ../_static/boxblur2d.png

    Args:
        kernel_size: size of the convolving kernel. Accepts int or tuple of two ints.
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.BoxBlur2D((3, 5))
        >>> print(layer(jnp.ones((1, 5, 5))))  # doctest: +SKIP
        [[[0.40000004 0.53333336 0.6666667  0.53333336 0.40000004]
          [0.6        0.8        1.         0.8        0.6       ]
          [0.6        0.8        1.         0.8        0.6       ]
          [0.6        0.8        1.         0.8        0.6       ]
          [0.40000004 0.53333336 0.6666667  0.53333336 0.40000004]]]
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        kernel_x, kernel_y = jax.lax.stop_gradient((self.kernel_x, self.kernel_y))
        x = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel_x)
        x = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel_y.T)
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
        >>> print(layer(jnp.ones((1, 5, 5))))
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
        self.kernel = kernel.astype(dtype)

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        kernel = jax.lax.stop_gradient_p.bind(self.kernel)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, kernel)
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
        >>> print(layer(jnp.ones((1, 5, 5))))  # doctest: +SKIP
        [[[4.0000005 6.0000005 6.000001  6.0000005 4.0000005]
          [6.0000005 9.        9.        9.        6.0000005]
          [6.0000005 9.        9.        9.        6.0000005]
          [6.0000005 9.        9.        9.        6.0000005]
          [4.        6.0000005 6.0000005 6.0000005 4.       ]]]
    """

    def __init__(self, kernel: jax.Array, *, dtype: DType = jnp.float32):
        if kernel.ndim != 2:
            raise ValueError("Expected `kernel` to be a 2D `ndarray` with shape (H, W)")

        self.kernel = kernel.astype(dtype)

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        kernel = jax.lax.stop_gradient_p.bind(self.kernel)
        x = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel)
        return x

    @property
    def spatial_ndim(self) -> int:
        return 2
