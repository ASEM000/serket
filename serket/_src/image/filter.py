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
from serket._src.image.geometric import rotate_2d
from serket._src.nn.convolution import fft_conv_general_dilated
from serket._src.nn.initialization import DType
from serket._src.utils import (
    CHWArray,
    HWArray,
    canonicalize,
    generate_conv_dim_numbers,
    kernel_map,
    resolve_string_padding,
    validate_spatial_nd,
)

# For filters that have fft implementation, the pattern is to inherit from
# a base class that creates the kernel and then the child class implements the
# specific implementation of the filter, either fft or direct convolution.


def filter_2d(array: HWArray, weight: HWArray) -> HWArray:
    """Filtering wrapping ``jax.lax.conv_general_dilated``.

    Args:
        array: 2D input array. shape is (height, width).
        weight: convolutional kernel. shape is (height, width).

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
        padding="same",
        rhs_dilation=(1, 1),
        dimension_numbers=generate_conv_dim_numbers(2),
        feature_group_count=array.shape[0],  # in_features
    )
    return jnp.squeeze(x, (0, 1))


def fft_filter_2d(array: HWArray, weight: HWArray) -> HWArray:
    """Filtering wrapping ``serket`` ``fft_conv_general_dilated``

    Args:
        array: 2D input array. shape is (height, width).
        weight: convolutional kernel. shape is (height, width).

    Note:
        - To filter 3D array, channel-wise use ``jax.vmap(filter_2d, in_axes=(0, None))``.
    """
    assert array.ndim == 2
    assert weight.ndim == 2

    array = jnp.expand_dims(array, 0)
    weight = jnp.expand_dims(weight, (0, 1))

    padding = resolve_string_padding(
        in_dim=array.shape[1:],
        padding="same",
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
    return jnp.squeeze(x, (0, 1))


def calculate_average_kernel(kernel_size: int, dtype: DType) -> HWArray:
    """Calculate average kernel.

    Args:
        kernel_size: size of the convolving kernel. Accept an int.
        dtype: data type of the kernel.

    Returns:
        Average kernel. shape is (1, kernel_size).
    """
    kernel = jnp.ones((kernel_size), dtype=dtype)
    kernel = kernel / jnp.sum(kernel)
    kernel = kernel.astype(dtype)
    kernel = jnp.expand_dims(kernel, 0)
    return kernel


def calculate_gaussian_kernel(kernel_size: int, sigma: float, dtype: DType) -> HWArray:
    """Calculate gaussian kernel.

    Args:
        kernel_size: size of the convolving kernel. Accept an int.
        sigma: sigma of gaussian kernel.
        dtype: data type of the kernel.

    Returns:
        gaussian kernel. shape is (1, kernel_size).
    """
    x = jnp.arange(kernel_size, dtype=dtype) - kernel_size // 2
    x = x + 0.5 if kernel_size % 2 == 0 else x
    kernel = jnp.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / jnp.sum(kernel)
    kernel = kernel.astype(dtype)
    kernel = jnp.expand_dims(kernel, (0))
    return kernel


def calculate_box_kernel(kernel_size: int, dtype: DType) -> HWArray:
    """Calculate box kernel.

    Args:
        kernel_size: size of the convolving kernel. Accept an int.
        dtype: data type of the kernel.

    Returns:
        Box kernel. shape is (1, kernel_size).
    """
    kernel = jnp.ones((kernel_size))
    kernel = kernel.astype(dtype)
    kernel = jnp.expand_dims(kernel, 0)
    return kernel / kernel_size


def calculate_laplacian_kernel(kernel_size: tuple[int, int], dtype: DType) -> HWArray:
    """Calculate laplacian kernel.

    Args:
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
        dtype: data type of the kernel.

    Returns:
        Laplacian kernel. shape is (kernel_size[0], kernel_size[1]).
    """
    ky, kx = kernel_size
    kernel = jnp.ones((ky, kx))
    kernel = kernel.at[ky // 2, kx // 2].set(1 - jnp.sum(kernel)).astype(dtype)
    return kernel


def calculate_motion_kernel(
    kernel_size: int,
    angle: float,
    direction,
    dtype: DType,
) -> HWArray:
    """Returns 2D motion blur filter.

    Args:
        kernel_size: motion kernel width and height. It should be odd and positive.
        angle: angle of the motion blur in degrees (anti-clockwise rotation).
        direction: direction of the motion blur.

    Returns:
        The motion blur kernel of shape (kernel_size, kernel_size).
    """
    kernel = jnp.zeros((kernel_size, kernel_size))
    direction = (jnp.clip(direction, -1.0, 1.0) + 1.0) / 2.0
    indices = jnp.arange(kernel_size)
    set_value = direction + ((1 - 2 * direction) / (kernel_size - 1)) * indices
    kernel = kernel.at[kernel_size // 2, indices].set(set_value)
    kernel = rotate_2d(kernel, angle)
    kernel = kernel / jnp.sum(kernel)
    return kernel.astype(dtype)


def calculate_sobel_kernel(dtype: DType = jnp.float32):
    """Calculate sobel kernel."""
    # used in separable manner
    x0 = jnp.array([1, 2, 1])
    x1 = jnp.array([1, 0, -1])
    y0 = jnp.array([1, 0, -1])
    y1 = jnp.array([1, 2, 1])
    return jnp.stack([x0, x1, y0, y1], axis=0).astype(dtype)


@ft.partial(jax.jit, inline=True, static_argnums=1)
def median_blur_2d(array: HWArray, kernel_size: tuple[int, int]) -> HWArray:
    """Median blur"""
    assert array.ndim == 2
    # def resolve_string_padding(in_dim, padding, kernel_size, strides):
    padding = resolve_string_padding(
        in_dim=array.shape,
        padding="same",
        kernel_size=kernel_size,
        strides=(1, 1),
    )

    @ft.partial(
        kernel_map,
        shape=array.shape,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding=padding,
    )
    def median_kernel(array: jax.Array) -> jax.Array:
        return jnp.median(array)

    return median_kernel(array)


def elastic_transform_2d(
    key: jax.Array,
    image: HWArray,
    kernel_y: HWArray,
    kernel_x: HWArray,
    alpha: tuple[float, float],
):
    ay, ax = alpha
    k1, k2 = jr.split(key)
    noise_y = jr.uniform(k1, shape=image.shape, dtype=image.dtype) * 2 - 1
    noise_x = jr.uniform(k2, shape=image.shape, dtype=image.dtype) * 2 - 1
    dy = filter_2d(filter_2d(noise_y, kernel_y), kernel_y.T) * ay
    dx = filter_2d(filter_2d(noise_x, kernel_x), kernel_x.T) * ax
    r, c = image.shape
    ny, nx = jnp.meshgrid(jnp.arange(r), jnp.arange(c), indexing="ij")
    ny = (ny + dy).reshape(-1, 1)
    nx = (nx + dx).reshape(-1, 1)
    return map_coordinates(image, (ny, nx), order=1, mode="nearest").reshape(r, c)


def fft_elastic_transform_2d(
    key: jax.Array,
    image: HWArray,
    kernel_y: HWArray,
    kernel_x: HWArray,
    alpha: tuple[float, float],
):
    ay, ax = alpha
    k1, k2 = jr.split(key)
    noise_y = jr.uniform(k1, shape=image.shape, dtype=image.dtype) * 2 - 1
    noise_x = jr.uniform(k2, shape=image.shape, dtype=image.dtype) * 2 - 1
    dy = fft_filter_2d(fft_filter_2d(noise_y, kernel_y), kernel_y.T) * ay
    dx = fft_filter_2d(fft_filter_2d(noise_x, kernel_x), kernel_x.T) * ax
    r, c = image.shape
    ny, nx = jnp.meshgrid(jnp.arange(r), jnp.arange(c), indexing="ij")
    ny = (ny + dy).reshape(-1, 1)
    nx = (nx + dx).reshape(-1, 1)
    return map_coordinates(image, (ny, nx), order=1, mode="nearest").reshape(r, c)


class BaseAvgBlur2D(sk.TreeClass):
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

    spatial_ndim: int = 2


class AvgBlur2D(BaseAvgBlur2D):
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
    def __call__(self, x: CHWArray) -> CHWArray:
        kernel_x, kernel_y = jax.lax.stop_gradient((self.kernel_x, self.kernel_y))
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, kernel_x)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, kernel_y.T)
        return x


class FFTAvgBlur2D(BaseAvgBlur2D):
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
    def __call__(self, x: CHWArray) -> CHWArray:
        kernel_x, kernel_y = jax.lax.stop_gradient((self.kernel_x, self.kernel_y))
        x = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel_x)
        x = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel_y.T)
        return x


class BaseGaussianBlur2D(sk.TreeClass):
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

    spatial_ndim: int = 2


class GaussianBlur2D(BaseGaussianBlur2D):
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
    def __call__(self, x: CHWArray) -> CHWArray:
        kernel_x, kernel_y = jax.lax.stop_gradient((self.kernel_x, self.kernel_y))
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, kernel_x)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, kernel_y.T)
        return x


class FFTGaussianBlur2D(BaseGaussianBlur2D):
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
    def __call__(self, x: CHWArray) -> CHWArray:
        kernel_x, kernel_y = jax.lax.stop_gradient((self.kernel_x, self.kernel_y))
        x = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel_x)
        x = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel_y.T)
        return x


class UnsharpMask2D(BaseGaussianBlur2D):
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
    def __call__(self, x: CHWArray) -> CHWArray:
        kernel_x, kernel_y = jax.lax.stop_gradient((self.kernel_x, self.kernel_y))
        blur = jax.vmap(filter_2d, in_axes=(0, None))(x, kernel_x)
        blur = jax.vmap(filter_2d, in_axes=(0, None))(blur, kernel_y.T)
        return x + (x - blur)


class FFTUnsharpMask2D(BaseGaussianBlur2D):
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
    def __call__(self, x: CHWArray) -> CHWArray:
        kernel_x, kernel_y = jax.lax.stop_gradient((self.kernel_x, self.kernel_y))
        blur = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel_x)
        blur = jax.vmap(fft_filter_2d, in_axes=(0, None))(blur, kernel_y.T)
        return x + (x - blur)


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

    spatial_ndim: int = 2


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
    def __call__(self, x: CHWArray) -> CHWArray:
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
    def __call__(self, x: CHWArray) -> CHWArray:
        kernel_x, kernel_y = jax.lax.stop_gradient((self.kernel_x, self.kernel_y))
        x = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel_x)
        x = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel_y.T)
        return x


class Laplacian2DBase(sk.TreeClass):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        *,
        dtype: DType = jnp.float32,
    ):
        self.kernel_size = canonicalize(kernel_size, ndim=2, name="kernel_size")
        self.kernel = calculate_laplacian_kernel(self.kernel_size, dtype)

    spatial_ndim: int = 2


class Laplacian2D(Laplacian2DBase):
    """Apply Laplacian filter to a channel-first image.

    .. image:: ../_static/laplacian2d.png

    Args:
        kernel_size: size of the convolving kernel. Accepts int or tuple of two ints.
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.Laplacian2D(kernel_size=(3, 5))
        >>> print(layer(jnp.ones((1, 5, 5))))  # doctest: +SKIP
        [[[-9. -7. -5. -7. -9.]
          [-6. -3.  0. -3. -6.]
          [-6. -3.  0. -3. -6.]
          [-6. -3.  0. -3. -6.]
          [-9. -7. -5. -7. -9.]]]

    Note:
        The laplacian considers all the neighbors of a pixel.
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: CHWArray) -> CHWArray:
        kernel = jax.lax.stop_gradient_p.bind(self.kernel)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, kernel)
        return x


class FFTLaplacian2D(Laplacian2DBase):
    """Apply Laplacian filter to a channel-first image using FFT.

    .. image:: ../_static/laplacian2d.png

    Args:
        kernel_size: size of the convolving kernel. Accepts int or tuple of two ints.
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.FFTLaplacian2D(kernel_size=(3, 5))
        >>> print(layer(jnp.ones((1, 5, 5))))  # doctest: +SKIP
        [[[-9. -7. -5. -7. -9.]
          [-6. -3.  0. -3. -6.]
          [-6. -3.  0. -3. -6.]
          [-6. -3.  0. -3. -6.]
          [-9. -7. -5. -7. -9.]]]

    Note:
        The laplacian considers all the neighbors of a pixel.
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: CHWArray) -> CHWArray:
        kernel = jax.lax.stop_gradient_p.bind(self.kernel)
        x = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel)
        return x


class MotionBlur2DBase(sk.TreeClass):
    def __init__(
        self,
        kernel_size: int,
        *,
        angle: float = 0.0,
        direction: float = 0.0,
        dtype: DType = jnp.float32,
    ):
        self.kernel_size = kernel_size
        self.angle = angle
        self.direction = direction
        args = (self.kernel_size, self.angle, self.direction, dtype)
        self.kernel = calculate_motion_kernel(*args)

    spatial_ndim: int = 2


class MotionBlur2D(MotionBlur2DBase):
    """Apply motion blur to a channel-first image.

    .. image:: ../_static/motionblur2d.png

    Args:
        kernel_size: motion kernel width and height. It should be odd and positive.
        angle: angle of the motion blur in degrees (anti-clockwise rotation).
        direction: direction of the motion blur.
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 17).reshape(1, 4, 4) + 0.0
        >>> print(sk.image.MotionBlur2D(3, angle=30, direction=0.5)(x))  # doctest: +SKIP
        [[[ 0.7827108  2.4696379  3.3715053  3.8119273]
          [ 2.8356633  6.3387947  7.3387947  7.1810846]
          [ 5.117592  10.338796  11.338796  10.550241 ]
          [ 6.472714  10.020969  10.770187   9.100007 ]]]
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: CHWArray) -> CHWArray:
        kernel = jax.lax.stop_gradient_p.bind(self.kernel)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, kernel)
        return x


class FFTMotionBlur2D(MotionBlur2DBase):
    """Apply motion blur to a channel-first image using FFT.

    .. image:: ../_static/motionblur2d.png

    Args:
        kernel_size: motion kernel width and height. It should be odd and positive.
        angle: angle of the motion blur in degrees (anti-clockwise rotation).
        direction: direction of the motion blur.
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 17).reshape(1, 4, 4) + 0.0
        >>> print(sk.image.MotionBlur2D(3, angle=30, direction=0.5)(x))  # doctest: +SKIP
        [[[ 0.7827108  2.4696379  3.3715053  3.8119273]
          [ 2.8356633  6.3387947  7.3387947  7.1810846]
          [ 5.117592  10.338796  11.338796  10.550241 ]
          [ 6.472714  10.020969  10.770187   9.100007 ]]]
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: CHWArray) -> CHWArray:
        kernel = jax.lax.stop_gradient_p.bind(self.kernel)
        x = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel)
        return x


class MedianBlur2D(sk.TreeClass):
    """Apply median filter to a channel-first image.

    .. image:: ../_static/medianblur2d.png

    Args:
        kernel_size: size of the convolving kernel. Accepts int or tuple of two ints.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5) + 0.0
        >>> print(x)
        [[[ 1.  2.  3.  4.  5.]
          [ 6.  7.  8.  9. 10.]
          [11. 12. 13. 14. 15.]
          [16. 17. 18. 19. 20.]
          [21. 22. 23. 24. 25.]]]
        >>> print(sk.image.MedianBlur2D(3)(x))
        [[[ 0.  2.  3.  4.  0.]
          [ 2.  7.  8.  9.  5.]
          [ 7. 12. 13. 14. 10.]
          [12. 17. 18. 19. 15.]
          [ 0. 17. 18. 19.  0.]]]
    """

    def __init__(self, kernel_size: int | tuple[int, int]):
        self.kernel_size = canonicalize(kernel_size, ndim=2, name="kernel_size")

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: CHWArray) -> CHWArray:
        x = jax.vmap(median_blur_2d, in_axes=(0, None))(x, self.kernel_size)
        return x

    spatial_ndim: int = 2


class Sobel2DBase(sk.TreeClass):
    def __init__(self, *, dtype: DType = jnp.float32):
        self.kernel = calculate_sobel_kernel(dtype)

    spatial_ndim: int = 2


class Sobel2D(Sobel2DBase):
    """Apply Sobel filter to a channel-first image.

    .. image:: ../_static/sobel2d.png

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1,26).reshape(1, 5,5).astype(jnp.float32)
        >>> layer = sk.image.Sobel2D()
        >>> layer(x)  # doctest: +SKIP
        [[[21.954498, 28.635643, 32.55764 , 36.496574, 33.61547 ],
          [41.036568, 40.792156, 40.792156, 40.792156, 46.8615  ],
          [56.603886, 40.792156, 40.792156, 40.792156, 63.529522],
          [74.323616, 40.792156, 40.792156, 40.792156, 81.706795],
          [78.24321 , 68.26419 , 72.249565, 76.23647 , 89.27486 ]]]
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: CHWArray) -> CHWArray:
        kernel = jax.lax.stop_gradient_p.bind(self.kernel)
        gx0, gx1, gy0, gy1 = jnp.split(kernel, 4)
        gx = jax.vmap(filter_2d, in_axes=(0, None))(x, gx0)
        gx = jax.vmap(filter_2d, in_axes=(0, None))(gx, gx1.T)
        gy = jax.vmap(filter_2d, in_axes=(0, None))(x, gy0)
        gy = jax.vmap(filter_2d, in_axes=(0, None))(gy, gy1.T)
        return jnp.sqrt(gx**2 + gy**2)


class FFTSobel2D(Sobel2DBase):
    """Apply Sobel filter to a channel-first image using FFT.

    .. image:: ../_static/sobel2d.png

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1,26).reshape(1, 5,5).astype(jnp.float32)
        >>> layer = sk.image.FFTSobel2D()
        >>> layer(x)  # doctest: +SKIP
        [[[21.954498, 28.635643, 32.55764 , 36.496574, 33.61547 ],
          [41.036568, 40.792156, 40.792156, 40.792156, 46.8615  ],
          [56.603886, 40.792156, 40.792156, 40.792156, 63.529522],
          [74.323616, 40.792156, 40.792156, 40.792156, 81.706795],
          [78.24321 , 68.26419 , 72.249565, 76.23647 , 89.27486 ]]]
    """

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: CHWArray) -> CHWArray:
        kernel = jax.lax.stop_gradient_p.bind(self.kernel)
        gx0, gx1, gy0, gy1 = jnp.split(kernel, 4)
        gx = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, gx0)
        gx = jax.vmap(fft_filter_2d, in_axes=(0, None))(gx, gx1.T)
        gy = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, gy0)
        gy = jax.vmap(fft_filter_2d, in_axes=(0, None))(gy, gy1.T)
        return jnp.sqrt(gx**2 + gy**2)


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
        >>> print(layer(jnp.ones((1, 5, 5))))  # doctest: +SKIP
        [[[4. 6. 6. 6. 4.]
          [6. 9. 9. 9. 6.]
          [6. 9. 9. 9. 6.]
          [6. 9. 9. 9. 6.]
          [4. 6. 6. 6. 4.]]]
    """

    def __init__(self, kernel: HWArray, *, dtype: DType = jnp.float32):
        self.kernel = kernel.astype(dtype)

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: CHWArray) -> CHWArray:
        kernel = jax.lax.stop_gradient_p.bind(self.kernel)
        x = jax.vmap(filter_2d, in_axes=(0, None))(x, kernel)
        return x

    spatial_ndim: int = 2


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

    def __init__(self, kernel: HWArray, *, dtype: DType = jnp.float32):
        self.kernel = kernel.astype(dtype)

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: CHWArray) -> CHWArray:
        kernel = jax.lax.stop_gradient_p.bind(self.kernel)
        x = jax.vmap(fft_filter_2d, in_axes=(0, None))(x, kernel)
        return x

    spatial_ndim: int = 2


class ElasticTransform2DBase(sk.TreeClass):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        sigma: float | tuple[float, float],
        alpha: float | tuple[float, float],
        *,
        dtype=jnp.float32,
    ):
        self.kernel_size = canonicalize(kernel_size, ndim=2, name="kernel_size")
        self.alpha = canonicalize(alpha, ndim=2, name="alpha")
        self.sigma = canonicalize(sigma, ndim=2, name="sigma")
        ky, kx = self.kernel_size
        sy, sx = self.sigma
        self.kernel_y = calculate_gaussian_kernel(ky, sy, dtype)
        self.kernel_x = calculate_gaussian_kernel(kx, sx, dtype)


class ElasticTransform2D(ElasticTransform2DBase):
    """Apply an elastic transform to an image.

    .. image:: ../_static/elastictransform2d.png

    Args:
        kernel_size: The size of the Gaussian kernel. either a single integer or
            a tuple of two integers.
        sigma: The standard deviation of the Gaussian in the y and x directions,
        alpha : The scaling factor that controls the intensity of the deformation
            in the y and x directions, respectively.

    Example:
        >>> import serket as sk
        >>> import jax.random as jr
        >>> import jax.numpy as jnp
        >>> layer = sk.image.ElasticTransform2D(kernel_size=3, sigma=1.0, alpha=1.0)
        >>> key = jr.PRNGKey(0)
        >>> image = jnp.arange(1, 26).reshape(1, 5, 5).astype(jnp.float32)
        >>> print(layer(image, key=key))  # doctest: +SKIP
        [[[ 1.0669159  2.2596366  3.210071   3.9703817  4.9207525]
          [ 5.70821    7.483665   8.857002   8.663773   8.794132 ]
          [13.809857  15.865877  15.109764  12.897442  13.0018215]
          [18.35189   18.817993  17.2193    15.731948  17.026705 ]
          [21.        21.659977  21.43855   21.138866  22.583244 ]]]
    """

    def __call__(self, image: CHWArray, *, key: jax.Array) -> CHWArray:
        in_axes = (None, 0, None, None, None)
        args = jax.lax.stop_gradient((self.kernel_y, self.kernel_x, self.alpha))
        return jax.vmap(elastic_transform_2d, in_axes=in_axes)(key, image, *args)


class FFTElasticTransform2D(ElasticTransform2DBase):
    """Apply an elastic transform to an image using FFT.

    .. image:: ../_static/elastictransform2d.png

    Args:
        kernel_size: The size of the Gaussian kernel. either a single integer or
            a tuple of two integers.
        sigma: The standard deviation of the Gaussian in the y and x directions,
        alpha : The scaling factor that controls the intensity of the deformation
            in the y and x directions, respectively.

    Example:
        >>> import serket as sk
        >>> import jax.random as jr
        >>> import jax.numpy as jnp
        >>> layer = sk.image.FFTElasticTransform2D(kernel_size=3, sigma=1.0, alpha=1.0)
        >>> key = jr.PRNGKey(0)
        >>> image = jnp.arange(1, 26).reshape(1, 5, 5).astype(jnp.float32)
        >>> print(layer(image, key=key))  # doctest: +SKIP
        [[[ 1.0669159  2.2596366  3.210071   3.9703817  4.9207525]
          [ 5.70821    7.483665   8.857002   8.663773   8.794132 ]
          [13.809857  15.865877  15.109764  12.897442  13.0018215]
          [18.35189   18.817993  17.2193    15.731948  17.026705 ]
          [21.        21.659977  21.43855   21.138866  22.583244 ]]]
    """

    def __call__(self, image: CHWArray, *, key: jax.Array) -> CHWArray:
        in_axes = (None, 0, None, None, None)
        args = jax.lax.stop_gradient((self.kernel_y, self.kernel_x, self.alpha))
        return jax.vmap(fft_elastic_transform_2d, in_axes=in_axes)(key, image, *args)
