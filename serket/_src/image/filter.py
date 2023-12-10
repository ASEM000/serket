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

import abc
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
    delayed_canonicalize_padding,
    generate_conv_dim_numbers,
    kernel_map,
    resolve_string_padding,
    validate_spatial_ndim,
)


def filter_2d(
    image: HWArray,
    weight: HWArray,
    strides: tuple[int, int] = (1, 1),
) -> HWArray:
    """Filtering wrapping ``jax.lax.conv_general_dilated``.

    Args:
        image: 2D input array. shape is ``(height, width)``.
        weight: convolutional kernel. shape is ``(height, width)``.
        strides: stride of the convolution. Accepts tuple of two ints.
    """
    assert image.ndim == 2
    assert weight.ndim == 2

    image = jnp.expand_dims(image, 0)
    weight = jnp.expand_dims(weight, (0, 1))

    x = jax.lax.conv_general_dilated(
        lhs=jnp.expand_dims(image, 0),
        rhs=weight,
        window_strides=strides,
        padding="same",
        rhs_dilation=(1, 1),
        dimension_numbers=generate_conv_dim_numbers(2),
        feature_group_count=1,
    )
    return jnp.squeeze(x, (0, 1))


def fft_filter_2d(
    image: HWArray,
    weight: HWArray,
    strides: tuple[int, int] = (1, 1),
) -> HWArray:
    """Filtering wrapping ``serket`` ``fft_conv_general_dilated``

    Args:
        image: 2D input array. shape is (height, width).
        weight: convolutional kernel. shape is (height, width).
        strides: stride of the convolution. Accepts tuple of two ints.
    """
    assert image.ndim == 2
    assert weight.ndim == 2

    array = jnp.expand_dims(image, 0)
    weight = jnp.expand_dims(weight, (0, 1))

    padding = resolve_string_padding(
        in_dim=array.shape[1:],
        padding="same",
        kernel_size=weight.shape[2:],
        strides=strides,
    )
    x = fft_conv_general_dilated(
        lhs=jnp.expand_dims(array, 0),
        rhs=weight,
        strides=strides,
        padding=padding,
        dilation=(1, 1),
        groups=1,
    )
    return jnp.squeeze(x, (0, 1))


def calculate_average_kernel_1d(kernel_size: int, dtype: DType) -> HWArray:
    """Calculate average kernel.

    Args:
        kernel_size: size of the convolving kernel. Accept an int.
        dtype: data type of the kernel.

    Returns:
        Average kernel. shape is (kernel_size, ).
    """
    kernel = jnp.ones((kernel_size), dtype=dtype)
    kernel = kernel / jnp.sum(kernel)
    return kernel.astype(dtype)


def avg_blur_2d(
    image: HWArray,
    kernel_size: tuple[int, int],
    dtype: DType | None = None,
) -> HWArray:
    """Average blur.

    Args:
        image: 2D input array. shape is (height, width).
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
        dtype: data type of the kernel. Defaults to ``None`` to use the same
            data type as ``array``.

    Returns:
        Average blurred array. shape is (height, width).
    """
    dtype = dtype or image.dtype
    (ky, kx) = kernel_size
    kernel_x = calculate_average_kernel_1d(kx, dtype)[None]  # (1, kx)
    kernel_y = calculate_average_kernel_1d(ky, dtype)[None]  # (1, ky)
    return filter_2d(filter_2d(image, kernel_x), kernel_y.T)


def fft_avg_blur_2d(
    image: HWArray,
    kernel_size: tuple[int, int],
    dtype: DType | None = None,
) -> HWArray:
    """Average blur using FFT.

    Args:
        image: 2D input array. shape is ``(height, width)``.
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
        dtype: data type of the kernel. Defaults to ``None`` to use the same
            data type as ``array``.

    Returns:
        Average blurred array. shape is ``(height, width)``.
    """
    dtype = dtype or image.dtype
    (ky, kx) = kernel_size
    kernel_x = calculate_average_kernel_1d(kx, dtype)[None]  # (1, kx)
    kernel_y = calculate_average_kernel_1d(ky, dtype)[None]  # (1, ky)
    return fft_filter_2d(fft_filter_2d(image, kernel_x), kernel_y.T)


def calculate_gaussian_kernel_1d(
    kernel_size: int,
    sigma: float,
    dtype: DType,
) -> HWArray:
    """Calculate gaussian kernel.

    Args:
        kernel_size: size of the convolving kernel. Accept an int.
        sigma: sigma of gaussian kernel.
        dtype: data type of the kernel.

    Returns:
        gaussian kernel. shape is (kernel_size, ).
    """
    x = jnp.arange(kernel_size, dtype=dtype) - kernel_size // 2
    x = x + 0.5 if kernel_size % 2 == 0 else x
    kernel = jnp.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / jnp.sum(kernel)
    return kernel.astype(dtype)


def gaussian_blur_2d(
    image: HWArray,
    kernel_size: tuple[int, int],
    sigma: tuple[float, float],
    dtype: DType | None = None,
) -> HWArray:
    """Gaussian blur.

    Args:
        image: 2D input array. shape is ``(height, width)``.
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
        sigma: sigma of gaussian kernel.
        dtype: data type of the kernel. Defaults to ``None`` to use the same
            data type as ``array``.
    """
    dtype = dtype or image.dtype
    (ky, kx), (sy, sx) = kernel_size, sigma
    gy = calculate_gaussian_kernel_1d(ky, sy, dtype)[None]
    gx = calculate_gaussian_kernel_1d(kx, sx, dtype)[None]
    return filter_2d(filter_2d(image, gy), gx.T)


def fft_gaussian_blur_2d(
    image: HWArray,
    kernel_size: tuple[int, int],
    sigma: tuple[float, float],
    dtype: DType | None = None,
) -> HWArray:
    """Gaussian blur using FFT.

    Args:
        image: 2D input array. shape is ``(height, width)``.
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
        sigma: sigma of gaussian kernel.
        dtype: data type of the kernel. Defaults to ``None`` to use the same
            data type as ``array``.
    """
    dtype = dtype or image.dtype
    (ky, kx), (sy, sx) = kernel_size, sigma
    gy = calculate_gaussian_kernel_1d(ky, sy, dtype)[None]
    gx = calculate_gaussian_kernel_1d(kx, sx, dtype)[None]
    return fft_filter_2d(fft_filter_2d(image, gy), gx.T)


def unsharp_mask_2d(
    image: HWArray,
    kernel_size: tuple[int, int],
    sigma: tuple[float, float],
    dtype: DType | None = None,
) -> HWArray:
    """Unsharp mask.

    Args:
        image: 2D input array. shape is ``(height, width)``.
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
        sigma: sigma of gaussian kernel.
        dtype: data type of the kernel. Defaults to ``None`` to use the same
            data type as ``array``.

    Returns:
        Unsharp masked array. shape is ``(height, width)``.
    """
    dtype = dtype or image.dtype
    (ky, kx), (sy, sx) = kernel_size, sigma
    gy = calculate_gaussian_kernel_1d(ky, sy, dtype)[None]
    gx = calculate_gaussian_kernel_1d(kx, sx, dtype)[None]
    blur = filter_2d(filter_2d(image, gy), gx.T)
    return image + (image - blur)


def fft_unsharp_mask_2d(
    image: HWArray,
    kernel_size: tuple[int, int],
    sigma: tuple[float, float],
    dtype: DType | None = None,
) -> HWArray:
    """Unsharp mask using FFT.

    Args:
        image: 2D input array. shape is (height, width).
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
        sigma: sigma of gaussian kernel.
        dtype: data type of the kernel. Defaults to ``None`` to use the same
            data type as ``array``.

    Returns:
        Unsharp masked array. shape is (height, width).
    """
    dtype = dtype or image.dtype
    (ky, kx), (sy, sx) = kernel_size, sigma
    gy = calculate_gaussian_kernel_1d(ky, sy, dtype)[None]
    gx = calculate_gaussian_kernel_1d(kx, sx, dtype)[None]
    blur = fft_filter_2d(fft_filter_2d(image, gy), gx.T)
    return image + (image - blur)


def calculate_box_kernel_1d(kernel_size: int, dtype: DType) -> HWArray:
    """Calculate box kernel.

    Args:
        kernel_size: size of the convolving kernel. Accept an int.
        dtype: data type of the kernel.

    Returns:
        Box kernel. shape is (1, kernel_size).
    """
    kernel = jnp.ones((kernel_size))
    kernel = kernel.astype(dtype)
    return kernel / kernel_size


def box_blur_2d(
    image: HWArray,
    kernel_size: tuple[int, int],
    dtype: DType | None = None,
) -> HWArray:
    """Box blur.

    Args:
        image: 2D input array. shape is ``(height, width)``.
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
        dtype: data type of the kernel. Defaults to ``None`` to use the same
            data type as ``array``.

    Returns:
        Box blurred array. shape is ``(height, width)``.
    """
    dtype = dtype or image.dtype
    (ky, kx) = kernel_size
    kernel_x = calculate_box_kernel_1d(kx, dtype)[None]
    kernel_y = calculate_box_kernel_1d(ky, dtype)[None]
    return filter_2d(filter_2d(image, kernel_x), kernel_y.T)


def fft_box_blur_2d(
    image: HWArray,
    kernel_size: tuple[int, int],
    dtype: DType | None = None,
) -> HWArray:
    """Box blur using FFT.

    Args:
        image: 2D input array. shape is ``(height, width)``.
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
        dtype: data type of the kernel. Defaults to ``None`` to use the same
            data type as ``array``.

    Returns:
        Box blurred array. shape is ``(height, width)``.
    """
    dtype = dtype or image.dtype
    (ky, kx) = kernel_size
    kernel_x = calculate_box_kernel_1d(kx, dtype)[None]
    kernel_y = calculate_box_kernel_1d(ky, dtype)[None]
    return fft_filter_2d(fft_filter_2d(image, kernel_x), kernel_y.T)


def calculate_laplacian_kernel_2d(
    kernel_size: tuple[int, int], dtype: DType
) -> HWArray:
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


def laplacian_2d(
    image: HWArray,
    kernel_size: tuple[int, int],
    dtype: DType | None = None,
) -> HWArray:
    """Laplacian.

    Args:
        image: 2D input array. shape is ``(height, width)``.
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
        dtype: data type of the kernel. Defaults to ``None`` to use the same
            data type as ``array``.

    Returns:
        Laplacian array. shape is ``(height, width)``.
    """
    dtype = dtype or image.dtype
    kernel = calculate_laplacian_kernel_2d(kernel_size, dtype)
    return filter_2d(image, kernel)


def fft_laplacian_2d(
    image: HWArray,
    kernel_size: tuple[int, int],
    dtype: DType | None = None,
) -> HWArray:
    """Laplacian using FFT.

    Args:
        image: 2D input array. shape is ``(height, width)``.
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
        dtype: data type of the kernel. Defaults to ``None`` to use the same
            data type as ``array``.

    Returns:
        Laplacian array. shape is ``(height, width)``.
    """
    dtype = dtype or image.dtype
    kernel = calculate_laplacian_kernel_2d(kernel_size, dtype)
    return fft_filter_2d(image, kernel)


def calculate_motion_kernel_2d(
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


def motion_blur_2d(
    image: HWArray,
    kernel_size: tuple[int, int],
    angle: float,
    direction: int | float,
    dtype: DType | None = None,
) -> HWArray:
    """Motion blur.

    Args:
        image: 2D input array. shape is ``(height, width)``.
        kernel_size: motion kernel width and height. It should be odd and positive.
        angle: angle of the motion blur in degrees (anti-clockwise rotation).
        direction: direction of the motion blur in degrees (anti-clockwise rotation).
        dtype: data type of the kernel. Defaults to ``None`` to use the same
            data type as ``array``.
    """
    dtype = dtype or image.dtype
    kernel = calculate_motion_kernel_2d(kernel_size, angle, direction, dtype)
    return filter_2d(image, kernel)


def fft_motion_blur_2d(
    image: HWArray,
    kernel_size: tuple[int, int],
    angle: float,
    direction: int | float,
    dtype: DType | None = None,
) -> HWArray:
    """Motion blur using FFT.

    Args:
        image: 2D input array. shape is ``(height, width)``.
        kernel_size: motion kernel width and height. It should be odd and positive.
        angle: angle of the motion blur in degrees (anti-clockwise rotation).
        direction: direction of the motion blur in degrees (anti-clockwise rotation).
        dtype: data type of the kernel. Defaults to ``None`` to use the same
            data type as ``array``.
    """
    dtype = dtype or image.dtype
    kernel = calculate_motion_kernel_2d(kernel_size, angle, direction, dtype)
    return fft_filter_2d(image, kernel)


def calculate_sobel_kernel_2d(dtype: DType):
    """Calculate sobel kernel."""
    # used in separable manner
    x0 = jnp.array([1, 2, 1])
    x1 = jnp.array([1, 0, -1])
    y0 = jnp.array([1, 0, -1])
    y1 = jnp.array([1, 2, 1])
    return jnp.stack([x0, x1, y0, y1], axis=0).astype(dtype)


def sobel_2d(image: HWArray, dtype: DType | None = None) -> HWArray:
    """Sobel filter.

    Args:
        image: 2D input array. shape is ``(height, width)``.
        dtype: data type of the kernel.
    """
    dtype = dtype or image.dtype
    kernel = calculate_sobel_kernel_2d(dtype)
    gx0, gx1, gy0, gy1 = jnp.split(kernel, 4)
    gx = filter_2d(filter_2d(image, gx0), gx1.T)
    gy = filter_2d(filter_2d(image, gy0), gy1.T)
    return jnp.sqrt(gx**2 + gy**2)


def fft_sobel_2d(image: HWArray, dtype: DType | None = None) -> HWArray:
    """Sobel filter using FFT.

    Args:
        image: 2D input array. shape is ``(height, width)``.
        dtype: data type of the kernel. Defaults to ``None`` to use the same
            data type as ``array``.
    """
    dtype = dtype or image.dtype
    kernel = calculate_sobel_kernel_2d(dtype)
    gx0, gx1, gy0, gy1 = jnp.split(kernel, 4)
    gx = fft_filter_2d(fft_filter_2d(image, gx0), gx1.T)
    gy = fft_filter_2d(fft_filter_2d(image, gy0), gy1.T)
    return jnp.sqrt(gx**2 + gy**2)


@ft.partial(jax.jit, inline=True, static_argnums=1)
def median_blur_2d(image: HWArray, kernel_size: tuple[int, int]) -> HWArray:
    """Median blur

    Args:
        image: 2D input array. shape is ``(height, width)``.
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
    """
    assert image.ndim == 2

    padding = resolve_string_padding(
        in_dim=image.shape,
        padding="same",
        kernel_size=kernel_size,
        strides=(1, 1),
    )

    @ft.partial(
        kernel_map,
        shape=image.shape,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding=padding,
        padding_mode=0,
    )
    def median_kernel(array: jax.Array) -> jax.Array:
        return jnp.median(array)

    return median_kernel(image)


def elastic_transform_2d(
    key: jax.Array,
    image: HWArray,
    kernel_size: tuple[int, int],
    sigma: tuple[float, float],
    alpha: tuple[float, float],
    dtype: DType | None = None,
):
    """Elastic transform.

    Args:
        key: jax random key.
        image: 2D input array. shape is ``(height, width)``.
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
        sigma: sigma of gaussian kernel.
        alpha: alpha of gaussian kernel.
        dtype: data type of the kernel. Defaults to ``None`` to use the same
            data type as ``array``.
    """
    dtype = dtype or image.dtype
    (ky, kx), (ay, ax), (sy, sx) = kernel_size, alpha, sigma
    kernel_y = calculate_gaussian_kernel_1d(ky, sy, dtype)[None]
    kernel_x = calculate_gaussian_kernel_1d(kx, sx, dtype)[None]
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
    kernel_size: tuple[int, int],
    sigma: tuple[float, float],
    alpha: tuple[float, float],
    dtype: DType | None = None,
):
    """Elastic transform using FFT.

    Args:
        key: jax random key.
        image: 2D input array. shape is ``(height, width)``.
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
        sigma: sigma of gaussian kernel.
        alpha: alpha of gaussian kernel.
        dtype: data type of the kernel. Defaults to ``None`` to use the same
            data type as ``array``.
    """
    dtype = dtype or image.dtype
    (ky, kx), (ay, ax), (sy, sx) = kernel_size, alpha, sigma
    kernel_y = calculate_gaussian_kernel_1d(ky, sy, dtype)[None]
    kernel_x = calculate_gaussian_kernel_1d(kx, sx, dtype)[None]
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


@ft.partial(jax.jit, inline=True, static_argnums=1)
def bilateral_blur_2d(
    image: HWArray,
    kernel_size: tuple[int, int],
    sigma_space: tuple[float, float],
    sigma_color: float,
    dtype: DType | None = None,
):
    """Bilateral blur.

    Args:
        image: 2D input array. shape is ``(height, width)``.
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
        sigma_space: sigma of gaussian kernel.
        sigma_color: sigma of gaussian kernel.
        dtype: data type of the kernel. Defaults to ``None`` to use the same
            data type as ``array``.
    """
    dtype = dtype or image.dtype
    (ky, kx), (sy, sx) = kernel_size, sigma_space
    center_index = (ky // 2, kx // 2)
    gy = calculate_gaussian_kernel_1d(ky, sy, dtype=dtype)[None]
    gx = calculate_gaussian_kernel_1d(kx, sx, dtype=dtype)[None]
    space_kernel = gy.T @ gx
    padding = delayed_canonicalize_padding(
        image.shape,
        padding="same",
        kernel_size=kernel_size,
        strides=(1, 1),
    )

    @ft.partial(
        kernel_map,
        shape=image.shape,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding=padding,
        padding_mode=0,
    )
    def bilateral_blur_2d(array):
        color_distance = (array - array[center_index]) ** 2
        color_kernel = jnp.exp(-0.5 / sigma_color**2 * color_distance)
        kernel = color_kernel * space_kernel
        return jnp.sum(array * kernel) / jnp.sum(kernel)

    return bilateral_blur_2d(image)


@ft.partial(jax.jit, inline=True, static_argnums=2)
def joint_bilateral_blur_2d(
    image: HWArray,
    guidance: HWArray,
    kernel_size: tuple[int, int],
    sigma_space: tuple[float, float],
    sigma_color: float,
    dtype: DType | None = None,
):
    """Joint bilateral blur.

    Args:
        image: 2D input array. shape is ``(height, width)``.
        guidance: 2D guidance array. shape is ``(height, width)``.
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
        sigma_space: sigma of gaussian kernel.
        sigma_color: sigma of gaussian kernel.
        dtype: data type of the kernel. Defaults to ``None`` to use the same
            data type as ``array``.
    """
    dtype = dtype or image.dtype
    (ky, kx), (sy, sx) = kernel_size, sigma_space
    center_index = (ky // 2, kx // 2)
    gy = calculate_gaussian_kernel_1d(ky, sy, dtype=dtype)[None]
    gx = calculate_gaussian_kernel_1d(kx, sx, dtype=dtype)[None]
    space_kernel = gy.T @ gx
    padding = delayed_canonicalize_padding(
        image.shape,
        padding="same",
        kernel_size=kernel_size,
        strides=(1, 1),
    )

    @ft.partial(
        kernel_map,
        shape=(2, *image.shape),
        kernel_size=(2, *kernel_size),
        strides=(1, 1, 1),
        padding=((0, 0), *padding),
        padding_mode=0,
    )
    def joint_bilateral_blur(array_guidance):
        array, guidance = array_guidance
        color_distance = (guidance - guidance[center_index]) ** 2
        color_kernel = jnp.exp(-0.5 / sigma_color**2 * color_distance)
        kernel = color_kernel * space_kernel
        return jnp.sum(array * kernel) / jnp.sum(kernel)

    return jnp.squeeze(joint_bilateral_blur(jnp.stack([image, guidance], axis=0)), 0)


def calculate_pascal_kernel_1d(kernel_size: int):
    kernel = jnp.array([1.0], dtype=float)
    stencil = jnp.array([1.0, 1.0], dtype=float)
    for _ in range(1, kernel_size):
        kernel = jnp.convolve(kernel, stencil, mode="full")
    return kernel


def blur_pool_2d(
    image: HWArray,
    kernel_size: tuple[int, int],
    strides: tuple[int, int],
) -> HWArray:
    """Blur pooling see https://arxiv.org/abs/1904.11486

    Args:
        image: 2D input array. shape is ``(height, width)``.
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
        strides: stride of the convolution. Accepts tuple of two ints.
    """
    ky, kx = kernel_size
    py = calculate_pascal_kernel_1d(ky)
    px = calculate_pascal_kernel_1d(kx)
    kernel = jnp.outer(py, px)
    kernel = kernel / jnp.sum(kernel)  # normalize
    return filter_2d(image, kernel, strides)


def fft_blur_pool_2d(
    image: HWArray,
    kernel_size: tuple[int, int],
    strides: tuple[int, int],
) -> HWArray:
    """Blur pooling see https://arxiv.org/abs/1904.11486

    Args:
        image: 2D input array. shape is ``(height, width)``.
        kernel_size: size of the convolving kernel. Accepts tuple of two ints.
        strides: stride of the convolution. Accepts tuple of two ints.
    """
    ky, kx = kernel_size
    py = calculate_pascal_kernel_1d(ky)
    px = calculate_pascal_kernel_1d(kx)
    kernel = jnp.outer(py, px)
    kernel = kernel / jnp.sum(kernel)  # normalize
    return fft_filter_2d(image, kernel, strides)


class BaseAvgBlur2D(sk.TreeClass):
    def __init__(self, kernel_size: int | tuple[int, int]):
        self.kernel_size = canonicalize(kernel_size, ndim=2, name="kernel_size")

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        in_axes = (0, None)
        args = (image, self.kernel_size)
        return jax.vmap(type(self).filter_op, in_axes=in_axes)(*args)

    spatial_ndim: int = 2

    @property
    @abc.abstractmethod
    def filter_op(self):
        ...


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

    filter_op = avg_blur_2d


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

    filter_op = fft_avg_blur_2d


class BaseGaussianBlur2D(sk.TreeClass):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        *,
        sigma: float | tuple[float, float] = 1.0,
    ):
        self.kernel_size = canonicalize(kernel_size, ndim=2, name="kernel_size")
        self.sigma = canonicalize(sigma, ndim=2, name="sigma")

    spatial_ndim: int = 2

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        in_axes = (0, None, None)
        sigma = jax.lax.stop_gradient(self.sigma)
        args = (image, self.kernel_size, sigma)
        return jax.vmap(type(self).filter_op, in_axes=in_axes)(*args)

    @property
    @abc.abstractmethod
    def filter_op(self):
        ...


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

    filter_op = gaussian_blur_2d


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

    filter_op = fft_gaussian_blur_2d


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

    filter_op = unsharp_mask_2d


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

    filter_op = fft_unsharp_mask_2d


class BoxBlur2DBase(sk.TreeClass):
    def __init__(self, kernel_size: int | tuple[int, int]):
        self.kernel_size = canonicalize(kernel_size, ndim=2, name="kernel_size")

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        in_axes = (0, None)
        args = (image, self.kernel_size)
        return jax.vmap(type(self).filter_op, in_axes=in_axes)(*args)

    spatial_ndim: int = 2

    @property
    @abc.abstractmethod
    def filter_op(self):
        ...


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

    filter_op = box_blur_2d


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

    filter_op = fft_box_blur_2d


class Laplacian2DBase(sk.TreeClass):
    def __init__(self, kernel_size: int | tuple[int, int]):
        self.kernel_size = canonicalize(kernel_size, ndim=2, name="kernel_size")

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        in_axes = (0, None)
        args = (image, self.kernel_size)
        return jax.vmap(type(self).filter_op, in_axes=in_axes)(*args)

    spatial_ndim: int = 2

    @property
    @abc.abstractmethod
    def filter_op(self):
        ...


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

    filter_op = laplacian_2d


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

    filter_op = fft_laplacian_2d


class MotionBlur2DBase(sk.TreeClass):
    def __init__(
        self,
        kernel_size: int,
        *,
        angle: float = 0.0,
        direction: float = 0.0,
    ):
        self.kernel_size = kernel_size
        self.angle = angle
        self.direction = direction

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        in_axes = (0, None, None, None)
        angle, direction = jax.lax.stop_gradient((self.angle, self.direction))
        args = (image, self.kernel_size, angle, direction)
        return jax.vmap(type(self).filter_op, in_axes=in_axes)(*args)

    spatial_ndim: int = 2

    @property
    @abc.abstractmethod
    def filter_op(self):
        ...


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

    filter_op = motion_blur_2d


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

    filter_op = fft_motion_blur_2d


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

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        in_axes = (0, None)
        args = (image, self.kernel_size)
        return jax.vmap(median_blur_2d, in_axes=in_axes)(*args)

    spatial_ndim: int = 2


class Sobel2DBase(sk.TreeClass):
    spatial_ndim: int = 2

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        return jax.vmap(type(self).filter_op)(image)

    @property
    @abc.abstractmethod
    def filter_op(self):
        ...


class Sobel2D(Sobel2DBase):
    """Apply Sobel filter to a channel-first image.

    .. image:: ../_static/sobel2d.png

    Args:
        dtype: data type of the layer. Defaults to ``float32``.

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

    filter_op = sobel_2d


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

    filter_op = fft_sobel_2d


class ElasticTransform2DBase(sk.TreeClass):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        sigma: float | tuple[float, float],
        alpha: float | tuple[float, float],
    ):
        self.kernel_size = canonicalize(kernel_size, ndim=2, name="kernel_size")
        self.alpha = canonicalize(alpha, ndim=2, name="alpha")
        self.sigma = canonicalize(sigma, ndim=2, name="sigma")

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray, *, key: jax.Array) -> CHWArray:
        in_axes = (None, 0, None, None, None)
        args = (image, self.kernel_size, self.sigma, self.alpha)
        return jax.vmap(type(self).filter_op, in_axes=in_axes)(key, *args)

    @property
    @abc.abstractmethod
    def filter_op(self):
        ...

    spatial_ndim: int = 2


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

    filter_op = elastic_transform_2d


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

    filter_op = fft_elastic_transform_2d


class BilateralBlur2D(sk.TreeClass):
    """Apply bilateral blur to a channel-first image.

    .. image:: ../_static/bilateralblur2d.png

    Args:
        kernel_size: kernel size. accepts int or tuple of two ints.
        sigma_space: sigma in the coordinate space. accepts float or tuple of two floats.
        sigma_color: sigma in the color space. accepts float.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.ones([1, 5, 5])
        >>> layer = sk.image.BilateralBlur2D((3, 5), sigma_space=(1.2, 1.3), sigma_color=1.5)
        >>> print(layer(x))  # doctest: +SKIP
        [[[0.5231399  0.6869784  0.75100434 0.6869784  0.5231399 ]
          [0.70914114 0.9193193  1.         0.9193192  0.70914114]
          [0.70914114 0.9193193  1.         0.9193192  0.70914114]
          [0.70914114 0.9193193  1.         0.9193192  0.70914114]
          [0.5231399  0.6869784  0.75100434 0.6869784  0.5231399 ]]]
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        *,
        sigma_space: float | tuple[float, float],
        sigma_color: float,
    ):
        self.kernel_size = canonicalize(kernel_size, ndim=2, name="kernel_size")
        self.sigma_space = canonicalize(sigma_space, ndim=2, name="sigma_space")
        self.sigma_color = sigma_color

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        in_axes = (0, None, None, None)
        args = (self.kernel_size, self.sigma_space, self.sigma_color)
        return jax.vmap(bilateral_blur_2d, in_axes=in_axes)(image, *args)

    spatial_ndim: int = 2


class JointBilateralBlur2D(sk.TreeClass):
    """Apply joint bilateral blur to a channel-first image.

    .. image:: ../_static/jointbilateralblur2d.png

    Args:
        kernel_size: kernel size. accepts int or tuple of two ints.
        sigma_space: sigma in the coordinate space. accepts float or tuple of two floats.
        sigma_color: sigma in the color space. accepts float.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.ones([1, 5, 5])
        >>> guide = jnp.ones([1, 5, 5])
        >>> layer = sk.image.JointBilateralBlur2D((3, 5), sigma_space=(1.2, 1.3), sigma_color=1.5)
        >>> print(layer(x, guide))  # doctest: +SKIP
        [[[0.5231399  0.6869784  0.75100434 0.6869784  0.5231399 ]
          [0.70914114 0.9193193  1.         0.9193192  0.70914114]
          [0.70914114 0.9193193  1.         0.9193192  0.70914114]
          [0.70914114 0.9193193  1.         0.9193192  0.70914114]
          [0.5231399  0.6869784  0.75100434 0.6869784  0.5231399 ]]]
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        *,
        sigma_space: float | tuple[float, float],
        sigma_color: float,
    ):
        self.kernel_size = canonicalize(kernel_size, ndim=2, name="kernel_size")
        self.sigma_space = canonicalize(sigma_space, ndim=2, name="sigma_space")
        self.sigma_color = sigma_color

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray, guide: CHWArray) -> CHWArray:
        """Apply joint bilateral blur to a channel-first image.

        Args:
            image: input image.
            guide: guide image used for computing the gaussian for color space.
        """
        in_axes = (0, 0, None, None, None)
        args = (self.kernel_size, self.sigma_space, self.sigma_color)
        return jax.vmap(joint_bilateral_blur_2d, in_axes=in_axes)(image, guide, *args)

    spatial_ndim: int = 2


class BlurPool2DBase(sk.TreeClass):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        strides: int | tuple[int, int],
    ):
        self.kernel_size = canonicalize(kernel_size, ndim=2, name="kernel_size")
        self.strides = canonicalize(strides, ndim=2, name="strides")

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        in_axes = (0, None, None)
        args = (image, self.kernel_size, self.strides)
        return jax.vmap(type(self).filter_op, in_axes=in_axes)(*args)

    @property
    @abc.abstractmethod
    def filter_op(self):
        ...

    spatial_ndim: int = 2


class BlurPool2D(BlurPool2DBase):
    """Blur and downsample a channel-first image.

    .. image:: ../_static/blurpool2d.png

    Args:
        kernel_size: kernel size. accepts int or tuple of two ints.
        strides: strides. accepts int or tuple of two ints.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> layer = sk.image.BlurPool2D(kernel_size=3, strides=2)
        >>> print(layer(x))  # doctest: +SKIP
        [[[ 1.6875  3.5     3.5625]
          [ 8.5    13.     11.    ]
          [11.0625 16.     12.9375]]]
    """

    filter_op = blur_pool_2d


class FFTBlurPool2D(BlurPool2DBase):
    """Blur and downsample a channel-first image using FFT.

    .. image:: ../_static/blurpool2d.png

    Args:
        kernel_size: kernel size. accepts int or tuple of two ints.
        strides: stride. accepts int or tuple of two ints.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> layer = sk.image.FFTBlurPool2D(kernel_size=3, strides=2)
        >>> print(layer(x))  # doctest: +SKIP
        [[[ 1.6875  3.5     3.5625]
          [ 8.5    13.     11.    ]
          [11.0625 16.     12.9375]]]
    """

    filter_op = fft_blur_pool_2d
