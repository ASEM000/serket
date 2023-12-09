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
from typing import Literal

import jax
import jax.random as jr

import serket as sk
from serket._src.custom_transform import tree_eval
from serket._src.nn.linear import Identity
from serket._src.utils import (
    KernelSizeType,
    PaddingType,
    StridesType,
    canonicalize,
    delayed_canonicalize_padding,
    kernel_map,
    validate_spatial_ndim,
)

MethodKind = Literal["nearest", "linear", "cubic", "lanczos3", "lanczos5"]


def random_crop_nd(
    key: jax.Array,
    input: jax.Array,
    crop_size: tuple[int, ...],
) -> jax.Array:
    """Crops an input to the given size at a random starts along each axis.

    Args:
        key: random key.
        input: input array.
        crop_size: size of the crop along each axis.Accepts a tuple of int.
    """
    start: tuple[int, ...] = tuple(
        jr.randint(key, shape=(), minval=0, maxval=input.shape[i] - s)
        for i, s in enumerate(crop_size)
    )
    return jax.lax.dynamic_slice(input, start, crop_size)


def center_crop_nd(input: jax.Array, sizes: tuple[int, ...]) -> jax.Array:
    """Crops an input to the given size at the center.

    Args:
        input: input array.
        sizes: size of the crop along each axis.Accepts a tuple of int.
    """
    shapes = input.shape
    starts = tuple(max(shape // 2 - size // 2, 0) for shape, size in zip(shapes, sizes))
    return jax.lax.dynamic_slice(input, starts, sizes)


def extract_patches(
    input: jax.Array,
    kernel_size: KernelSizeType,
    strides: StridesType = 1,
    padding: PaddingType = "same",
):
    """Extract patches from an input array

    Args:
        input: input array
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        strides: stride of the convolution. Defaults to 1. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. Default ``same``. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

    Returns:
        A patches of the input array stacked along the first dimension.

    Example:

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> input = jnp.arange(15).reshape(5, 3)
        >>> print(input)
        [[ 0  1  2]
         [ 3  4  5]
         [ 6  7  8]
         [ 9 10 11]
         [12 13 14]]
        >>> kernel_size = 3
        >>> strides = 1
        >>> padding = "same"
        >>> patches = sk.nn.extract_patches(input, kernel_size, strides, padding)
        >>> print(patches.shape)
        (15, 3, 3)
        >>> print(patches(input)[0])
        [[0 0 0]
         [0 0 1]
         [0 3 4]]
        >>> print(patches(input)[1])
        [[0 0 0]
         [0 1 2]
         [3 4 5]]
    """
    # this function is performing faster than the `jax` version
    # on colab and m1 cpu but it does not support dilation
    kernel_size = canonicalize(kernel_size, input.ndim)
    strides = canonicalize(strides, input.ndim)
    padding = delayed_canonicalize_padding(
        in_dim=input.shape,
        padding=padding,
        kernel_size=kernel_size,
        strides=strides,
    )
    patch_func = kernel_map(
        # the function simply returns the view of the input
        func=lambda view: view,
        shape=input.shape,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
    )
    # stack along the first dimension
    return patch_func(input).reshape(-1, *kernel_size)


def upsample_nd(
    input: jax.Array,
    scale: int | tuple[int, ...],
    method: MethodKind = "nearest",
) -> jax.Array:
    """Upsample a 1D input to a given size using a given interpolation method.

    Args:
        input: input array.
        scale: the scale of the output. accetps a tuple of int denoting the scale
            multiplier along each axis.
        method: Interpolation method defaults to ``nearest``. choices are:

            - ``nearest``: Nearest neighbor interpolation. The values of antialias
              and precision are ignored.
            - ``linear``, ``bilinear``, ``trilinear``, ``triangle``: Linear interpolation.
              If ``antialias`` is True, uses a triangular filter when downsampling.
            - ``cubic``, ``bicubic``, ``tricubic``: Cubic interpolation, using
              the keys cubic kernel.
            - ``lanczos3``: Lanczos resampling, using a kernel of radius 3.
            - ``lanczos5``: Lanczos resampling, using a kernel of radius 5.
    """
    resized_shape = tuple(s * input.shape[i] for i, s in enumerate(scale))
    return jax.image.resize(input, resized_shape, method)


class UpsampleND(sk.TreeClass):
    def __init__(
        self,
        scale: int | tuple[int, ...] = 1,
        method: MethodKind = "nearest",
    ):
        # the difference between this and ResizeND is that UpsamplingND
        # use scale instead of size
        # assert types
        self.scale = canonicalize(scale, self.spatial_ndim, name="scale")
        self.method = method

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, input: jax.Array) -> jax.Array:
        in_axes = (0, None, None)
        args = (input, self.scale, self.method)
        return jax.vmap(upsample_nd, in_axes=in_axes)(*args)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        ...


class Upsample1D(UpsampleND):
    """Upsample a 1D input to a given size using a given interpolation method.

    Args:
        scale: the scale of the output.
        method: Interpolation method defaults to ``nearest``. choices are:

            - ``nearest``: Nearest neighbor interpolation. The values of antialias
              and precision are ignored.
            - ``linear``, ``bilinear``, ``trilinear``, ``triangle``: Linear interpolation.
              If ``antialias`` is True, uses a triangular filter when downsampling.
            - ``cubic``, ``bicubic``, ``tricubic``: Cubic interpolation, using
              the keys cubic kernel.
            - ``lanczos3``: Lanczos resampling, using a kernel of radius 3.
            - ``lanczos5``: Lanczos resampling, using a kernel of radius 5.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.nn.Upsample1D(scale=2)
        >>> input = jnp.arange(1, 6).reshape(1, 5)
        >>> print(layer(input))
        [[1 1 2 2 3 3 4 4 5 5]]
    """

    spatial_ndim: int = 1


class Upsample2D(UpsampleND):
    """Upsample a 2D input to a given size using a given interpolation method.

    Args:
        scale: the scale of the output. accetps a single int or a tuple of two
            int denoting the scale multiplier along each axis.
        method: Interpolation method defaults to ``nearest``. choices are:

            - ``nearest``: Nearest neighbor interpolation. The values of antialias
              and precision are ignored.
            - ``linear``, ``bilinear``, ``trilinear``, ``triangle``: Linear interpolation.
              If ``antialias`` is True, uses a triangular filter when downsampling.
            - ``cubic``, ``bicubic``, ``tricubic``: Cubic interpolation, using
              the keys cubic kernel.
            - ``lanczos3``: Lanczos resampling, using a kernel of radius 3.
            - ``lanczos5``: Lanczos resampling, using a kernel of radius 5.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.nn.Upsample2D(scale=(1, 2))
        >>> input = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(layer(input))
        [[[ 1  1  2  2  3  3  4  4  5  5]
          [ 6  6  7  7  8  8  9  9 10 10]
          [11 11 12 12 13 13 14 14 15 15]
          [16 16 17 17 18 18 19 19 20 20]
          [21 21 22 22 23 23 24 24 25 25]]]
    """

    spatial_ndim: int = 2


class Upsample3D(UpsampleND):
    """Upsample a 1D input to a given size using a given interpolation method.

    Args:
        scale: the scale of the output. accetps a single int or a tuple of three
            int denoting the scale multiplier along each axis.
        method: Interpolation method defaults to ``nearest``. choices are:

            - ``nearest``: Nearest neighbor interpolation. The values of antialias
              and precision are ignored.
            - ``linear``, ``bilinear``, ``trilinear``, ``triangle``: Linear interpolation.
              If ``antialias`` is True, uses a triangular filter when downsampling.
            - ``cubic``, ``bicubic``, ``tricubic``: Cubic interpolation, using
              the keys cubic kernel.
            - ``lanczos3``: Lanczos resampling, using a kernel of radius 3.
            - ``lanczos5``: Lanczos resampling, using a kernel of radius 5.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.nn.Upsample3D(scale=(1, 2, 1))
        >>> input = jnp.arange(1, 9).reshape(1, 2, 2, 2)
        >>> print(layer(input))
        [[[[1 2]
          [1 2]
          [3 4]
          [3 4]]
        <BLANKLINE>
         [[5 6]
          [5 6]
          [7 8]
          [7 8]]]]
    """

    spatial_ndim: int = 3


class RandomCropND(sk.TreeClass):
    def __init__(self, size: int | tuple[int, ...]):
        self.size = canonicalize(size, self.spatial_ndim, name="size")

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, input: jax.Array, *, key: jax.Array) -> jax.Array:
        crop_size = (input.shape[0], *self.size)
        return random_crop_nd(key, input, crop_size=crop_size)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        ...


class RandomCrop1D(RandomCropND):
    """Crop a 1D input to the given size at a random start.

    Args:
        size: size of the slice, either a single int or a tuple of int. accepted
            values are either a single int or a tuple of int denoting the size.
    """

    spatial_ndim: int = 1


class RandomCrop2D(RandomCropND):
    """Crop a 2D input to the given size at a random start.

    Args:
        size: size of the slice in each axis. accepted values are either a single int
            or a tuple of two ints denoting the size along each axis.
    """

    spatial_ndim: int = 2


class RandomCrop3D(RandomCropND):
    """Crop a 3D input to the given size at a random start.

    Args:
        size: size of the slice in each axis. accepted values are either a single int
            or a tuple of three ints denoting the size along each axis.
    """

    spatial_ndim: int = 3


class CenterCropND(sk.TreeClass):
    def __init__(self, size: int | tuple[int, ...]):
        self.size = canonicalize(size, self.spatial_ndim, name="size")

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.vmap(ft.partial(center_crop_nd, sizes=self.size))(input)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        ...


class CenterCrop1D(CenterCropND):
    """Crops a 1D input to the given size at the center.

    Args:
        size: The size of the output image. accepts a single int.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.nn.CenterCrop1D(4)
        >>> input = jnp.arange(1, 13).reshape(1, 12)
        >>> print(input)
        [[ 1  2  3  4  5  6  7  8  9 10 11 12]]
        >>> print(layer(input))
        [[5 6 7 8]]
    """

    spatial_ndim: int = 1


class CenterCrop2D(CenterCropND):
    """Crop the center of a channel-first image.

    .. image:: ../_static/centercrop2d.png

    Args:
        size: The size of the output image. accepts a single int or a tuple of two ints.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.nn.CenterCrop2D(4)
        >>> input = jnp.arange(1, 145).reshape(1, 12, 12)
        >>> print(input)
        [[[  1   2   3   4   5   6   7   8   9  10  11  12]
          [ 13  14  15  16  17  18  19  20  21  22  23  24]
          [ 25  26  27  28  29  30  31  32  33  34  35  36]
          [ 37  38  39  40  41  42  43  44  45  46  47  48]
          [ 49  50  51  52  53  54  55  56  57  58  59  60]
          [ 61  62  63  64  65  66  67  68  69  70  71  72]
          [ 73  74  75  76  77  78  79  80  81  82  83  84]
          [ 85  86  87  88  89  90  91  92  93  94  95  96]
          [ 97  98  99 100 101 102 103 104 105 106 107 108]
          [109 110 111 112 113 114 115 116 117 118 119 120]
          [121 122 123 124 125 126 127 128 129 130 131 132]
          [133 134 135 136 137 138 139 140 141 142 143 144]]]
        >>> print(layer(input))
        [[[53 54 55 56]
          [65 66 67 68]
          [77 78 79 80]
          [89 90 91 92]]]
    """

    spatial_ndim: int = 2


class CenterCrop3D(CenterCropND):
    """Crops a 3D input to the given size at the center."""

    spatial_ndim: int = 3


@tree_eval.def_eval(RandomCrop1D)
@tree_eval.def_eval(RandomCrop2D)
@tree_eval.def_eval(RandomCrop3D)
def _(_) -> Identity:
    return Identity()
