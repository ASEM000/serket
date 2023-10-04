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
import jax.numpy as jnp
import jax.random as jr

import serket as sk
from serket._src.custom_transform import tree_eval
from serket._src.nn.linear import Identity
from serket._src.utils import (
    IsInstance,
    canonicalize,
    delayed_canonicalize_padding,
    validate_spatial_nd,
)

MethodKind = Literal["nearest", "linear", "cubic", "lanczos3", "lanczos5"]


def random_crop_nd(
    key: jax.Array,
    x: jax.Array,
    *,
    crop_size: tuple[int, ...],
) -> jax.Array:
    start: tuple[int, ...] = tuple(
        jr.randint(key, shape=(), minval=0, maxval=x.shape[i] - s)
        for i, s in enumerate(crop_size)
    )
    return jax.lax.dynamic_slice(x, start, crop_size)


def center_crop_nd(array: jax.Array, sizes: tuple[int, ...]) -> jax.Array:
    """Crops an array to the given size at the center."""
    shapes = array.shape
    starts = tuple(max(shape // 2 - size // 2, 0) for shape, size in zip(shapes, sizes))
    return jax.lax.dynamic_slice(array, starts, sizes)


def zoom_in_along_axis(
    array: jax.Array,
    factor: float,
    axis: int,
    method: MethodKind = "linear",
) -> jax.Array:
    assert factor > 0
    shape = array.shape
    shape = list(shape)
    shape[axis] = int(shape[axis] * (1 + factor))
    return jax.image.resize(array, shape=shape, method=method)


def zoom_out_along_axis(
    array: jax.Array,
    factor: float,
    axis: int,
    method: MethodKind = "linear",
) -> jax.Array:
    assert factor < 0
    shape = array.shape
    shape = list(shape)
    shape[axis] = int(shape[axis] / (1 - factor))
    return jax.image.resize(array, shape=shape, method=method)


def zoom_nd(
    array: jax.Array,
    factor: tuple[int, ...],
    method: MethodKind = "linear",
) -> jax.Array:
    for axis, fi in enumerate(factor):
        if fi < 0:
            shape = array.shape
            array = zoom_out_along_axis(array, fi, axis, method=method)
            pad_width = [(0, 0)] * len(array.shape)
            left = (shape[axis] - array.shape[axis]) // 2
            right = shape[axis] - array.shape[axis] - left
            pad_width[axis] = (left, right)
            array = jnp.pad(array, pad_width=pad_width)
        elif fi > 0:
            shape = array.shape
            array = zoom_in_along_axis(array, fi, axis, method=method)
            array = center_crop_nd(array, shape)
    return array


def random_zoom_nd(
    key: jax.Array,
    array: jax.Array,
    factor: tuple[int, ...],
    method: MethodKind = "linear",
) -> jax.Array:
    for axis, (fi, ki) in enumerate(zip(factor, jr.split(key, len(factor)))):
        if fi < 0:
            shape = array.shape
            array = zoom_out_along_axis(array, fi, axis, method=method)
            pad_width = [(0, 0)] * len(array.shape)
            max_pad = shape[axis] - array.shape[axis]
            left = jr.randint(ki, shape=(), minval=0, maxval=max_pad)
            right = max_pad - left
            pad_width[axis] = (left, right)
            array = jnp.pad(array, pad_width=pad_width)
        elif fi > 0:
            shape = array.shape
            array = zoom_in_along_axis(array, fi, axis, method=method)
            array = random_crop_nd(ki, array, crop_size=shape)
    return array


def flatten(array: jax.Array, start_dim: int, end_dim: int):
    # wrapper around jax.lax.collapse
    # with inclusive end_dim and negative indexing support
    start_dim = start_dim + (0 if start_dim >= 0 else array.ndim)
    end_dim = end_dim + 1 + (0 if end_dim >= 0 else array.ndim)
    return jax.lax.collapse(array, start_dim, end_dim)


def unflatten(array: jax.Array, dim: int, shape: tuple[int, ...]):
    in_shape = list(array.shape)
    out_shape = [*in_shape[:dim], *shape, *in_shape[dim + 1 :]]
    return jnp.reshape(array, out_shape)


class ResizeND(sk.TreeClass):
    def __init__(
        self,
        size: int | tuple[int, ...],
        method: MethodKind = "nearest",
        antialias: bool = True,
    ):
        self.size = canonicalize(size, self.spatial_ndim, name="size")
        self.method = method
        self.antialias = antialias

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        in_axes = (0, None, None, None)
        args = (x, self.size, self.method, self.antialias)
        return jax.vmap(jax.image.resize, in_axes=in_axes)(*args)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        ...


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

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        resized_shape = tuple(s * x.shape[i + 1] for i, s in enumerate(self.scale))
        in_axes = (0, None, None)
        args = (x, resized_shape, self.method)
        return jax.vmap(jax.image.resize, in_axes=in_axes)(*args)

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
        >>> x = jnp.arange(1, 6).reshape(1, 5)
        >>> print(sk.nn.Upsample1D(scale=2)(x))
        [[1 1 2 2 3 3 4 4 5 5]]
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


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
        >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
        >>> print(sk.nn.Upsample2D(scale=(1, 2))(x))
        [[[ 1  1  2  2  3  3  4  4  5  5]
          [ 6  6  7  7  8  8  9  9 10 10]
          [11 11 12 12 13 13 14 14 15 15]
          [16 16 17 17 18 18 19 19 20 20]
          [21 21 22 22 23 23 24 24 25 25]]]
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


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
        >>> x = jnp.arange(1, 9).reshape(1, 2, 2, 2)
        >>> print(sk.nn.Upsample3D(scale=(1, 2, 1))(x))
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

    @property
    def spatial_ndim(self) -> int:
        return 3


class CropND(sk.TreeClass):
    def __init__(self, size: int | tuple[int, ...], start: int | tuple[int, ...]):
        self.size = canonicalize(size, self.spatial_ndim, name="size")
        self.start = canonicalize(start, self.spatial_ndim, name="start")

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        in_axes = (0, None, None)
        args = (x, self.start, self.size)
        return jax.vmap(jax.lax.dynamic_slice, in_axes=in_axes)(*args)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class Crop1D(CropND):
    """Applies ``jax.lax.dynamic_slice_in_dim`` to the second dimension of the input.

    Args:
        size: size of the slice, either a single int or a tuple of int.
        start: start of the slice, either a single int or a tuple of int.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> x = jnp.arange(1, 6).reshape(1, 5)
        >>> print(sk.nn.Crop1D(size=3, start=1)(x))
        [[2 3 4]]
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class Crop2D(CropND):
    """Applies ``jax.lax.dynamic_slice_in_dim`` to the second dimension of the input.

    Args:
        size: size of the slice, either a single int or a tuple of two ints
            for size along each axis.
        start: start of the slice, either a single int or a tuple of two ints
            for start along each axis.

    Example:
        >>> # start = (2, 0) and size = (3, 3)
        >>> # i.e. start at index 2 along the first axis and index 0 along the second axis
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> x = jnp.arange(1, 26).reshape((1, 5, 5))
        >>> print(x)
        [[[ 1  2  3  4  5]
          [ 6  7  8  9 10]
          [11 12 13 14 15]
          [16 17 18 19 20]
          [21 22 23 24 25]]]
        >>> print(sk.nn.Crop2D(size=3, start=(2, 0))(x))
        [[[11 12 13]
          [16 17 18]
          [21 22 23]]]
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class Crop3D(CropND):
    """Applies ``jax.lax.dynamic_slice_in_dim`` to the second dimension of the input.

    Args:
        size: size of the slice, either a single int or a tuple of three ints
            for size along each axis.
        start: start of the slice, either a single int or a tuple of three
            ints for start along each axis.
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class PadND(sk.TreeClass):
    def __init__(self, padding: int | tuple[int, int], value: float = 0.0):
        kernel_size = ((1,),) * self.spatial_ndim
        self.padding = delayed_canonicalize_padding(None, padding, kernel_size, None)
        self.value = value

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        value = jax.lax.stop_gradient(self.value)
        pad = ft.partial(jnp.pad, pad_width=self.padding, constant_values=value)
        return jax.vmap(pad)(x)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        ...


class Pad1D(PadND):
    """Pad a 1D tensor.

    Args:
        padding: padding to apply to each side of the input. accepts a single int
            or a tuple of tuple of two ints for padding along each axis. e.g.
            ``((1, 2),)`` for padding of 1 on the left and 2 on the right.
        value: value to pad with. Defaults to 0.0.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 6).reshape(1, 5)
        >>> print(sk.nn.Pad1D(((1, 2),))(x))
        [[0 1 2 3 4 5 0 0]]

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding1D
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class Pad2D(PadND):
    """Pad a 2D tensor.

    Args:
        padding: padding to apply to each side of the input. accepts a single int
            or a tuple of two tuples of two ints for padding along each axis. e.g.
            ``((1, 2), (3, 4))`` for padding of 1 on the left and 2 on the right
            along the and 3 on the top and 4 on the bottom.
        value: value to pad with. Defaults to 0.0.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 10).reshape(1, 3, 3)
        >>> print(sk.nn.Pad2D(((1, 2), (3, 4)))(x))
        [[[0 0 0 0 0 0 0 0 0 0]
          [0 0 0 1 2 3 0 0 0 0]
          [0 0 0 4 5 6 0 0 0 0]
          [0 0 0 7 8 9 0 0 0 0]
          [0 0 0 0 0 0 0 0 0 0]
          [0 0 0 0 0 0 0 0 0 0]]]

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class Pad3D(PadND):
    """Pad a 3D tensor.

    Args:
        padding: padding to apply to each side of the input. accepts a single int
            or a tuple of three tuples of two ints for padding along each axis.
        value: value to pad with. Defaults to 0.0.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 9).reshape(1, 2, 2, 2)
        >>> print(sk.nn.Pad3D(((0, 0), (2, 0), (2, 0)))(x))
        [[[[0 0 0 0]
          [0 0 0 0]
          [0 0 1 2]
          [0 0 3 4]]
        <BLANKLINE>
         [[0 0 0 0]
         [0 0 0 0]
         [0 0 5 6]
         [0 0 7 8]]]]

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding3D
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class Resize1D(ResizeND):
    """Resize 1D to a given size using a given interpolation method.

    Args:
        size: the size of the output. if size is None, the output size is
            calculated as input size * scale
        method: Interpolation method defaults to ``nearest``. choices are:

            - ``nearest``: Nearest neighbor interpolation. The values of antialias
              and precision are ignored.
            - ``linear``, ``bilinear``, ``trilinear``, ``triangle``: Linear interpolation.
              If ``antialias`` is True, uses a triangular filter when downsampling.
            - ``cubic``, ``bicubic``, ``tricubic``: Cubic interpolation, using
              the keys cubic kernel.
            - ``lanczos3``: Lanczos resampling, using a kernel of radius 3.
            - ``lanczos5``: Lanczos resampling, using a kernel of radius 5.

        antialias: whether to use antialiasing. Defaults to True.
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class Resize2D(ResizeND):
    """Resize 2D input to a given size using a given interpolation method.

    Args:
        size: the size of the output. if size is None, the output size is
            calculated as input size * scale
        method: Interpolation method defaults to ``nearest``. choices are:

            - ``nearest``: Nearest neighbor interpolation. The values of antialias
              and precision are ignored.
            - ``linear``, ``bilinear``, ``trilinear``, ``triangle``: Linear interpolation.
              If ``antialias`` is True, uses a triangular filter when downsampling.
            - ``cubic``, ``bicubic``, ``tricubic``: Cubic interpolation, using
              the keys cubic kernel.
            - ``lanczos3``: Lanczos resampling, using a kernel of radius 3.
            - ``lanczos5``: Lanczos resampling, using a kernel of radius 5.

        antialias: whether to use antialiasing. Defaults to True.
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class Resize3D(ResizeND):
    """Resize 3D input to a given size using a given interpolation method.

    Args:
        size: the size of the output. if size is None, the output size is
            calculated as input size * scale
        method: Interpolation method defaults to ``nearest``. choices are:

            - ``nearest``: Nearest neighbor interpolation. The values of antialias
              and precision are ignored.
            - ``linear``, ``bilinear``, ``trilinear``, ``triangle``: Linear interpolation.
              If ``antialias`` is True, uses a triangular filter when downsampling.
            - ``cubic``, ``bicubic``, ``tricubic``: Cubic interpolation, using
              the keys cubic kernel.
            - ``lanczos3``: Lanczos resampling, using a kernel of radius 3.
            - ``lanczos5``: Lanczos resampling, using a kernel of radius 5.

        antialias: whether to use antialiasing. Defaults to True.
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


@sk.autoinit
class Flatten(sk.TreeClass):
    """Flatten an array from dim ``start_dim`` to ``end_dim`` (inclusive).

    Args:
        start_dim: the first dimension to flatten
        end_dim: the last dimension to flatten (inclusive)

    Returns:
        a function that flattens a ``jax.Array``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> sk.nn.Flatten(0,1)(jnp.ones([1,2,3,4,5])).shape
        (2, 3, 4, 5)
        >>> sk.nn.Flatten(0,2)(jnp.ones([1,2,3,4,5])).shape
        (6, 4, 5)
        >>> sk.nn.Flatten(1,2)(jnp.ones([1,2,3,4,5])).shape
        (1, 6, 4, 5)
        >>> sk.nn.Flatten(-1,-1)(jnp.ones([1,2,3,4,5])).shape
        (1, 2, 3, 4, 5)
        >>> sk.nn.Flatten(-2,-1)(jnp.ones([1,2,3,4,5])).shape
        (1, 2, 3, 20)
        >>> sk.nn.Flatten(-3,-1)(jnp.ones([1,2,3,4,5])).shape
        (1, 2, 60)
    """

    start_dim: int = sk.field(default=0, on_setattr=[IsInstance(int)])
    end_dim: int = sk.field(default=-1, on_setattr=[IsInstance(int)])

    def __call__(self, x: jax.Array) -> jax.Array:
        return flatten(x, self.start_dim, self.end_dim)


@sk.autoinit
class Unflatten(sk.TreeClass):
    """Unflatten an array.

    Args:
        dim: the dimension to unflatten.
        shape: the shape to unflatten to. accepts a tuple of ints.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> sk.nn.Unflatten(0, (1,2,3,4,5))(jnp.ones([120])).shape
        (1, 2, 3, 4, 5)
        >>> sk.nn.Unflatten(2,(2,3))(jnp.ones([1,2,6])).shape
        (1, 2, 2, 3)
    """

    dim: int = sk.field(default=0, on_setattr=[IsInstance(int)])
    shape: tuple = sk.field(default=None, on_setattr=[IsInstance(tuple)])

    def __call__(self, x: jax.Array) -> jax.Array:
        return unflatten(x, self.dim, self.shape)


class RandomCropND(sk.TreeClass):
    def __init__(self, size: int | tuple[int, ...]):
        self.size = canonicalize(size, self.spatial_ndim, name="size")

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, *, key: jax.Array) -> jax.Array:
        crop_size = [x.shape[0], *self.size]
        return random_crop_nd(key, x, crop_size=crop_size)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        ...


class RandomCrop1D(RandomCropND):
    """Applies ``jax.lax.dynamic_slice_in_dim`` with a random start along each axis

    Args:
        size: size of the slice, either a single int or a tuple of int. accepted
            values are either a single int or a tuple of int denoting the size.
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class RandomCrop2D(RandomCropND):
    """Applies ``jax.lax.dynamic_slice_in_dim`` with a random start along each axis

    Args:
        size: size of the slice in each axis. accepted values are either a single int
            or a tuple of two ints denoting the size along each axis.
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class RandomCrop3D(RandomCropND):
    """Applies ``jax.lax.dynamic_slice_in_dim`` with a random start along each axis

    Args:
        size: size of the slice in each axis. accepted values are either a single int
            or a tuple of three ints denoting the size along each axis.
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class ZoomND(sk.TreeClass):
    def __init__(self, factor: float | tuple[float, ...]):
        self.factor = canonicalize(factor, self.spatial_ndim, name="factor")

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        factor = jax.lax.stop_gradient(self.factor)
        return jax.vmap(zoom_nd, in_axes=(0, None))(x, factor)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class Zoom1D(ZoomND):
    """Zoom a 1D spatial tensor.

    Zooming in is equivalent to resizing the tensor to a larger size followed
    by center cropping. Zooming out is equivalent to resizing the tensor to a
    smaller size followed by equal padding on both sides. Zooming in is defined
    by positive values of ``factor`` and zooming out is defined by negative
    values of ``factor``.

    Args:
        factor: zoom factor. accepts a single float or a tuple of float denoting
            the zoom factor along each axis. if positive, zoom in, if negative,
            zoom out, if 0, no zoom.
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class Zoom2D(ZoomND):
    """Zoom a 2D spatial tensor.

    .. image:: ../_static/zoom2d.png

    Zooming in is equivalent to resizing the tensor to a larger size followed
    by center cropping. Zooming out is equivalent to resizing the tensor to a
    smaller size followed by equal padding on both sides. Zooming in is defined
    by positive values of ``factor`` and zooming out is defined by negative
    values of ``factor``.


    Args:
        factor: zoom factor. accepts a single float or a tuple of float denoting
            the zoom factor along each axis. if positive, zoom in, if negative,
            zoom out, if 0, no zoom.
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class Zoom3D(ZoomND):
    """Zoom a 3D spatial tensor.

    Zooming in is equivalent to resizing the tensor to a larger size followed
    by center cropping. Zooming out is equivalent to resizing the tensor to a
    smaller size followed by equal padding on both sides. Zooming in is defined
    by positive values of ``factor`` and zooming out is defined by negative
    values of ``factor``.


    Args:
        factor: zoom factor. accepts a single float or a tuple of float denoting
            the zoom factor along each axis. if positive, zoom in, if negative,
            zoom out, if 0, no zoom.
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class RandomZoom1D(sk.TreeClass):
    """Randomly zoom a 1D spatial tensor.

    Random zooming in is equivalent to resizing the tensor to a larger size
    followed by random cropping. Zooming out is equivalent to resizing the tensor
    to a smaller size followed by random padding. Zooming in is defined
    by positive values of ``factor`` and zooming out is defined by negative
    values of ``factor``.

    Args:
        length_range: a tuple of two floats denoting the range of the zoom factor.
    """

    def __init__(self, length_range: tuple[int, int] = (0.0, 1.0)):
        if not (isinstance(length_range, tuple) and len(length_range) == 2):
            raise ValueError("`length_range` must be a tuple of length 2")

        self.length_range = length_range

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, *, key: jax.Array) -> jax.Array:
        k1, k2 = jr.split(key, 2)
        low, high = jax.lax.stop_gradient(self.length_range)
        factor = (jr.uniform(k1, minval=low, maxval=high),)
        return jax.vmap(random_zoom_nd, in_axes=(None, 0, None))(k2, x, factor)

    @property
    def spatial_ndim(self) -> int:
        return 1


class RandomZoom2D(sk.TreeClass):
    """Randomly zoom a 2D spatial tensor.

    .. image:: ../_static/zoom2d.png

    Random zooming in is equivalent to resizing the tensor to a larger size
    followed by random cropping. Zooming out is equivalent to resizing the tensor
    to a smaller size followed by random padding. Zooming in is defined
    by positive values of ``factor`` and zooming out is defined by negative
    values of ``factor``.

    Args:
        height_range: a tuple of two floats denoting the range of the zoom factor.
        width_range: a tuple of two floats denoting the range of the zoom factor.
    """

    def __init__(
        self,
        height_range: tuple[float, float] = (0.0, 1.0),
        width_range: tuple[float, float] = (0.0, 1.0),
    ):
        if not (isinstance(height_range, tuple) and len(height_range) == 2):
            raise ValueError("`height_range` must be a tuple of length 2")

        if not (isinstance(width_range, tuple) and len(width_range) == 2):
            raise ValueError("`width_range` must be a tuple of length 2")

        self.height_range = height_range
        self.width_range = width_range

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, *, key: jax.Array) -> jax.Array:
        k1, k2, k3 = jr.split(key, 3)
        factors = (self.height_range, self.width_range)
        ((hfl, hfh), (wfl, wfh)) = jax.lax.stop_gradient(factors)
        factor_r = jr.uniform(k1, minval=hfl, maxval=hfh)
        factor_c = jr.uniform(k2, minval=wfl, maxval=wfh)
        factor = (factor_r, factor_c)
        return jax.vmap(random_zoom_nd, in_axes=(None, 0, None))(k3, x, factor)

    @property
    def spatial_ndim(self) -> int:
        return 2


class RandomZoom3D(sk.TreeClass):
    """Randomly zoom a 3D spatial tensor.

    Random zooming in is equivalent to resizing the tensor to a larger size
    followed by random cropping. Zooming out is equivalent to resizing the tensor
    to a smaller size followed by random padding. Zooming in is defined
    by positive values of ``factor`` and zooming out is defined by negative
    values of ``factor``.

    Args:
        height_range: a tuple of two floats denoting the range of the zoom factor.
        width_range: a tuple of two floats denoting the range of the zoom factor.
        depth_range: a tuple of two floats denoting the range of the zoom factor.
    """

    def __init__(
        self,
        height_range: tuple[float, float] = (0.0, 1.0),
        width_range: tuple[float, float] = (0.0, 1.0),
        depth_range: tuple[float, float] = (0.0, 1.0),
    ):
        if not (isinstance(height_range, tuple) and len(height_range) == 2):
            raise ValueError("`height_range` must be a tuple of length 2")

        if not (isinstance(width_range, tuple) and len(width_range) == 2):
            raise ValueError("`width_range` must be a tuple of length 2")

        if not (isinstance(depth_range, tuple) and len(depth_range) == 2):
            raise ValueError("`depth_range` must be a tuple of length 2")

        self.height_range = height_range
        self.width_range = width_range
        self.depth_range = depth_range

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, *, key: jax.Array) -> jax.Array:
        k1, k2, k3, k4 = jr.split(key, 4)
        factors = (self.height_range, self.width_range, self.depth_range)
        ((hfl, hfh), (wfl, wfh), (dfl, dfh)) = jax.lax.stop_gradient(factors)
        factor_r = jr.uniform(k1, minval=hfl, maxval=hfh)
        factor_c = jr.uniform(k2, minval=wfl, maxval=wfh)
        factor_d = jr.uniform(k3, minval=dfl, maxval=dfh)
        factor = (factor_r, factor_c, factor_d)
        return jax.vmap(random_zoom_nd, in_axes=(None, 0, None))(k4, x, factor)

    @property
    def spatial_ndim(self) -> int:
        return 3


class CenterCropND(sk.TreeClass):
    def __init__(self, size: int | tuple[int, ...]):
        self.size = canonicalize(size, self.spatial_ndim, name="size")

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.vmap(ft.partial(center_crop_nd, sizes=self.size))(x)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        ...


class CenterCrop1D(CenterCropND):
    """Crops a 1D array to the given size at the center.

    Args:
        size: The size of the output image. accepts a single int.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 13).reshape(1, 12)
        >>> print(x)
        [[ 1  2  3  4  5  6  7  8  9 10 11 12]]
        >>> print(sk.nn.CenterCrop1D(4)(x))
        [[5 6 7 8]]
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class CenterCrop2D(CenterCropND):
    """Crop the center of a channel-first image.

    .. image:: ../_static/centercrop2d.png

    Args:
        size: The size of the output image. accepts a single int or a tuple of two ints.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(1, 145).reshape(1, 12, 12)
        >>> print(x)
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
        >>> print(sk.nn.CenterCrop2D(4)(x))
        [[[53 54 55 56]
          [65 66 67 68]
          [77 78 79 80]
          [89 90 91 92]]]
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class CenterCrop3D(CenterCropND):
    """Crops a 3D array to the given size at the center."""

    @property
    def spatial_ndim(self) -> int:
        return 3


@tree_eval.def_eval(RandomCrop1D)
@tree_eval.def_eval(RandomCrop2D)
@tree_eval.def_eval(RandomCrop3D)
@tree_eval.def_eval(RandomZoom1D)
@tree_eval.def_eval(RandomZoom2D)
@tree_eval.def_eval(RandomZoom3D)
def _(_) -> Identity:
    return Identity()
