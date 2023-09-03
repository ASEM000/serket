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

import abc
import functools as ft
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr

import serket as sk
from serket.custom_transform import tree_eval
from serket.nn.linear import Identity
from serket.utils import (
    IsInstance,
    canonicalize,
    delayed_canonicalize_padding,
    validate_spatial_ndim,
)

MethodKind = Literal["nearest", "linear", "cubic", "lanczos3", "lanczos5"]


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

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.image.resize(
            x,
            shape=(x.shape[0], *self.size),
            method=self.method,
            antialias=self.antialias,
        )

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

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        resized_shape = tuple(s * x.shape[i + 1] for i, s in enumerate(self.scale))
        return jax.image.resize(
            x,
            shape=(x.shape[0], *resized_shape),
            method=self.method,
        )

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        ...


class CropND(sk.TreeClass):
    """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

    Args:
        size: size of the slice, accepted values are integers or tuples of integers.
        start: start of the slice, accepted values are integers or tuples of integers.
    """

    def __init__(self, size: int | tuple[int, ...], start: int | tuple[int, ...]):
        self.size = canonicalize(size, self.spatial_ndim, name="size")
        self.start = canonicalize(start, self.spatial_ndim, name="start")

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        shape = ((0, *self.start), (x.shape[0], *self.size))
        return jax.lax.stop_gradient(jax.lax.dynamic_slice(x, *shape))

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class PadND(sk.TreeClass):
    def __init__(self, padding: int | tuple[int, int], value: float = 0.0):
        self.padding = delayed_canonicalize_padding(
            in_dim=None,
            padding=padding,
            kernel_size=((1,),) * self.spatial_ndim,
            strides=None,
        )
        self.value = value

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        # do not pad the channel axis
        shape = ((0, 0), *self.padding)
        value = jax.lax.stop_gradient(self.value)
        return jnp.pad(x, shape, constant_values=value)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        ...


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


class Upsample1D(UpsampleND):
    """Upsample a 1D input to a given size using a given interpolation method.

    Args:
        scale: the scale of the output.
        method: the method of interpolation. Defaults to "nearest".
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class Upsample2D(UpsampleND):
    """Upsample a 2D input to a given size using a given interpolation method.

    Args:
        scale: the scale of the output.
        method: the method of interpolation. Defaults to "nearest".
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class Upsample3D(UpsampleND):
    """Upsample a 1D input to a given size using a given interpolation method.

    Args:
        scale: the scale of the output.
        method: the method of interpolation. Defaults to "nearest".
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

    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html?highlight=flatten#torch.nn.Flatten
    """

    start_dim: int = sk.field(default=0, callbacks=[IsInstance(int)])
    end_dim: int = sk.field(default=-1, callbacks=[IsInstance(int)])

    def __call__(self, x: jax.Array) -> jax.Array:
        start_dim = self.start_dim + (0 if self.start_dim >= 0 else x.ndim)
        end_dim = self.end_dim + 1 + (0 if self.end_dim >= 0 else x.ndim)
        return jax.lax.collapse(x, start_dim, end_dim)


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

    Reference:
        - https://pytorch.org/docs/stable/generated/torch.nn.Unflatten.html?highlight=unflatten
    """

    dim: int = sk.field(default=0, callbacks=[IsInstance(int)])
    shape: tuple = sk.field(default=None, callbacks=[IsInstance(tuple)])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        shape = list(x.shape)
        shape = [*shape[: self.dim], *self.shape, *shape[self.dim + 1 :]]
        return jnp.reshape(x, shape)


class Pad1D(PadND):
    """Pad a 1D tensor.

    Args:
        padding: padding to apply to each side of the input.
        value: value to pad with. Defaults to 0.0.

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
        padding: padding to apply to each side of the input.
        value: value to pad with. Defaults to 0.0.

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
        padding: padding to apply to each side of the input.
        value: value to pad with. Defaults to 0.0.

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding3D
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class Crop1D(CropND):
    """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

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
    """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

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
    """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

    Args:
        size: size of the slice, either a single int or a tuple of three ints
            for size along each axis.
        start: start of the slice, either a single int or a tuple of three
            ints for start along each axis.
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


def random_crop_nd(
    x: jax.Array,
    *,
    crop_size: tuple[int, ...],
    key: jr.KeyArray,
) -> jax.Array:
    start: tuple[int, ...] = tuple(
        jr.randint(key, shape=(), minval=0, maxval=x.shape[i] - s)
        for i, s in enumerate(crop_size)
    )
    return jax.lax.dynamic_slice(x, start, crop_size)


class RandomCropND(sk.TreeClass):
    def __init__(self, size: int | tuple[int, ...]):
        self.size = canonicalize(size, self.spatial_ndim, name="size")

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, *, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        crop_size = [x.shape[0], *self.size]
        return random_crop_nd(x, crop_size=crop_size, key=key)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        ...


class RandomCrop1D(RandomCropND):
    """Applies jax.lax.dynamic_slice_in_dim with a random start along each axis

    Args:
        size: size of the slice, either a single int or a tuple of int. accepted
            values are either a single int or a tuple of int denoting the size.
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class RandomCrop2D(RandomCropND):
    """Applies jax.lax.dynamic_slice_in_dim with a random start along each axis

    Args:
        size: size of the slice in each axis. accepted values are either a single int
            or a tuple of two ints denoting the size along each axis.
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class RandomCrop3D(RandomCropND):
    """Applies jax.lax.dynamic_slice_in_dim with a random start along each axis

    Args:
        size: size of the slice in each axis. accepted values are either a single int
            or a tuple of three ints denoting the size along each axis.
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


def zoom_axis(
    x: jax.Array,
    factor: float,
    key: jr.KeyArray,
    axis: int,
) -> jax.Array:
    if factor == 0:
        return x

    axis_size = x.shape[axis]
    dtype = x.dtype
    resized_axis_size = int(axis_size * (1 + factor))

    def zoom_in(x):
        shape = list(x.shape)
        resized_shape = list(shape)
        resized_shape[axis] = resized_axis_size
        x = jax.image.resize(x, shape=resized_shape, method="linear")
        x = random_crop_nd(x, crop_size=shape, key=key)
        return x.astype(dtype)

    def zoom_out(x):
        shape = list(x.shape)
        resized_shape = list(shape)
        resized_shape[axis] = resized_axis_size
        x = jax.image.resize(x, shape=resized_shape, method="linear")
        pad_width = [(0, 0)] * len(x.shape)
        left = (axis_size - resized_axis_size) // 2
        right = axis_size - resized_axis_size - left
        pad_width[axis] = (left, right)
        x = jnp.pad(x, pad_width=pad_width)
        return x.astype(dtype)

    return zoom_out(x) if factor < 0 else zoom_in(x)


class RandomZoom1D(sk.TreeClass):
    def __init__(self, length_factor: tuple[int, int] = (0.0, 1.0)):
        """Randomly zooms a 1D spatial tensor.

        Positive values are zoom in, negative values are zoom out, and 0 is no zoom.

        Args:
            length_factor: (min, max)

        Example:
            >>> import serket as sk
            >>> import jax.numpy as jnp
            >>> import jax
            >>> x = jnp.arange(1, 10).reshape(1, -1)
            >>> # 0% zoom (unchanged)
            >>> print(sk.nn.RandomZoom1D((0.0, 0.0))(x, key=jax.random.PRNGKey(1)))
            [[1 2 3 4 5 6 7 8 9]]
            >>> # 50%-100% probability of zoom
            >>> length_factor = (0.5, 1.0)
            >>> key = jax.random.PRNGKey(0)
            >>> print(sk.nn.RandomZoom1D(length_factor=length_factor)(x, key=key))
            [[4 4 5 5 6 6 7 8 8]]

        Reference:
            - https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomZoom
        """
        if not (isinstance(length_factor, tuple) and len(length_factor) == 2):
            raise ValueError("`length_factor` must be a tuple of length 2")

        self.length_factor = length_factor

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        k1, k2 = jr.split(key, 2)
        low, high = self.length_factor
        x = zoom_axis(x, jr.uniform(k1, minval=low, maxval=high), k2, axis=1)
        return jax.lax.stop_gradient(x)

    @property
    def spatial_ndim(self) -> int:
        return 1


class RandomZoom2D(sk.TreeClass):
    def __init__(
        self,
        height_factor: tuple[float, float] = (0.0, 1.0),
        width_factor: tuple[float, float] = (0.0, 1.0),
    ):
        """Randomly zooms a features-first 2D spatial tensor.

        Positive values are zoom in, negative values are zoom out, and 0 is no zoom.

        Args:
            height_factor: (min, max)
            width_factor: (min, max)

        Example:
            >>> import serket as sk
            >>> import jax.numpy as jnp
            >>> import jax
            >>> x = jnp.arange(1, 26).reshape(1, 5, 5)
            >>> # 0% zoom (unchanged)
            >>> height_factor = (0.0, 0.0)
            >>> width_factor = (0.0, 0.0)
            >>> key = jax.random.PRNGKey(1)
            >>> print(sk.nn.RandomZoom2D(height_factor=height_factor, width_factor=width_factor)(x, key=key))
            [[1 2 3 4 5 6 7 8 9]]
            >>> # 50%-100% probability of zoom
            >>> height_factor = (0.5, 1.0)
            >>> width_factor = (0.5, 1.0)
            >>> key = jax.random.PRNGKey(0)
            >>> print(sk.nn.RandomZoom2D(height_factor=height_factor, width_factor=width_factor)(x, key=key))
            [[[ 1  2  3  3  4]
            [ 2  3  4  4  5]
            [ 5  6  7  7  8]
            [ 8  9 10 10 11]
            [11 12 13 13 14]]]

        Reference:
            - https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomZoom
        """
        if not (isinstance(height_factor, tuple) and len(height_factor) == 2):
            raise ValueError("`height_factor` must be a tuple of length 2")

        if not (isinstance(width_factor, tuple) and len(width_factor) == 2):
            raise ValueError("`width_factor` must be a tuple of length 2")

        self.height_factor = height_factor
        self.width_factor = width_factor

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        k1, k2, k3, k4 = jr.split(key, 4)
        low, high = self.height_factor
        x = zoom_axis(x, jr.uniform(k1, minval=low, maxval=high), k3, axis=1)
        low, high == self.width_factor
        x = zoom_axis(x, jr.uniform(k2, minval=low, maxval=high), k4, axis=2)
        return jax.lax.stop_gradient(x)

    @property
    def spatial_ndim(self) -> int:
        return 2


class RandomZoom3D(sk.TreeClass):
    def __init__(
        self,
        height_factor: tuple[float, float] = (0.0, 1.0),
        width_factor: tuple[float, float] = (0.0, 1.0),
        depth_factor: tuple[float, float] = (0.0, 1.0),
    ):
        """Randomly zooms a features-first 3D spatial tensor.

        Positive values are zoom in, negative values are zoom out, and 0 is no zoom.

        Args:
            height_factor: (min, max)
            width_factor: (min, max)
            depth_factor: (min, max)

        Reference:
            - https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomZoom
        """
        if not (isinstance(height_factor, tuple) and len(height_factor) == 2):
            raise ValueError("`height_factor` must be a tuple of length 2")

        if not (isinstance(width_factor, tuple) and len(width_factor) == 2):
            raise ValueError("`width_factor` must be a tuple of length 2")

        if not (isinstance(depth_factor, tuple) and len(depth_factor) == 2):
            raise ValueError("`depth_factor` must be a tuple of length 2")

        self.height_factor = height_factor
        self.width_factor = width_factor
        self.depth_factor = depth_factor

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        k1, k2, k3, k4, k5, k6 = jr.split(key, 6)
        low, high = self.height_factor
        x = zoom_axis(x, jr.uniform(k1, minval=low, maxval=high), k3, axis=1)
        low, high == self.width_factor
        x = zoom_axis(x, jr.uniform(k2, minval=low, maxval=high), k4, axis=2)
        low, high == self.depth_factor
        x = zoom_axis(x, jr.uniform(k5, minval=low, maxval=high), k6, axis=3)
        return jax.lax.stop_gradient(x)

    @property
    def spatial_ndim(self) -> int:
        return 3


@tree_eval.def_eval(RandomCrop1D)
@tree_eval.def_eval(RandomCrop2D)
@tree_eval.def_eval(RandomCrop3D)
@tree_eval.def_eval(RandomZoom1D)
@tree_eval.def_eval(RandomZoom2D)
@tree_eval.def_eval(RandomZoom3D)
def random_transform_eval(_) -> Identity:
    return Identity()
