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

import jax
import jax.random as jr
from jax import lax

import serket as sk
from serket.nn.utils import canonicalize, validate_spatial_ndim


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
        return lax.stop_gradient(jax.lax.dynamic_slice(x, *shape))

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        ...


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

    def __init__(self, size: int | tuple[int], start: int | tuple[int]):
        super().__init__(size, start)

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

    def __init__(self, size: int | tuple[int, int], start: int | tuple[int, int]):
        super().__init__(size, start)

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

    def __init__(
        self,
        size: int | tuple[int, int, int],
        start: int | tuple[int, int, int],
    ):
        super().__init__(size, start)

    @property
    def spatial_ndim(self) -> int:
        return 3


class RandomCropND(sk.TreeClass):
    def __init__(self, size: int | tuple[int, ...]):
        self.size = canonicalize(size, self.spatial_ndim, name="size")

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, *, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        start = tuple(
            jr.randint(
                key,
                shape=(),
                minval=0,
                maxval=x.shape[i] - s,
            )
            for i, s in enumerate(self.size)
        )
        return jax.lax.dynamic_slice(x, (0, *start), (x.shape[0], *self.size))

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

    def __init__(self, size: int | tuple[int]):
        super().__init__(size)

    @property
    def spatial_ndim(self) -> int:
        return 1


class RandomCrop2D(RandomCropND):
    """Applies jax.lax.dynamic_slice_in_dim with a random start along each axis

    Args:
        size: size of the slice in each axis. accepted values are either a single int
            or a tuple of two ints denoting the size along each axis.
    """

    def __init__(self, size: int | tuple[int, int]):
        super().__init__(size)

    @property
    def spatial_ndim(self) -> int:
        return 2


class RandomCrop3D(RandomCropND):
    """Applies jax.lax.dynamic_slice_in_dim with a random start along each axis

    Args:
        size: size of the slice in each axis. accepted values are either a single int
            or a tuple of three ints denoting the size along each axis.
    """

    def __init__(self, size: int | tuple[int, int, int]):
        super().__init__(size)

    @property
    def spatial_ndim(self) -> int:
        return 3
