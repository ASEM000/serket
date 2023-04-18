from __future__ import annotations

import functools as ft

import jax
import jax.random as jr
import pytreeclass as pytc

from serket.nn.callbacks import validate_spatial_in_shape
from serket.nn.utils import canonicalize


class CropND(pytc.TreeClass):
    size: int | tuple[int, ...]
    start: int | tuple[int, ...]

    def __init__(self, size, start, spatial_ndim):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            size: size of the slice
            start: start of the slice
        """
        self.size = canonicalize(size, spatial_ndim, "size")
        self.start = canonicalize(start, spatial_ndim, "start")
        self.spatial_ndim = spatial_ndim

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.lax.dynamic_slice(x, (0, *self.start), (x.shape[0], *self.size))


class Crop1D(CropND):
    def __init__(self, size: int | tuple[int], start: int | tuple[int, int]):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Example:
            >>> import jax
            >>> import jax.numpy as jnp
            >>> import serket as sk
            >>> x = jnp.arange(1, 6).reshape(1, 5)
            >>> print(sk.nn.Crop1D(size=3, start=1)(x))
            [[2 3 4]]
        """
        super().__init__(size, start, spatial_ndim=1)


class Crop2D(CropND):
    def __init__(self, size: int | tuple[int, int], start: int | tuple[int, int]):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            size: size of the slice, either a single int or a tuple of two ints
            start: start of the slice, either a single int or a tuple of two ints for start along each axis

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
        super().__init__(size, start, spatial_ndim=2)


class Crop3D(CropND):
    def __init__(
        self, size: int | tuple[int, int, int], start: int | tuple[int, int, int]
    ):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            size: size of the slice, either a single int or a tuple of two ints
            start: start of the slice, either a single int or a tuple of three ints for start along each axis
        """
        super().__init__(size, start, spatial_ndim=3)


class RandomCropND(pytc.TreeClass):
    def __init__(self, size: int | tuple[int, ...], spatial_ndim: int):
        """Applies jax.lax.dynamic_slice_in_dim with a random start along each axis

        Args:
            size: size of the slice
        """
        self.size = canonicalize(size, spatial_ndim, "size")
        self.spatial_ndim = spatial_ndim

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, *, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        start = tuple(
            jr.randint(key, shape=(), minval=0, maxval=x.shape[i] - s)
            for i, s in enumerate(self.size)
        )
        return jax.lax.dynamic_slice(x, (0, *start), (x.shape[0], *self.size))


class RandomCrop1D(RandomCropND):
    def __init__(self, size: int | tuple[int]):
        """Applies jax.lax.dynamic_slice_in_dim with a random start along each axis

        Args:
            size: size of the slice
        """
        super().__init__(size, spatial_ndim=1)


class RandomCrop2D(RandomCropND):
    def __init__(self, size: int | tuple[int, int]):
        """Applies jax.lax.dynamic_slice_in_dim with a random start along each axis

        Args:
            size: size of the slice
        """
        super().__init__(size, spatial_ndim=2)


class RandomCrop3D(RandomCropND):
    def __init__(self, size: int | tuple[int, int, int]):
        """Applies jax.lax.dynamic_slice_in_dim with a random start along each axis

        Args:
            size: size of the slice
        """
        super().__init__(size, spatial_ndim=3)
