from __future__ import annotations

import functools as ft

import jax
import jax.random as jr
import pytreeclass as pytc

from serket.nn.callbacks import validate_spatial_in_shape
from serket.nn.utils import canonicalize


@pytc.treeclass
class CropND:
    size: int | tuple[int, ...] = pytc.field(callbacks=[pytc.freeze])
    start: int | tuple[int, ...] = pytc.field(callbacks=[pytc.freeze], default=0)

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
    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        return jax.lax.dynamic_slice(x, (0, *self.start), (x.shape[0], *self.size))


@pytc.treeclass
class Crop1D(CropND):
    def __init__(self, size: int | tuple[int, ...], start: int | tuple[int, ...]):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input."""
        super().__init__(size, start, spatial_ndim=1)


@pytc.treeclass
class Crop2D(CropND):
    def __init__(self, size: int | tuple[int, ...], start: int | tuple[int, ...]):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input."""
        super().__init__(size, start, spatial_ndim=2)


@pytc.treeclass
class Crop3D(CropND):
    def __init__(self, size: int | tuple[int, ...], start: int | tuple[int, ...]):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input."""
        super().__init__(size, start, spatial_ndim=3)


@pytc.treeclass
class RandomCropND:
    def __init__(self, size, spatial_ndim):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            size: size of the slice
            start: start of the slice
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


@pytc.treeclass
class RandomCrop1D(RandomCropND):
    def __init__(self, size):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            size: size of the slice
            start: start of the slice
        """
        super().__init__(size, spatial_ndim=1)


@pytc.treeclass
class RandomCrop2D(RandomCropND):
    def __init__(self, size):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            size: size of the slice
            start: start of the slice
        """
        super().__init__(size, spatial_ndim=2)


@pytc.treeclass
class RandomCrop3D(RandomCropND):
    def __init__(self, size):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            size: size of the slice
            start: start of the slice
        """
        super().__init__(size, spatial_ndim=3)
