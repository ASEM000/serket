from __future__ import annotations

import functools as ft

import jax
import jax.random as jr
import pytreeclass as pytc

from serket.nn.callbacks import validate_spatial_in_shape
from serket.nn.utils import _canonicalize, _check_spatial_in_shape


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class CropND:
    size: int | tuple[int, ...] = pytc.field(callbacks=[pytc.freeze])
    start: int | tuple[int, ...] = pytc.field(callbacks=[pytc.freeze], default=0)

    def __init__(self, size, start, spatial_ndim):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            size: size of the slice
            start: start of the slice
        """
        self.size = _canonicalize(size, spatial_ndim, "size")
        self.start = _canonicalize(start, spatial_ndim, "start")
        self.spatial_ndim = spatial_ndim

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        _check_spatial_in_shape(x, self.spatial_ndim)
        return jax.lax.dynamic_slice(x, (0, *self.start), (x.shape[0], *self.size))


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class Crop1D(CropND):
    def __init__(self, size: int | tuple[int, ...], start: int | tuple[int, ...]):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input."""
        super().__init__(size, start, spatial_ndim=1)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class Crop2D(CropND):
    def __init__(self, size: int | tuple[int, ...], start: int | tuple[int, ...]):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input."""
        super().__init__(size, start, spatial_ndim=2)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class Crop3D(CropND):
    def __init__(self, size: int | tuple[int, ...], start: int | tuple[int, ...]):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input."""
        super().__init__(size, start, spatial_ndim=3)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class RandomCropND:
    def __init__(self, size, spatial_ndim):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            size: size of the slice
            start: start of the slice
        """
        self.size = _canonicalize(size, spatial_ndim, "size")
        self.spatial_ndim = spatial_ndim

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, *, key: jr.PRNGKey = jr.PRNGKey(0)) -> jax.Array:
        start = tuple(
            jr.randint(key, shape=(), minval=0, maxval=x.shape[i] - s)
            for i, s in enumerate(self.size)
        )
        return jax.lax.dynamic_slice(x, (0, *start), (x.shape[0], *self.size))


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class RandomCrop1D(RandomCropND):
    def __init__(self, size):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            size: size of the slice
            start: start of the slice
        """
        super().__init__(size, spatial_ndim=1)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class RandomCrop2D(RandomCropND):
    def __init__(self, size):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            size: size of the slice
            start: start of the slice
        """
        super().__init__(size, spatial_ndim=2)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class RandomCrop3D(RandomCropND):
    def __init__(self, size):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            size: size of the slice
            start: start of the slice
        """
        super().__init__(size, spatial_ndim=3)
