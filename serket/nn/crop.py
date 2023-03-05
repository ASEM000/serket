from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from serket.nn.utils import _canonicalize, _check_spatial_in_shape


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
        self.size = _canonicalize(size, spatial_ndim, "size")
        self.start = _canonicalize(start, spatial_ndim, "start")
        self.spatial_ndim = spatial_ndim

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        _check_spatial_in_shape(x, self.spatial_ndim)
        return jax.lax.dynamic_slice(x, (0, *self.start), (x.shape[0], *self.size))


@pytc.treeclass
class Crop1D(CropND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=1)


@pytc.treeclass
class Crop2D(CropND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=2)


@pytc.treeclass
class Crop3D(CropND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=3)


@pytc.treeclass
class RandomCropND:
    def __init__(self, size, spatial_ndim):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            size: size of the slice
            start: start of the slice
        """
        self.size = _canonicalize(size, spatial_ndim, "size")
        self.spatial_ndim = spatial_ndim

    def __call__(
        self, x: jnp.ndarray, *, key: jr.PRNGKey = jr.PRNGKey(0)
    ) -> jnp.ndarray:
        _check_spatial_in_shape(x, self.spatial_ndim)
        start = tuple(
            jr.randint(key, shape=(), minval=0, maxval=x.shape[i] - s)
            for i, s in enumerate(self.size)
        )

        return jax.lax.dynamic_slice(x, (0, *start), (x.shape[0], *self.size))


@pytc.treeclass
class RandomCrop1D(RandomCropND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=1)


@pytc.treeclass
class RandomCrop2D(RandomCropND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=2)


@pytc.treeclass
class RandomCrop3D(RandomCropND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=3)
