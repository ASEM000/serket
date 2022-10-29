from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from serket.nn.utils import _check_and_return, _check_spatial_in_shape


@pytc.treeclass
class CropND:
    size: int | tuple[int, ...] = pytc.nondiff_field()
    start: int | tuple[int, ...] = pytc.nondiff_field(default=0)

    def __init__(self, size, start, ndim):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            size: size of the slice
            start: start of the slice
        """
        self.size = _check_and_return(size, ndim, "size")
        self.start = _check_and_return(start, ndim, "start")
        self.ndim = ndim

    @_check_spatial_in_shape
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.lax.dynamic_slice(x, (0, *self.start), (x.shape[0], *self.size))


@pytc.treeclass
class Crop1D(CropND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=1)


@pytc.treeclass
class Crop2D(CropND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=2)


@pytc.treeclass
class Crop3D(CropND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=3)


@pytc.treeclass
class RandomCropND:
    def __init__(self, size, ndim):
        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            size: size of the slice
            start: start of the slice
        """
        self.size = _check_and_return(size, ndim, "size")
        self.ndim = ndim

    @_check_spatial_in_shape
    def __call__(
        self, x: jnp.ndarray, *, key: jr.PRNGKey = jr.PRNGKey(0)
    ) -> jnp.ndarray:
        start = tuple(
            jr.randint(key, shape=(), minval=0, maxval=x.shape[i] - s)
            for i, s in enumerate(self.size)
        )

        return jax.lax.dynamic_slice(x, (0, *start), (x.shape[0], *self.size))


@pytc.treeclass
class RandomCrop1D(RandomCropND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=1)


@pytc.treeclass
class RandomCrop2D(RandomCropND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=2)


@pytc.treeclass
class RandomCrop3D(RandomCropND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=3)
