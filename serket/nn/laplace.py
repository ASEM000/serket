from __future__ import annotations

import jax
import kernex as kex
import pytreeclass as pytc

from serket.nn.utils import _check_spatial_in_shape


@pytc.treeclass
class Laplace2D:
    def __init__(self):
        # apply laplace operator on channel axis
        @jax.vmap
        @kex.kmap(kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        def op(x):
            return -4 * x[1, 1] + x[0, 1] + x[2, 1] + x[1, 0] + x[1, 2]

        self._func = op
        self.spatial_ndim = 2

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        _check_spatial_in_shape(x, self.spatial_ndim)
        return self._func(x)
