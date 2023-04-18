from __future__ import annotations

import functools as ft

import jax
import kernex as kex
import pytreeclass as pytc

from serket.nn.callbacks import validate_spatial_in_shape


class Laplace2D(pytc.TreeClass):
    def __init__(self):
        # apply laplace operator on channel axis
        @jax.vmap
        @kex.kmap(kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        def op(x):
            return -4 * x[1, 1] + x[0, 1] + x[2, 1] + x[1, 0] + x[1, 2]

        self._func = op
        self.spatial_ndim = 2

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self._func(x)
