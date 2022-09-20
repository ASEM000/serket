from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import kernex as kex
import pytreeclass as pytc


@pytc.treeclass
class Laplace2D:
    func: Callable = pytc.nondiff_field(repr=False)

    def __init__(self):
        # apply laplace operator on channel axis
        @jax.vmap
        @kex.kmap(kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        def op(x):
            return -4 * x[1, 1] + x[0, 1] + x[2, 1] + x[1, 0] + x[1, 2]

        self.func = op

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 3, "`Input` must be 3D."
        return self.func(x)
