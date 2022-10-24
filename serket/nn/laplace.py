from __future__ import annotations

import jax
import jax.numpy as jnp
import kernex as kex
import pytreeclass as pytc


@pytc.treeclass
class Laplace2D:
    def __init__(self):
        # apply laplace operator on channel axis
        @jax.vmap
        @kex.kmap(kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        def op(x):
            return -4 * x[1, 1] + x[0, 1] + x[2, 1] + x[1, 0] + x[1, 2]

        self._func = op

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        msg = f"Input must have 3 dimensions, got {x.ndim}."
        assert x.ndim == 3, msg
        return self._func(x)
