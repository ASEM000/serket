from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import kernex as kex
import pytreeclass as pytc


@pytc.treeclass
class AvgBlur2D:
    func: Callable = pytc.nondiff_field(repr=False)

    def __init__(self, kernel_size: int | tuple[int, int]):
        kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )

        # vectorize on channels dimension
        @jax.vmap
        @kex.kmap(kernel_size=kernel_size, padding="SAME")
        def op(x):
            kernel = jnp.ones([*kernel_size]) / jnp.array(kernel_size).prod()
            return jnp.sum(x * kernel, dtype=jnp.float32)

        self.func = op

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 3, "`Input` must be 3D."
        return self.func(x)
