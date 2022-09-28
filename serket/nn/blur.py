from __future__ import annotations

import jax
import jax.numpy as jnp
import kernex as kex
import pytreeclass as pytc

# from typing import Callable


@pytc.treeclass
class AvgBlur2D:
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


# @pytc.treeclass
# class GaussianBlur2D:
#     kernel_size: int = pytc.nondiff_field()
#     sigma: float = pytc.nondiff_field()
#     func: Callable = pytc.nondiff_field(repr=False, init=False)

#     def __post_init__(self):
#         # this implementation should be faster than
#         # https://github.com/deepmind/dm_pix/blob/master/dm_pix/_src/augment.py
#         # that uses  depthwise conv with seperable filters for kernel_size<13

#         d = self.kernel_size

#         x = jnp.linspace(-(d - 1) / 2.0, (d - 1) / 2.0, d)
#         w = jnp.exp(-0.5 * jnp.square(x) * jax.lax.rsqrt(self.sigma))
#         w = jnp.outer(w, w)
#         w = w / w.sum()

#         @jax.vmap
#         @kex.kmap(kernel_size=(d, d), padding="same")
#         def conv(x):
#             return jnp.sum(x * w)

#         self.func = conv

#     def __call__(self, x, **kwargs):
#         assert x.ndim == 3, "`Input` must be 3D."
#         return self.func(x)
