from __future__ import annotations

import jax.numpy as jnp
import kernex as kex
import pytreeclass as pytc


@pytc.treeclass
class Repeat1D:
    """repeats input along axis 1"""

    scale: int = pytc.nondiff_field(default=1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        @kex.kmap(kernel_size=(-1, -1), strides=(1, 1), padding="valid")
        def _repeat(x):
            return x.repeat(self.scale, axis=1)

        return jnp.squeeze(_repeat(x), axis=(1, 2))


@pytc.treeclass
class Repeat2D:
    """repeats input along axis 1,2"""

    scale: int = pytc.nondiff_field(default=1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        @kex.kmap(kernel_size=(-1, -1, -1), strides=(1, 1, 1), padding="valid")
        def _repeat(x):
            return x.repeat(self.scale, axis=2).repeat(self.scale, axis=1)

        return jnp.squeeze(_repeat(x), axis=(1, 2, 3))


@pytc.treeclass
class Repeat3D:
    """repeats input along axis 1,2,3"""

    scale: int = pytc.nondiff_field(default=1)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        @kex.kmap(kernel_size=(-1, -1, -1, -1), strides=(1, 1, 1, 1), padding="valid")
        def _repeat(x):
            return (
                x.repeat(self.scale, axis=3)
                .repeat(self.scale, axis=2)
                .repeat(self.scale, axis=1)
            )

        return jnp.squeeze(_repeat(x), axis=(1, 2, 3, 4))
