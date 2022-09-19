from __future__ import annotations

import jax.numpy as jnp
import pytreeclass as pytc


@pytc.treeclass
class AdaptiveReLU:
    # https://arxiv.org/pdf/1906.01170.pdf
    a: float

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jnp.maximum(0, self.a * x)


@pytc.treeclass
class AdaptiveLeakyReLU:
    # https://arxiv.org/pdf/1906.01170.pdf
    a: float
    v: float = pytc.static_field()

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jnp.maximum(0, self.a * x) - self.v * jnp.maximum(0, -self.a * x)


@pytc.treeclass
class AdaptiveSigmoid:
    # https://arxiv.org/pdf/1906.01170.pdf
    a: float

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return 1 / (1 + jnp.exp(-self.a * x))


@pytc.treeclass
class AdaptiveTanh:
    # https://arxiv.org/pdf/1906.01170.pdf
    a: float

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return (jnp.exp(self.a * x) - jnp.exp(-self.a * x)) / (
            jnp.exp(self.a * x) + jnp.exp(-self.a * x)
        )
