from __future__ import annotations

from typing import Any, Callable, Sequence

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc


@pytc.treeclass
class Lambda:
    func: Callable = pytc.static_field()

    def __call__(self, x: jnp.ndarray, *, key: jr.PRNGKey | None = None) -> jnp.ndarray:
        return self.func(x)


@pytc.treeclass
class Sequential:
    layers: Sequence[Any]

    def __call__(self, x: jnp.ndarray, *, key: jr.PRNGKey | None = None) -> jnp.ndarray:
        keys = (
            jr.split(key, len(self.layers))
            if key is not None
            else [None] * len(self.layers)
        )
        for ki, layer in zip(keys, self.layers):
            x = layer(x, key=ki)
        return x
