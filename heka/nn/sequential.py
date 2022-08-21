from __future__ import annotations

from typing import Callable, Sequence

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc


@pytc.treeclass
class Lambda:
    func: Callable

    def __call__(self, x: jnp.ndarray, *, key: jr.PRNGKey | None = None) -> jnp.ndarray:
        return self.func(x)


@pytc.treeclass
class Sequential:
    def __init__(self, layers: Sequence[pytc.treeclass]):
        # Done like this for better repr
        # in essence instead of using a python container (list/tuple) to store the layers
        # we register each layer to the class separately
        for i, layer in enumerate(layers):
            setattr(self, f"Layer_{i}", layer)

    def __call__(self, x: jnp.ndarray, *, key: jr.PRNGKey | None = None) -> jnp.ndarray:
        layers = [self.__dict__[k] for k in self.__dict__ if k.startswith("Layer_")]
        keys = jr.split(key, len(layers)) if key is not None else [None] * len(layers)
        for ki, layer in zip(keys, layers):
            x = layer(x, key=ki)
        return x
