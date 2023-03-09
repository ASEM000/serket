from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from serket.nn.callbacks import instance_cb


@pytc.treeclass
class Lambda:
    func: Callable[[Any], jnp.ndarray] = pytc.field(callbacks=[pytc.freeze])

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return self.func(x)


@pytc.treeclass
class Sequential:
    layers: tuple[Any, ...] = pytc.field(callbacks=[instance_cb(tuple)])

    def __call__(self, x: jnp.ndarray, *, key: jr.PRNGKey | None = None) -> jnp.ndarray:
        key = key or jr.PRNGKey(0)
        key = jr.split(key, len(self.layers))

        for key, layer in zip(key, self.layers):
            # assume that layer is a callable object
            # that takes x and key as arguments
            x = layer(x, key=key)
        return x

    def __getitem__(self, i: int | slice):
        if isinstance(i, slice):
            # return a new Sequential object with the sliced layers
            return self.__class__(self.layers[i])
        return self.layers[i]

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __reversed__(self):
        return reversed(self.layers)
