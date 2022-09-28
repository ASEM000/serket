from __future__ import annotations

from typing import Any, Callable, Sequence

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc
from pytreeclass._src.tree_util import is_treeclass


@pytc.treeclass
class Lambda:
    func: Callable = pytc.nondiff_field()

    def __call__(self, x: jnp.ndarray, *, key: jr.PRNGKey | None = None) -> jnp.ndarray:
        return self.func(x)


@pytc.treeclass
class Sequential:
    layers: Sequence[Any]

    def __post_init__(self):
        # wrap non-treeclass with `Lambda`
        self.layers = tuple(
            layer if is_treeclass(layer) else Lambda(layer) for layer in self.layers
        )

    def __call__(self, x: jnp.ndarray, *, key: jr.PRNGKey | None = None) -> jnp.ndarray:
        keys = (
            jr.split(key, len(self.layers))
            if key is not None
            else [None] * len(self.layers)
        )
        for ki, layer in zip(keys, self.layers):
            x = layer(x, key=ki)
        return x

    def __getitem__(self, index: int | slice) -> Any:

        if isinstance(index, int):
            return self.layers[index]
        elif isinstance(index, slice):
            return Sequential(self.layers[index])
        else:
            raise TypeError(f"`index` must be int or slice. Found {type(index)}")

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __reversed__(self):
        return reversed(self.layers)

    def __contains__(self, item):
        return item in self.layers
