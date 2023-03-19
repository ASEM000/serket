from __future__ import annotations

import functools as ft
from typing import Any, Callable

import jax
import jax.random as jr
import pytreeclass as pytc

from serket.nn.callbacks import instance_cb_factory


@pytc.treeclass
class Lambda:
    func: Callable[[Any], Any] = pytc.field(callbacks=[pytc.freeze])

    def __init__(self, func: Callable[[Any], Any]):
        """A layer that applies a function to its input.

        Args:
            func: a function that takes a single argument and returns a jax.numpy.ndarray.

        Example:
            >>> import jax.numpy as jnp
            >>> from serket.nn import Lambda
            >>> layer = Lambda(lambda x: x + 1)
            >>> layer(jnp.array([1, 2, 3]))
            [2 3 4]
        """
        self.func = func

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class Sequential:
    layers: tuple[Any, ...] = pytc.field(callbacks=[instance_cb_factory(tuple)])

    def __init__(self, layers: tuple[Any, ...]):
        """A sequential container for layers.

        Args:
            layers: a tuple of layers.

        Example:
            >>> import jax.numpy as jnp
            >>> import jax.random as jr
            >>> from serket.nn import Sequential, Lambda
            >>> layers = Sequential((Lambda(lambda x: x + 1), Lambda(lambda x: x * 2)))
            >>> layers(jnp.array([1, 2, 3]), key=jr.PRNGKey(0))
        """
        self.layers = layers

    def __call__(self, x: jax.Array, *, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
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
        if isinstance(i, int):
            return self.layers[i]
        raise TypeError(f"Invalid index type: {type(i)}")

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __reversed__(self):
        return reversed(self.layers)
