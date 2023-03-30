from __future__ import annotations

from typing import Any, Callable

import jax
import jax.random as jr
import pytreeclass as pytc

from serket.nn.callbacks import instance_cb_factory


@pytc.treeclass
class Lambda:
    """A layer that applies a function to its input.

    Args:
        func: a function that takes a single argument and returns a jax.numpy.ndarray.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> layer = sk.nn.Lambda(lambda x: x + 1)
        >>> print(layer(jnp.array([1, 2, 3])))
        [2 3 4]
    """

    func: Callable[[Any], Any] = pytc.field(callbacks=[pytc.freeze])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class Sequential:
    """A sequential container for layers.

    Args:
        layers: a tuple of layers.

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import serket as sk
        >>> layers = sk.nn.Sequential((sk.nn.Lambda(lambda x: x + 1), sk.nn.Lambda(lambda x: x * 2)))
        >>> print(layers(jnp.array([1, 2, 3]), key=jr.PRNGKey(0)))
        [4 6 8]
    """

    layers: tuple[Any, ...] = pytc.field(callbacks=[instance_cb_factory(tuple)])

    def __call__(self, x: jax.Array, *, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        for key, layer in zip(jr.split(key, len(self.layers)), self.layers):
            # assume that layer is a callable object
            # that takes x and key as arguments
            x = layer(x, key=key)
        return x

    def __getitem__(self, key: int | slice):
        if isinstance(key, slice):
            # return a new Sequential object with the sliced layers
            return self.__class__(self.layers[key])
        if isinstance(key, int):
            return self.layers[key]
        raise TypeError(f"Invalid index type: {type(key)}")

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __reversed__(self):
        return reversed(self.layers)
