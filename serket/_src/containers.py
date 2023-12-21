# Copyright 2023 serket authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools as ft
from typing import Any, Callable, Sequence

import jax
import jax.random as jr

import serket as sk
from serket._src.custom_transform import tree_eval


@ft.singledispatch
def sequential(key: jax.Array, _1, _2):
    raise TypeError(f"Invalid {type(key)=}")


@sequential.register(type(None))
def _(key: None, layers: Sequence[Callable[..., Any]], array: Any):
    del key  # no key is supplied then no random number generation is needed
    return ft.reduce(lambda x, layer: layer(x), layers, array)


@sequential.register(jax.Array)
def _(key: jax.Array, layers: Sequence[Callable[..., Any]], array: Any):
    """Applies a sequence of layers to an array.

    Args:
        key: a random number generator key supplied to the layers.
        layers: a tuple callables.
        array: an array to apply the layers to.
    """
    for key, layer in zip(jr.split(key, len(layers)), layers):
        try:
            array = layer(array, key=key)
        except TypeError:
            array = layer(array)
    return array


class Sequential(sk.TreeClass):
    """A sequential container for layers.

    Args:
        layers: a tuple or a list of layers. if a list is passed, it will
            be casted to a tuple to maintain immutable behavior.

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import serket as sk
        >>> layers = sk.Sequential(lambda x: x + 1, lambda x: x * 2)
        >>> print(layers(jnp.array([1, 2, 3]), key=jr.PRNGKey(0)))
        [4 6 8]

    Note:
        Layer might be a function or a class with a ``__call__`` method, additionally
        it might have a key argument for random number generation.
    """

    def __init__(self, *layers):
        # use var args to enforce tuple type to maintain immutability
        self.layers = layers

    def __call__(self, input: jax.Array, *, key: jax.Array | None = None) -> jax.Array:
        return sequential(key, self.layers, input)

    @ft.singledispatchmethod
    def __getitem__(self, key):
        raise TypeError(f"Invalid index type: {type(key)}")

    @__getitem__.register(slice)
    def _(self, key: slice):
        # return a new Sequential object with the sliced layers
        return type(self)(*self.layers[key])

    @__getitem__.register(int)
    def _(self, key: int):
        return self.layers[key]

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __reversed__(self):
        return reversed(self.layers)


def random_choice(key: jax.Array, layers: tuple[Callable[..., Any], ...], array: Any):
    """Randomly selects one of the given layers/functions.

    Args:
        layers: variable number of layers/functions to select from.
        array: an array to apply the layer to.
        key: a random number generator key.
    """
    index = jr.randint(key, (), 0, len(layers))
    return jax.lax.switch(index, layers, array)


class RandomChoice(sk.TreeClass):
    """Randomly selects one of the given layers/functions.

    Args:
        layers: variable number of layers/functions to select from.

    Example:
        >>> import serket as sk
        >>> import jax.random as jr
        >>> key = jr.PRNGKey(0)
        >>> print(sk.RandomChoice(lambda x: x + 2, lambda x: x * 2)(1.0, key=key))
        3.0
        >>> key = jr.PRNGKey(10)
        >>> print(sk.RandomChoice(lambda x: x + 2, lambda x: x * 2)(1.0, key=key))
        2.0

    Note:
        Using :func:`tree_eval` will convert this layer to a :class:`.Sequential`
        to apply the all layers sequentially.

        >>> import serket as sk
        >>> layer = sk.RandomChoice(lambda x: x + 2, lambda x: x * 2)
        >>> # will apply all layers sequentially
        >>> print(sk.tree_eval(layer)(1.0))
        6.0

    Reference:
        - https://imgaug.readthedocs.io/en/latest/source/overview/meta.html#OneOf
        - https://pytorch.org/vision/main/generated/torchvision.transforms.RandomChoice.html
    """

    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, input: jax.Array, *, key: jax.Array):
        return random_choice(key, self.layers, input)


@tree_eval.def_eval(RandomChoice)
def tree_eval_sequential(layer) -> Sequential:
    return Sequential(*layer.layers)
