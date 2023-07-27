# Copyright 2023 Serket authors
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
from typing import Any

import jax
import jax.random as jr

import serket as sk
from serket.nn.custom_transform import tree_evaluation
from serket.nn.utils import Range


class Sequential(sk.TreeClass):
    """A sequential container for layers.

    Args:
        layers: a tuple or a list of layers. if a list is passed, it will
            be casted to a tuple to maintain immutable behavior.

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import serket as sk
        >>> layers = sk.nn.Sequential(lambda x: x + 1, lambda x: x * 2)
        >>> print(layers(jnp.array([1, 2, 3]), key=jr.PRNGKey(0)))
        [4 6 8]

    Note:
        Layer might be a function or a class with a ``__call__`` method, additionally
        it might have a key argument for random number generation.
    """

    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x: jax.Array, *, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        for key, layer in zip(jr.split(key, len(self.layers)), self.layers):
            try:
                x = layer(x, key=key)
            except TypeError:
                x = layer(x)
        return x

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


@sk.autoinit
class RandomApply(sk.TreeClass):
    """
    Randomly applies a layer with probability p.

    Args:
        layer: layer to apply.
        p: probability of applying the layer

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.nn.RandomApply(sk.nn.MaxPool2D(kernel_size=2, strides=2), p=0.0)
        >>> layer(jnp.ones((1, 10, 10))).shape
        (1, 10, 10)
        >>> layer = sk.nn.RandomApply(sk.nn.MaxPool2D(kernel_size=2, strides=2), p=1.0)
        >>> layer(jnp.ones((1, 10, 10))).shape
        (1, 5, 5)

    Reference:
        - https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomApply
        - Use :func:`nn.Sequential` to apply multiple layers.
    """

    layer: Any
    p: float = sk.field(default=0.5, callbacks=[Range(0, 1)])

    def __call__(self, x: jax.Array, key: jr.KeyArray = jr.PRNGKey(0)):
        p = jax.lax.stop_gradient(self.p)
        return self.layer(x) if jr.bernoulli(key, p) else x


@tree_evaluation.def_evaluation(RandomApply)
def tree_evaluation_random_apply(layer: RandomApply):
    return layer.layer
