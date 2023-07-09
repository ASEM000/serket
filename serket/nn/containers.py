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
import pytreeclass as pytc

from serket.nn.utils import IsInstance


class Sequential(pytc.TreeClass):
    """A sequential container for layers.

    Args:
        layers: a tuple or a list of layers. if a list is passed, it will
            be casted to a tuple to maintain immutable behavior.

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import serket as sk
        >>> layers = sk.nn.Sequential((lambda x: x + 1, lambda x: x * 2))
        >>> print(layers(jnp.array([1, 2, 3]), key=jr.PRNGKey(0)))
        [4 6 8]
    """

    # allow list then cast to tuple avoid mutability issues
    layers: tuple[Any, ...] = pytc.field(callbacks=[IsInstance((tuple, list)), tuple])

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
        return type(self)(self.layers[key])

    @__getitem__.register(int)
    def _(self, key: int):
        return self.layers[key]

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __reversed__(self):
        return reversed(self.layers)
