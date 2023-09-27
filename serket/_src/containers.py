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
from serket._src.utils import Range


@ft.singledispatch
def sequential(key: jr.KeyArray, _, __):
    raise TypeError(f"Invalid {type(key)=}")


@sequential.register(type(None))
def _(key: None, layers: Sequence[Callable[..., Any]], array: Any):
    del key  # no key is supplied then no random number generation is needed
    return ft.reduce(lambda x, layer: layer(x), layers, array)


@sequential.register(jax.Array)
def _(key: jr.KeyArray, layers: Sequence[Callable[..., Any]], array: Any):
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
        self.layers = layers

    def __call__(self, x: jax.Array, *, key: jr.KeyArray | None = None) -> jax.Array:
        return sequential(key, self.layers, x)

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


def random_apply(
    key: jr.KeyArray,
    layer: Sequence[Callable[..., Any]],
    array: Any,
    rate: float,
):
    """Randomly applies a layer with probability ``rate``.

    Args:
        layer: layer to apply.
        array: an array to apply the layer to.
        rate: probability of applying the layer
        key: a random number generator key.
    """
    return layer(array) if jr.bernoulli(key, rate) else array


@sk.autoinit
class RandomApply(sk.TreeClass):
    """Randomly applies a layer with probability ``rate``.

    Args:
        layer: layer to apply.
        rate: probability of applying the layer

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.RandomApply(sk.nn.MaxPool2D(kernel_size=2, strides=2), rate=0.0)
        >>> layer(jnp.ones((1, 10, 10))).shape
        (1, 10, 10)
        >>> layer = sk.RandomApply(sk.nn.MaxPool2D(kernel_size=2, strides=2), rate=1.0)
        >>> layer(jnp.ones((1, 10, 10))).shape
        (1, 5, 5)

    Note:
        Using :func:`tree_eval` will always apply the layer/function.

    Reference:
        - https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomApply
        - Use :func:`nn.Sequential` to apply multiple layers.
    """

    layer: Any
    rate: float = sk.field(default=0.5, on_setattr=[Range(0, 1)])

    def __call__(self, x: jax.Array, *, key: jr.KeyArray = jr.PRNGKey(0)):
        rate = jax.lax.stop_gradient(self.rate)
        return random_apply(key, self.layer, x, rate)


def random_choice(key: jr.KeyArray, layers: tuple[Callable[..., Any], ...], array: Any):
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

    def __call__(self, x: jax.Array, *, key: jr.KeyArray):
        return random_choice(key, self.layers, x)


def random_order(key:jr.KeyArray, layers: tuple[Callable[..., Any], ...], array: Any):
    """Randomly applies the given layers/functions in a random order.

    Args:
        layers: variable number of layers/functions to select from.
        array: an array to apply the layer to.
        key: a random number generator key.
    """
    k1,k2 = jr.split(key)
    indices = jr.permutation(k1, len(layers), independent=True)
    layers = tuple(layers[i] for i in indices)
    return sequential(k2, layers, array)


class RandomOrder(sk.TreeClass):
    """Randomly applies the given layers/functions in a random order.

    Args:
        layers: variable number of layers/functions to select from.

    Note:
        Using :func:`tree_eval` will convert this layer to a :Class:`.Sequential`
        to apply the all layers sequentially in a fixed order.

    Example:
        >>> import serket as sk
        >>> import jax.random as jr
        >>> k1 = jr.PRNGKey(0)
        >>> k2 = jr.PRNGKey(6)
        >>> def f1(x):
        ...     return x + 1
        >>> def f2(x):
        ...    return x ** 2
        >>> sk.RandomOrder(f1, f2)(2, key=k1)  # f1(f2(x))
        5
        >>> sk.RandomOrder(f1, f2)(2, key=k2)  # f2(f1(x))
        9
    """ 
    def __init__(self, *layers):
        self.layers = layers
    
    def __call__(self, x: jax.Array, *, key: jr.KeyArray):
        return random_order(key, self.layers, x)


@tree_eval.def_eval(RandomOrder)
@tree_eval.def_eval(RandomChoice)
def tree_eval_sequential(layer) -> Sequential:
    return Sequential(*layer.layers)

@tree_eval.def_eval(RandomApply)
def tree_eval_random_apply(layer: RandomApply):
    return layer.layer