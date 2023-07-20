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

"""Define dispatchers for custom tree state."""

from __future__ import annotations

import functools as ft
from typing import Any, Callable, TypeVar

import jax

import serket as sk

T = TypeVar("T")


class NoState(sk.TreeClass):
    """No state placeholder."""

    def __init__(self, _: Any, __: Any):
        del _, __


def tree_state(tree: T, array: jax.Array | None = None) -> T:
    """Build state for a tree of layers.

    Some layers require state to be initialized before training. For example,
    `BatchNorm` layers require `running_mean` and `running_var` to be initialized
    before training. This function initializes the state for a tree of layers,
    based on the layer defined ``state`` rule using ``tree_state.def_state``.

    Args:
        tree: A tree of layers.
        array: An array to use for initializing state required by some layers
            (e.g. ConvGRUNDCell). default: ``None``.

    Returns:
        A tree of state leaves if it has state, otherwise ``None``.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> tree = [1, 2, sk.nn.BatchNorm(5)]
        >>> sk.tree_state(tree)
        [NoState(), NoState(), BatchNormState(
          running_mean=f32[5](μ=0.00, σ=0.00, ∈(0.00,0.00)),
          running_var=f32[5](μ=1.00, σ=0.00, ∈(1.00,1.00))
        )]

    Example:
        >>> # define state initialization rule for a custom layer
        >>> import jax
        >>> import serket as sk
        >>> class LayerWithState(sk.TreeClass):
        ...    pass
        >>> # state function accept the `layer` and optional input array as arguments
        >>> @sk.tree_state.def_state(LayerWithState)
        ... def _(leaf, _):
        ...    del _  # array is not used
        ...    return "some state"
        >>> sk.tree_state(LayerWithState())
        'some state'
        >>> sk.tree_state(LayerWithState(), jax.numpy.ones((1, 1)))
        'some state'
    """

    def is_leaf(x: Callable[[Any], bool]) -> bool:
        types = set(tree_state.state_dispatcher.registry.keys())
        types.discard(object)
        return isinstance(x, tuple(types))

    def dispatch_func(node):
        return tree_state.state_dispatcher(node, array)

    return jax.tree_map(dispatch_func, tree, is_leaf=is_leaf)


tree_state.state_dispatcher = ft.singledispatch(NoState)
tree_state.def_state = tree_state.state_dispatcher.register
