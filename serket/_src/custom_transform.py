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

"""Define dispatchers for custom tree transformations."""

from __future__ import annotations

import functools as ft
from inspect import getfullargspec
from typing import Any, Callable, TypeVar

import jax

import serket as sk

T = TypeVar("T")


class NoState(sk.TreeClass):
    """No state placeholder."""

    def __init__(self, layer: Any, **_):
        del layer, _


def tree_state(tree: T, **kwargs) -> T:
    """Build state for a tree of layers.

    Some layers require state to be initialized before training. For example,
    :class:`nn.BatchNorm` layers requires ``running_mean`` and ``running_var`` to
    be initialized before training. This function initializes the state for a
    tree of layers, based on the layer defined ``state`` rule using
    ``tree_state.def_state``.

    :func:`.tree_state` objective is to provide a simple and consistent way to
    initialize state for a tree of layers. Specifically, it provides a way to
    separate the state initialization logic from the layer definition. This
    allows for a more clear separation of concerns, and makes it easier to
    define new layers.

    Args:
        tree: A tree of layers.
        kwargs: Keyword arguments to pass to the state initialization rule. of the
            tree layers.

    Returns:
        A tree of state leaves if it has state, otherwise ``NoState`` placeholder.

    Note:
        To define a state initialization rule for a custom layer, use the decorator
        :func:`.tree_state.def_state` on a function that accepts the layer and
        keyword arguments. Meaning that the function should have the signature
        ``func(layer, *, user_kwargs, **rest_kwargs)``. The extra ``**rest_kwargs``
        is used to allow for additional keyword arguments to be passed to the state
        initialization rule for different layers.

        >>> import jax
        >>> import serket as sk
        >>> class LayerWithState(sk.TreeClass):
        ...    pass
        >>> # state function accept the `layer` and  input array
        >>> @sk.tree_state.def_state(LayerWithState)
        ... def _(leaf, **kwargs):
        ...    # pull out some keyword argument
        ...    array = kwargs["array"]
        ...    return jax.random.normal(jax.random.PRNGKey(0), array.shape)
        >>> sk.tree_state(LayerWithState(), array=jax.numpy.ones((1, 1))).shape
        (1, 1)

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> tree = [1, 2, sk.nn.BatchNorm(5, key=jr.PRNGKey(0))]
        >>> sk.tree_state(tree)
        [NoState(), NoState(), BatchNormState(
          running_mean=f32[5](μ=0.00, σ=0.00, ∈[0.00,0.00]),
          running_var=f32[5](μ=1.00, σ=0.00, ∈[1.00,1.00])
        )]
    """

    types = tuple(set(tree_state.state_dispatcher.registry) - {object})

    def is_leaf(x: Any) -> bool:
        return isinstance(x, types)

    def dispatch_func(leaf):
        return tree_state.state_dispatcher(leaf, **kwargs)

    return jax.tree_map(dispatch_func, tree, is_leaf=is_leaf)


tree_state.state_dispatcher = ft.singledispatch(NoState)


def def_state(klass: type, func: Callable[..., Any] | None = None):
    # check if the function pass `**kwargs` before registering
    def check_func(func: Callable[..., Any]) -> Callable[..., Any]:
        if getfullargspec(func).varkw is not None:
            return func
        raise TypeError(
            f"Upon registering {klass} to `tree_state`,"
            f" the function {func} is missing variable keyword argument."
        )

    if func is None:
        return lambda F: tree_state.state_dispatcher.register(klass, check_func(F))

    return tree_state.state_dispatcher.register(klass, check_func(func))


tree_state.def_state = def_state


def tree_eval(tree):
    """Modify tree layers to disable any trainning related behavior.

    For example, :class:`nn.Dropout` layer is replaced by an :class:`nn.Identity` layer
    and :class:`nn.BatchNorm` layer is replaced by :class:`nn.EvalNorm` layer when
    evaluating the tree.

    :func:`.tree_eval` objective is to provide a simple and consistent way to
    disable any trainning related behavior for a tree of layers. Specifically,
    it provides a way to separate the evaluation logic from the layer definition.
    This allows for a more clear separation of concerns, and makes it easier to
    define new layers that has a single behavior.

    Args:
        tree: A tree of layers.

    Returns:
        A tree of layers with evaluation behavior of same structure as ``tree``.

    Example:
        >>> # dropout is replaced by an identity layer in evaluation mode
        >>> # by registering `tree_eval.def_eval(sk.nn.Dropout, sk.nn.Identity)`
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> layer = sk.nn.Dropout(0.5)
        >>> sk.tree_eval(layer)
        Identity()

    Note:
        To define evaluation rule for a custom layer, use the decorator
        :func:`.tree_eval.def_eval` on a function that accepts the layer. The
        function should return the evaluation layer.

        >>> import serket as sk
        >>> import jax
        >>> class AddOne(sk.TreeClass):
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return x + 1
        >>> x = jax.numpy.ones([3, 3])
        >>> add_one = AddOne()
        >>> print(add_one(x))  # add one to each element
        [[2. 2. 2.]
         [2. 2. 2.]
         [2. 2. 2.]]
        <BLANKLINE>
        >>> class AddOneEval(sk.TreeClass):
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return x  # no-op
        <BLANKLINE>
        >>> # register `AddOne` to be replaced by `AddOneEval` in evaluation mode
        >>> @sk.tree_eval.def_eval(AddOne)
        ... def _(_: AddOne) -> AddOneEval:
        ...    return AddOneEval()
        >>> print(sk.tree_eval(add_one)(x))
        [[1. 1. 1.]
         [1. 1. 1.]
         [1. 1. 1.]]
    """

    types = tuple(set(tree_eval.eval_dispatcher.registry) - {object})

    def is_leaf(x: Any) -> bool:
        return isinstance(x, types)

    return jax.tree_map(tree_eval.eval_dispatcher, tree, is_leaf=is_leaf)


tree_eval.eval_dispatcher = ft.singledispatch(lambda x: x)
tree_eval.def_eval = tree_eval.eval_dispatcher.register
