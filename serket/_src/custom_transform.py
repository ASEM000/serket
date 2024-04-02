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

from inspect import getfullargspec
from typing import Any, TypeVar

import jax

import serket as sk
from serket._src.utils import single_dispatch

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
        :func:`.tree_state.def_state` on a function that accepts the layer as the
        first argument, for any additional arguments, use keyword only arguments.

        >>> import jax
        >>> import serket as sk
        >>> class LayerWithState(sk.TreeClass):
        ...    pass
        >>> # state function accept the `layer` and  input array
        >>> @sk.tree_state.def_state(LayerWithState)
        ... def _(leaf, *, input: jax.Array) -> jax.Array:
        ...    return jax.random.normal(jax.random.PRNGKey(0), input.shape)
        >>> sk.tree_state(LayerWithState(), input=jax.numpy.ones((1, 1))).shape
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
    # tree_state handles state initialization for different layers
    # like RNN cells, BatchNorm, KMeans, etc.
    # one challenge is that the state initialization rule for a layer
    # may depend only on the layer itself, or may depend on the layer
    # and the input. For example, the state initialization rule for
    # ConvRNN Cells depends on the layer and sample input, but the state
    # initialization rule for some RNN cells (e.g. LSTM) does not depend on the
    # input. This poses a challenge for the user to pass the correct input
    # to the state initialization rule.

    types = tuple(set(tree_state.dispatcher.registry) - {object})

    def is_leaf(node: Any) -> bool:
        return isinstance(node, types)

    def dispatch_func(leaf):
        try:
            return tree_state.dispatcher(leaf, **kwargs)

        except TypeError as e:
            # check if the leaf has a state rule

            for mro in type(leaf).__mro__[:-1]:
                if mro in (registry := tree_state.dispatcher.registry):
                    func = registry[mro]
                    break
            else:
                # no state rule is registered for this leaf
                # however type error is raised for other reasons
                raise type(e)(e)

            # the state rule is registered and the kwargs passed to `tree_state`
            # check if all necessary kwargs for this state rule are passed
            state_kwargs = getfullargspec(func).kwonlyargs

            if set(state_kwargs).issubset(set(kwargs)):
                # the state rule is registered and the kwargs passed to `tree_state`
                return func(leaf, **{key: kwargs[key] for key in state_kwargs})

            # the state rule is registered and the kwargs passed to `tree_state`
            # are not a subset of the kwargs needed by the state rule (not found)
            raise type(e)(
                f"{type(leaf)=} has a registered state rule {sk.tree_str(func)}."
                f"\nHowever, the  kwargs = {','.join(set(kwargs)-set(state_kwargs))}"
                f"are not passed to the state rule.\n{e}"
            )

    return jax.tree_map(dispatch_func, tree, is_leaf=is_leaf)


tree_state.dispatcher = single_dispatch(argnum=0)(NoState)
tree_state.def_state = tree_state.dispatcher.def_type


def tree_eval(tree):
    """Modify tree layers to disable any trainning related behavior.

    For example, :class:`nn.Dropout` layer is replaced by an :class:`nn.Identity` layer
    and :class:`nn.BatchNorm` layer is replaced by :class:`.EvalBatchNorm` layer when
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
        ...    def __call__(self, input: jax.Array) -> jax.Array:
        ...        return input + 1
        >>> input = jax.numpy.ones([3, 3])
        >>> add_one = AddOne()
        >>> print(add_one(input))  # add one to each element
        [[2. 2. 2.]
         [2. 2. 2.]
         [2. 2. 2.]]
        <BLANKLINE>
        >>> class AddOneEval(sk.TreeClass):
        ...    def __call__(self, input: jax.Array) -> jax.Array:
        ...        return input  # no-op
        <BLANKLINE>
        >>> # register `AddOne` to be replaced by `AddOneEval` in evaluation mode
        >>> @sk.tree_eval.def_eval(AddOne)
        ... def _(_: AddOne) -> AddOneEval:
        ...    return AddOneEval()
        >>> print(sk.tree_eval(add_one)(input))
        [[1. 1. 1.]
         [1. 1. 1.]
         [1. 1. 1.]]
    """

    types = tuple(set(tree_eval.dispatcher.registry) - {object})

    def is_leaf(node: Any) -> bool:
        return isinstance(node, types)

    return jax.tree_map(tree_eval.dispatcher, tree, is_leaf=is_leaf)


tree_eval.dispatcher = single_dispatch(argnum=0)(lambda x: x)
tree_eval.def_eval = tree_eval.dispatcher.def_type
