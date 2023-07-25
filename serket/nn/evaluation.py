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

"""Define dispatchers for custom tree evaluation."""

from __future__ import annotations

import functools as ft
from typing import Any, Callable

import jax


def tree_evaluation(tree):
    """Modify tree layers to disable any trainning related behavior.

    For example, :class:`nn.Dropout` layer is replaced by an :class:`nn.Identity` layer
    and :class:`nn.BatchNorm` layer ``evaluation`` is set to ``True`` when
    evaluating the tree.

    Args:
        tree: A tree of layers.

    Returns:
        A tree of layers with evaluation behavior of same structure as ``tree``.

    Example:
        >>> # dropout is replaced by an identity layer in evaluation mode
        >>> # by registering `tree_evaluation.def_evaluation(sk.nn.Dropout, sk.nn.Identity)`
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> layer = sk.nn.Dropout(0.5)
        >>> sk.tree_evaluation(layer)
        Identity()
    """

    types: set[type] = set(tree_evaluation.evaluation_dispatcher.registry) - {object}

    def is_leaf(x: Callable[[Any], bool]) -> bool:
        return isinstance(x, tuple(types))

    return jax.tree_map(tree_evaluation.evaluation_dispatcher, tree, is_leaf=is_leaf)


tree_evaluation.evaluation_dispatcher = ft.singledispatch(lambda x: x)
tree_evaluation.def_evalutation = tree_evaluation.evaluation_dispatcher.register
