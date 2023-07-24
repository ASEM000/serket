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

import jax
import jax.numpy as jnp

import serket as sk
from serket.nn.utils import IsInstance


@sk.autoinit
class Flatten(sk.TreeClass):
    """Flatten an array from dim `start_dim` to `end_dim` (inclusive).

    Args:
        start_dim: the first dim to flatten
        end_dim: the last dim to flatten (inclusive)

    Returns:
        a function that flattens a jnp.ndarray

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> sk.nn.Flatten(0,1)(jnp.ones([1,2,3,4,5])).shape
        (2, 3, 4, 5)
        >>> sk.nn.Flatten(0,2)(jnp.ones([1,2,3,4,5])).shape
        (6, 4, 5)
        >>> sk.nn.Flatten(1,2)(jnp.ones([1,2,3,4,5])).shape
        (1, 6, 4, 5)
        >>> sk.nn.Flatten(-1,-1)(jnp.ones([1,2,3,4,5])).shape
        (1, 2, 3, 4, 5)
        >>> sk.nn.Flatten(-2,-1)(jnp.ones([1,2,3,4,5])).shape
        (1, 2, 3, 20)
        >>> sk.nn.Flatten(-3,-1)(jnp.ones([1,2,3,4,5])).shape
        (1, 2, 60)

    Note:
        https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html?highlight=flatten#torch.nn.Flatten
    """

    start_dim: int = sk.field(default=0, callbacks=[IsInstance(int)])
    end_dim: int = sk.field(default=-1, callbacks=[IsInstance(int)])

    def __call__(self, x: jax.Array) -> jax.Array:
        start_dim = self.start_dim + (0 if self.start_dim >= 0 else x.ndim)
        end_dim = self.end_dim + 1 + (0 if self.end_dim >= 0 else x.ndim)
        return jax.lax.collapse(x, start_dim, end_dim)


@sk.autoinit
class Unflatten(sk.TreeClass):
    """Unflatten an array.

    Args:
        dim: the dim to unflatten.
        shape: the shape to unflatten to. accepts a tuple of ints.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> sk.nn.Unflatten(0, (1,2,3,4,5))(jnp.ones([120])).shape
        (1, 2, 3, 4, 5)
        >>> sk.nn.Unflatten(2,(2,3))(jnp.ones([1,2,6])).shape
        (1, 2, 2, 3)

    Note:
        https://pytorch.org/docs/stable/generated/torch.nn.Unflatten.html?highlight=unflatten
    """

    dim: int = sk.field(default=0, callbacks=[IsInstance(int)])
    shape: tuple = sk.field(default=None, callbacks=[IsInstance(tuple)])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        shape = list(x.shape)
        shape = [*shape[: self.dim], *self.shape, *shape[self.dim + 1 :]]
        return jnp.reshape(x, shape)
