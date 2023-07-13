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

import jax
import jax.numpy as jnp

import serket as sk
from serket.nn.utils import validate_spatial_ndim


class FlipLeftRight2D(sk.TreeClass):
    """Flip channels left to right.

    Note:
        https://github.com/deepmind/dm_pix/blob/master/dm_pix/_src/augment.py

    Examples:
        >>> x = jnp.arange(1,10).reshape(1,3, 3)
        >>> print(x)
        [[[1 2 3]
        [4 5 6]
        [7 8 9]]]

        >>> print(FlipLeftRight2D()(x))
        [[[3 2 1]
        [6 5 4]
        [9 8 7]]]
    """

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        flip = lambda x: jnp.flip(x, axis=1)
        return jax.vmap(flip)(x)

    @property
    def spatial_ndim(self) -> int:
        return 2


class FlipUpDown2D(sk.TreeClass):
    """Flip channels up to down.

    Note:
        https://github.com/deepmind/dm_pix/blob/master/dm_pix/_src/augment.py

    Examples:
        >>> x = jnp.arange(1,10).reshape(1,3, 3)
        >>> print(x)
        [[[1 2 3]
        [4 5 6]
        [7 8 9]]]

        >>> print(FlipUpDown2D()(x))
        [[[7 8 9]
        [4 5 6]
        [1 2 3]]]
    """

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        flip = lambda x: jnp.flip(x, axis=0)
        return jax.vmap(flip)(x)

    @property
    def spatial_ndim(self) -> int:
        return 2
