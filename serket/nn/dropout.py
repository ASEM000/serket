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

import abc
import functools as ft

import jax.numpy as jnp
import jax.random as jr
from jax import lax

import serket as sk
from serket.nn.utils import Range, validate_spatial_ndim


class Dropout(sk.TreeClass):
    """Randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.

    Args:
        p: probability of an element to be zeroed. Default: 0.5

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.nn.Dropout(0.5)
        >>> # change `p` to 0.0 to turn off dropout
        >>> layer = layer.at["p"].set(0.0, is_leaf=pytc.is_frozen)

    Note:
        Use `p`= 0.0 to turn off dropout.
    """

    p: float = sk.field(default=0.5, callbacks=[Range(0, 1)])

    def __call__(self, x, *, key: jr.KeyArray = jr.PRNGKey(0)):
        return jnp.where(
            (keep_prop := lax.stop_gradient(1 - self.p)) == 0.0,
            jnp.zeros_like(x),
            jnp.where(jr.bernoulli(key, keep_prop, x.shape), x / keep_prop, 0),
        )


class DropoutND(sk.TreeClass):
    """Drops full feature maps along the channel axis."""

    p: float = sk.field(default=0.5, callbacks=[Range(0, 1)])

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x, *, key=jr.PRNGKey(0)):
        # drops full feature maps along first axis.
        shape = (x.shape[0], *([1] * (x.ndim - 1)))

        return jnp.where(
            (keep_prop := lax.stop_gradient(1 - self.p)) == 0.0,
            jnp.zeros_like(x),
            jnp.where(jr.bernoulli(key, keep_prop, shape=shape), x / keep_prop, 0),
        )

    @property
    @abc.abstractmethod
    def spatial_ndim(self):
        ...


class Dropout1D(DropoutND):
    def __init__(self, p: float = 0.5):
        """Drops full feature maps along the channel axis.

        Args:
            p: fraction of an elements to be zeroed out.

        Example:
            >>> import serket as sk
            >>> import jax.numpy as jnp
            >>> layer = sk.nn.Dropout1D(0.5)
            >>> print(layer(jnp.ones((1, 10))))
            [[2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]]

        Note:
            https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
            https://arxiv.org/abs/1411.4280
        """
        super().__init__(p=p)

    @property
    def spatial_ndim(self) -> int:
        return 1


class Dropout2D(DropoutND):
    def __init__(self, p: float = 0.5):
        """Drops full feature maps along the channel axis.

        Args:
            p: fraction of an elements to be zeroed out.

        Note:
            https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
            https://arxiv.org/abs/1411.4280

        Example:
            >>> import serket as sk
            >>> import jax.numpy as jnp
            >>> layer = sk.nn.Dropout2D(0.5)
            >>> print(layer(jnp.ones((1, 5, 5))))  # doctest: +NORMALIZE_WHITESPACE
            [[[2. 2. 2. 2. 2.]
             [2. 2. 2. 2. 2.]
             [2. 2. 2. 2. 2.]
             [2. 2. 2. 2. 2.]
             [2. 2. 2. 2. 2.]]]
        """
        super().__init__(p=p)

    @property
    def spatial_ndim(self) -> int:
        return 2


class Dropout3D(DropoutND):
    def __init__(self, p: float = 0.5):
        """Drops full feature maps along the channel axis.

        Args:
            p: fraction of an elements to be zeroed out.

        Example:
            >>> import serket as sk
            >>> import jax.numpy as jnp
            >>> layer = sk.nn.Dropout3D(0.5)
            >>> print(layer(jnp.ones((1, 2, 2, 2))))  # doctest: +NORMALIZE_WHITESPACE
            [[[[2. 2.]
            [2. 2.]]
            <BLANKLINE>
            [[2. 2.]
            [2. 2.]]]]

        Note:
            https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
            https://arxiv.org/abs/1411.4280
        """
        super().__init__(p=p)

    @property
    def spatial_ndim(self) -> int:
        return 3
