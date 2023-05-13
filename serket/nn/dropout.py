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
import pytreeclass as pytc
from jax import lax

from serket.nn.utils import Range, validate_spatial_in_shape


def dropout(x, *, p: float = 0.5, key: jr.KeyArray = jr.PRNGKey(0)):
    """Randomly zeroes some of the elements of the input tensor with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Args:
        p: probability of an element to be zeroed. Default: 0.5. Use `p`= 0.0
            to turn off dropout.
        key: random key
    """
    if p == 0:
        return x
    if p == 1:
        return jnp.zeros_like(x)
    keep_rate = 1 - p
    mask = jr.bernoulli(key, keep_rate, x.shape)
    return jnp.where(mask, x / keep_rate, 0)


def dropout_nd(x, *, p: float = 0.5, key: jr.KeyArray = jr.PRNGKey(0)):
    """Drops full feature maps along the channel axis.

    Args:
        p: fraction of an elements to be zeroed out. Default: 0.5.
            Use `p`= 0.0 to turn off dropout.
        key: random key
    """
    if p == 0:
        return x
    if p == 1:
        return jnp.zeros_like(x)
    keep_rate = 1 - p
    mask = jr.bernoulli(key, keep_rate, x.shape)
    return jnp.where(mask, x / keep_rate, 0)


class Dropout(pytc.TreeClass):
    r"""Randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.

    Args:
        p: probability of an element to be zeroed. Default: 0.5

    Example:
        >>> import serket as sk
        >>> import pytreeclass as pytc
        >>> layer = sk.nn.Dropout(0.5)
        >>> # change `p` to 0.0 to turn off dropout
        >>> layer = layer.at["p"].set(0.0, is_leaf=pytc.is_frozen)
    Note:
        Use `p`= 0.0 to turn off dropout.
    """

    p: float = pytc.field(default=0.5, callbacks=[Range(0, 1)])

    def __call__(self, x, *, key: jr.KeyArray = jr.PRNGKey(0)):
        return dropout(x, p=lax.stop_gradient(self.p), key=key)


class DropoutND(pytc.TreeClass):
    """Drops full feature maps along the channel axis.

    Args:
        p: fraction of an elements to be zeroed out

    Note:
        https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
        https://arxiv.org/abs/1411.4280

    Example:
        >>> layer = DropoutND(0.5)
        >>> layer(jnp.ones((1, 10)))
        [[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]]
    """

    spatial_ndim: int
    p: float = pytc.field(default=0.5, callbacks=[Range(0, 1)])

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x, *, key=jr.PRNGKey(0)):
        return dropout_nd(x, p=lax.stop_gradient(self.p), key=key)

    @property
    @abc.abstractmethod
    def spatial_ndim(self):
        ...


class Dropout1D(DropoutND):
    def __init__(self, p: float = 0.5):
        """Drops full feature maps along the channel axis.

        Args:

            p: fraction of an elements to be zeroed out

        Note:
            https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
            https://arxiv.org/abs/1411.4280

        Example:
            >>> layer = DropoutND(0.5)
            >>> layer(jnp.ones((1, 10)))
            [[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]]
        """
        super().__init__(p=p)

    @property
    def spatial_ndim(self) -> int:
        return 1


class Dropout2D(DropoutND):
    def __init__(self, p: float = 0.5):
        """Drops full feature maps along the channel axis.

        Args:

            p: fraction of an elements to be zeroed out

        Note:
            https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
            https://arxiv.org/abs/1411.4280

        Example:
            >>> layer = DropoutND(0.5)
            >>> layer(jnp.ones((1, 10)))
            [[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]]
        """
        super().__init__(p=p, spatial_ndim=2)

    @property
    def spatial_ndim(self) -> int:
        return 2


class Dropout3D(DropoutND):
    def __init__(self, p: float = 0.5):
        """Drops full feature maps along the channel axis.

        Args:

            p: fraction of an elements to be zeroed out

        Note:
            https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
            https://arxiv.org/abs/1411.4280

        Example:
            >>> layer = DropoutND(0.5)
            >>> layer(jnp.ones((1, 10)))
            [[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]]
        """
        super().__init__(p=p, spatial_ndim=3)

    @property
    def spatial_ndim(self) -> int:
        return 3
