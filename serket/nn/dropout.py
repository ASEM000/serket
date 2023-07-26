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
from serket.nn.evaluation import tree_evaluation
from serket.nn.linear import Identity
from serket.nn.utils import Range, validate_spatial_ndim


@sk.autoinit
class Dropout(sk.TreeClass):
    """Drop some elements of the input tensor.

    Randomly zeroes some of the elements of the input tensor with
    probability ``p`` using samples from a Bernoulli distribution.

    Args:
        p: probability of an element to be zeroed. Default: 0.5

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.nn.Dropout(0.5)
        >>> # change `p` to 0.0 to turn off dropout
        >>> layer = layer.at["p"].set(0.0, is_leaf=sk.is_frozen)

    Note:
        Use :func:`.tree_evaluation` to turn off dropout during evaluation.

        >>> import serket as sk
        >>> layers = sk.nn.Sequential(sk.nn.Dropout(0.5), sk.nn.Linear(10, 10))
        >>> sk.tree_evaluation(layers)
        Sequential(
            layers=(
                Identity(),
                Linear(
                in_features=(10),
                out_features=10,
                weight=f32[10,10](μ=0.04, σ=0.43, ∈[-0.86,0.95]),
                bias=f32[10](μ=1.00, σ=0.00, ∈[1.00,1.00])
                )
            )
        )
    """

    p: float = sk.field(default=0.5, callbacks=[Range(0, 1)])

    def __call__(self, x, *, key: jr.KeyArray = jr.PRNGKey(0)):
        return jnp.where(
            (keep_prop := lax.stop_gradient(1 - self.p)) == 0.0,
            jnp.zeros_like(x),
            jnp.where(jr.bernoulli(key, keep_prop, x.shape), x / keep_prop, 0),
        )


@sk.autoinit
class DropoutND(sk.TreeClass):
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
        Use :func:`.tree_evaluation` to turn off dropout during evaluation.

        >>> import serket as sk
        >>> layers = sk.nn.Sequential(sk.nn.Dropout1D(0.5), sk.nn.Linear(10, 10))
        >>> sk.tree_evaluation(layers)
        Sequential(
            layers=(
                Identity(),
                Linear(
                in_features=(10),
                out_features=10,
                weight=f32[10,10](μ=0.04, σ=0.43, ∈[-0.86,0.95]),
                bias=f32[10](μ=1.00, σ=0.00, ∈[1.00,1.00])
                )
            )
        )

    Reference:
        - https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
        - https://arxiv.org/abs/1411.4280
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class Dropout2D(DropoutND):
    """Drops full feature maps along the channel axis.

    Args:
        p: fraction of an elements to be zeroed out.

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

    Note:
        Use :func:`.tree_evaluation` to turn off dropout during evaluation.

        >>> import serket as sk
        >>> layers = sk.nn.Sequential(sk.nn.Dropout2D(0.5), sk.nn.Linear(10, 10))
        >>> sk.tree_evaluation(layers)
        Sequential(
            layers=(
                Identity(),
                Linear(
                in_features=(10),
                out_features=10,
                weight=f32[10,10](μ=0.04, σ=0.43, ∈[-0.86,0.95]),
                bias=f32[10](μ=1.00, σ=0.00, ∈[1.00,1.00])
                )
            )
        )

    Reference:
        - https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
        - https://arxiv.org/abs/1411.4280
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class Dropout3D(DropoutND):
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
        Use :func:`.tree_evaluation` to turn off dropout during evaluation.

        >>> import serket as sk
        >>> layers = sk.nn.Sequential(sk.nn.Dropout2D(0.5), sk.nn.Linear(10, 10))
        >>> sk.tree_evaluation(layers)
        Sequential(
            layers=(
                Identity(),
                Linear(
                in_features=(10),
                out_features=10,
                weight=f32[10,10](μ=0.04, σ=0.43, ∈[-0.86,0.95]),
                bias=f32[10](μ=1.00, σ=0.00, ∈[1.00,1.00])
                )
            )
        )

    Reference:
        - https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
        - https://arxiv.org/abs/1411.4280
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


@tree_evaluation.def_evalutation(Dropout)
@tree_evaluation.def_evalutation(DropoutND)
def dropout_evaluation(_) -> Identity:
    # dropout is a no-op during evaluation
    return Identity()
