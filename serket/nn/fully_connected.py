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

from typing import Any, Generic, Sequence, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr

import serket as sk
from serket.nn.activation import (
    ActivationFunctionType,
    ActivationType,
    resolve_activation,
)
from serket.nn.initialization import InitType
from serket.nn.linear import Linear

T = TypeVar("T")


class Batched(Generic[T]):
    pass


PyTree = Any


class FNN(sk.TreeClass):
    """Fully connected neural network

    Args:
        layers: Sequence of layer sizes
        act_func: a single Activation function to be applied between layers or
            ``len(layers)-2`` Sequence of activation functions applied between
            layers.
        weight_init_func: Weight initializer function.
        bias_init_func: Bias initializer function. Defaults to lambda key,
            shape: jnp.ones(shape).
        key: Random key for weight and bias initialization.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> fnn = sk.nn.FNN([10, 5, 2])
        >>> fnn(jnp.ones((3, 10))).shape
        (3, 2)

    Note:
        - layers argument yields ``len(layers) - 1`` linear layers with required
          ``len(layers)-2`` activation functions, for example, ``layers=[10, 5, 2]``
          yields 2 linear layers with weight shapes (10, 5) and (5, 2)
          and single activation function is applied between them.
        - :class:`.FNN` uses python ``for`` loop to apply layers and activation functions.

    """

    def __init__(
        self,
        layers: Sequence[int],
        *,
        act_func: ActivationType | tuple[ActivationType, ...] = "tanh",
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        keys = jr.split(key, len(layers) - 1)
        num_hidden_layers = len(layers) - 2

        if isinstance(act_func, tuple):
            if len(act_func) != (num_hidden_layers):
                raise ValueError(f"{len(act_func)=} != {(num_hidden_layers)=}")

            self.act_func = tuple(resolve_activation(act) for act in act_func)
        else:
            self.act_func = resolve_activation(act_func)

        self.layers = tuple(
            Linear(
                in_features=di,
                out_features=do,
                key=ki,
                weight_init_func=weight_init_func,
                bias_init_func=bias_init_func,
            )
            for (ki, di, do) in (zip(keys, layers[:-1], layers[1:]))
        )

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        *layers, last = self.layers

        if isinstance(self.act_func, tuple):
            for ai, li in zip(self.act_func, layers):
                x = ai(li(x))
        else:
            for li in layers:
                x = self.act_func(li(x))

        return last(x)


def _scan_batched_layer_with_single_activation(
    x: Batched[jax.Array],
    layer: Batched[Linear],
    act_func: ActivationFunctionType,
) -> jax.Array:
    if layer.bias is None:

        def scan_func(x: jax.Array, bias: Batched[jax.Array]):
            return act_func(x + bias), None

        x, _ = jax.lax.scan(scan_func, x, layer.weight)
        return x

    def scan_func(x: jax.Array, weight_bias: Batched[jax.Array]):
        weight, bias = weight_bias[..., :-1], weight_bias[..., -1]
        return act_func(x @ weight + bias), None

    weight_bias = jnp.concatenate([layer.weight, layer.bias[:, :, None]], axis=-1)
    x, _ = jax.lax.scan(scan_func, x, weight_bias)
    return x


def _scan_batched_layer_with_multiple_activations(
    x: Batched[jax.Array],
    layer: Batched[Linear],
    act_func: Sequence[ActivationFunctionType],
) -> jax.Array:
    if layer.bias is None:

        def scan_func(x_index: tuple[jax.Array, int], weight: Batched[jax.Array]):
            x, index = x_index
            x = jax.lax.switch(index, act_func, x @ weight)
            return (x, index + 1), None

        (x, _), _ = jax.lax.scan(scan_func, (x, 0), layer.weight)
        return x

    def scan_func(x_index: jax.Array, weight_bias: Batched[jax.Array]):
        x, index = x_index
        weight, bias = weight_bias[..., :-1], weight_bias[..., -1]
        x = jax.lax.switch(index, act_func, x @ weight + bias)
        return [x, index + 1], None

    weight_bias = jnp.concatenate([layer.weight, layer.bias[:, :, None]], axis=-1)
    (x, _), _ = jax.lax.scan(scan_func, [x, 0], weight_bias)
    return x


class MLP(sk.TreeClass):
    """Multi-layer perceptron.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        hidden_size: Number of hidden units in each hidden layer.
        num_hidden_layers: Number of hidden layers including the output layer.
        act_func: Activation function.
        weight_init_func: Weight initialization function.
        bias_init_func: Bias initialization function.
        key: Random number generator key.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> mlp = sk.nn.MLP(1, 2, hidden_size=4, num_hidden_layers=2)
        >>> mlp(jnp.ones((3, 1))).shape
        (3, 2)

    Note:
        - :class:`.MLP` with ``in_features=1``, ``out_features=2``, ``hidden_size=4``,
          ``num_hidden_layers=2`` is equivalent to ``[1, 4, 4, 2]`` which has one
          input layer (1, 4), one intermediate  layer (4, 4), and one output
          layer (4, 2) = ``num_hidden_layers + 1``

    Note:
        - :class:`.MLP` exploits same input/out size for intermediate layers to use
          ``jax.lax.scan``, which offers better compilation speed for large
          number of layers and producing a smaller ``jaxpr`` but could be
          slower than equivalent :class:`.FNN` for small number of layers.

        The following compares the size of ``jaxpr`` for :class:`.MLP` and :class:`.FNN`
        of equivalent layers.

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import numpy.testing as npt
        >>> fnn = sk.nn.FNN([1] + [4] * 100 + [2])
        >>> mlp = sk.nn.MLP(1, 2, hidden_size=4, num_hidden_layers=100)
        >>> x = jnp.ones((100, 1))
        >>> fnn_jaxpr = jax.make_jaxpr(fnn)(x)
        >>> mlp_jaxpr = jax.make_jaxpr(mlp)(x)
        >>> npt.assert_allclose(fnn(x), mlp(x), atol=1e-6)
        >>> len(fnn_jaxpr.jaxpr.eqns)
        403
        >>> len(mlp_jaxpr.jaxpr.eqns)
        10
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_size: int,
        num_hidden_layers: int,
        act_func: ActivationType | tuple[ActivationType, ...] = "tanh",
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        if hidden_size < 1:
            raise ValueError(f"`{hidden_size=}` must be positive.")

        keys = jr.split(key, num_hidden_layers + 1)

        if isinstance(act_func, tuple):
            if len(act_func) != (num_hidden_layers):
                raise ValueError(f"{len(act_func)=} != {(num_hidden_layers)=}")
            self.act_func = tuple(resolve_activation(act) for act in act_func)
        else:
            self.act_func = resolve_activation(act_func)

        kwargs = dict(weight_init_func=weight_init_func, bias_init_func=bias_init_func)

        def batched_linear(key) -> Batched[Linear]:
            return sk.tree_mask(Linear(hidden_size, hidden_size, key=key, **kwargs))

        self.layers = tuple(
            [Linear(in_features, hidden_size, key=keys[0], **kwargs)]
            + [sk.tree_unmask(jax.vmap(batched_linear)(keys[1:-1]))]
            + [Linear(hidden_size, out_features, key=keys[-1], **kwargs)]
        )

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        l0, lm, lh = self.layers

        if isinstance(self.act_func, tuple):
            a0, *ah = self.act_func
            x = a0(l0(x))
            x = _scan_batched_layer_with_multiple_activations(x, lm, ah)
            return lh(x)

        a0 = self.act_func
        x = a0(l0(x))
        x = _scan_batched_layer_with_single_activation(x, lm, a0)
        return lh(x)
