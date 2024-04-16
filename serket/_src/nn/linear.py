# Copyright 2024 serket authors
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
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr

import serket as sk
from serket._src.nn.activation import (
    ActivationFunctionType,
    ActivationType,
    resolve_act,
)
from serket._src.nn.initialization import resolve_init
from serket._src.utils.convert import tuplify
from serket._src.utils.dispatch import single_dispatch
from serket._src.utils.lazy import maybe_lazy_call, maybe_lazy_init
from serket._src.utils.typing import Batched, DType, InitType
from serket._src.utils.validate import validate_pos_int


@ft.lru_cache(maxsize=None)
def generate_einsum_pattern(
    lhs_ndim: int,
    rhs_ndim: int,
    in_axis: tuple[int, ...],
    out_axis: tuple[int, ...],
) -> tuple[str, str, str]:
    # helper function to generate the einsum pattern for linear layer
    # with flexible input and output axes
    lhs_alpha = "abcdefghijklmnopqrstuvwxyz"
    rhs_alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert (len(in_axis) + len(out_axis)) == rhs_ndim
    in_axis = [axis if axis >= 0 else axis + lhs_ndim for axis in in_axis]
    lhs = "".join(lhs_alpha[axis] for axis in range(lhs_ndim))
    rhs = rhs_alpha[: len(out_axis)] + "".join(lhs_alpha[axis] for axis in in_axis)
    rest_out = [lhs_alpha[axis] for axis in range(lhs_ndim) if axis not in in_axis]
    out = [None] * (len(out_axis) + len(rest_out))
    out_axis = [o if o >= 0 else o + len(out) for o in out_axis]
    for i, axis in enumerate(out_axis):
        out[axis] = rhs_alpha[i]
    out = "".join(rest_out.pop(0) if o is None else o for o in out)
    return lhs, rhs, out


@single_dispatch(argnum=1)
def linear(
    input: jax.Array,
    weight: Any,
    bias: jax.Array | None,
    in_axis: Sequence[int] = (-1,),
    out_axis: Sequence[int] = (-1,),
) -> jax.Array:
    """Apply a linear transformation to the input.

    Args:
        input: input array.
        weight: weight array with shape (out_feature_1, out_feature_2, ..., in_feature_1,
            in_feature_2, ...). `in_feature_i` corresponds to `in_axis[i]` and `out_feature_i`
            corresponds to `out_axis[i]`.
        bias: bias array with shape (out_feature_1, out_feature_2, ...) or ``None``
            for no bias.
        in_axis: axes to apply the linear layer to.
        out_axis: result axis.
    """
    del input, bias, in_axis, out_axis
    raise NotImplementedError(f"{type(weight)=}")


@linear.def_type(jax.Array)
def _(
    input: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None,
    in_axis: Sequence[int] = (-1,),
    out_axis: Sequence[int] = (-1,),
) -> jax.Array:
    in_axis, out_axis = tuple(in_axis), tuple(out_axis)
    lhs, rhs, out = generate_einsum_pattern(input.ndim, weight.ndim, in_axis, out_axis)
    result = jnp.einsum(f"{lhs},{rhs}->{out}", input, weight)

    if bias is None:
        return result

    bias = bias.reshape(*bias.shape, *[1] * (result.ndim - bias.ndim))
    bias = jnp.einsum(f"{''.join(sorted(out))}->{out}", bias)
    return result + bias


def is_lazy_call(instance, *_1, **_2) -> bool:
    return getattr(instance, "in_features", False) is None


def is_lazy_init(_1, in_features, *_2, **_3) -> bool:
    return in_features is None


def infer_in_features(instance, x, **_) -> tuple[int, ...]:
    in_axis = getattr(instance, "in_axis", ())
    return tuple(x.shape[i] for i in tuplify(in_axis))


updates = dict(in_features=infer_in_features)


class Linear(sk.TreeClass):
    """Apply a Linear Layer to input.

    Args:
        in_features: number of input features corresponding to ``in_axis``. Accepts
            int or tuple of ints.
        out_features: number of output features corresponding to ``out_axis``.
            Accepts int or tuple of ints.
        key: key to use for initializing the weight and biases.
        in_axis: axes to apply the linear layer to. Accepts int or tuple of ints.
            Defaults to -1.
        out_axis: result axis. Accepts int or tuple of ints. Defaults to -1.
        weight_init: weight initialization function. Defaults to ``glorot_uniform``.
        bias_init: bias initialization function. Defaults to ``zeros``.
        dtype: dtype of the weight and biases. ``float32``

    Example:
        Apply :class:`.Linear` layer t0 the last dimension of input

        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> input = jnp.ones([1, 2, 3, 4])
        >>> key = jr.PRNGKey(0)
        >>> layer = sk.nn.Linear(4, 5, key=key)
        >>> layer(input).shape
        (1, 2, 3, 5)

    Example:
        Apply :class:`.Linear` layer to first and second axes of input

        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> input = jnp.ones([1, 2, 3, 4])
        >>> in_axis = (0, 1)  # which axes to apply the linear layer to
        >>> in_features = (1, 2)  # number of input features corresponding to ``in_axis``
        >>> out_axis = (0, 2) # which axes to map the output to
        >>> out_features = (3, 4)  # number of output features corresponding to ``out_axis``
        >>> key = jr.PRNGKey(0)
        >>> layer = sk.nn.Linear(in_features, out_features, in_axis=in_axis, out_axis=out_axis, key=key)
        >>> layer(input).shape
        (3, 3, 4, 4)

    Note:
        :class:`.Linear` supports lazy initialization, meaning that the weight and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> key = jr.PRNGKey(0)
        >>> input = jnp.ones((10, 5, 4))
        >>> lazy = sk.nn.Linear(None, 12, in_axis=(0, 2), key=key)
        >>> _, material = sk.value_and_tree(lambda lazy: lazy(input))(lazy)
        >>> material.in_features
        (10, 4)
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int | Sequence[int] | None,
        out_features: int | Sequence[int],
        *,
        key: jax.Array,
        in_axis: int | Sequence[int] = -1,
        out_axis: int | Sequence[int] = -1,
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        dtype: DType = jnp.float32,
    ):

        in_features = tuplify(in_features)
        in_axis = tuplify(in_axis)

        out_features = tuplify(out_features)
        out_axis = tuplify(out_axis)

        if len(in_axis) != len(in_features):
            raise ValueError(f"{len(in_axis)=} != {len(in_features)=}")

        if len(out_axis) != len(out_features):
            raise ValueError(f"{len(out_axis)=} != {len(out_features)=}")

        if not (all(isinstance(i, int) for i in in_features)):
            raise TypeError(f"Expected tuple of ints for {in_features=}")

        if not (all(isinstance(i, int) for i in in_axis)):
            raise TypeError(f"Expected tuple of ints for {in_axis=}")

        self.in_features = in_features
        self.out_features = out_features
        self.in_axis = in_axis
        self.out_axis = out_axis
        self.weight_init = weight_init
        self.bias_init = bias_init

        k1, k2 = jr.split(key)

        weight_shape = (*out_features, *in_features)
        self.weight = resolve_init(weight_init)(k1, weight_shape, dtype)
        self.bias = resolve_init(bias_init)(k2, out_features, dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    def __call__(self, input: jax.Array) -> jax.Array:
        """Apply a linear transformation to the input."""
        return self.linear_op(
            input=input,
            weight=self.weight,
            bias=self.bias,
            in_axis=self.in_axis,
            out_axis=self.out_axis,
        )

    linear_op = staticmethod(linear)


class Identity(sk.TreeClass):
    """Identity layer. Returns the input."""

    def __call__(self, input: jax.Array, **_) -> jax.Array:
        return input


class Embedding(sk.TreeClass):
    """Defines an embedding layer.

    Args:
        in_features: vocabulary size.
        out_features: embedding size.
        key: random key to initialize the weight.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> # 10 words in the vocabulary, each word is represented by a 3 dimensional vector
        >>> key = jr.PRNGKey(0)
        >>> table = sk.nn.Embedding(10, 3, key=key)
        >>> # take the last word in the vocab
        >>> input = jnp.array([9])
        >>> output = table(input)
        >>> output.shape
        (1, 3)
    """

    def __init__(self, in_features: int, out_features: int, key: jax.Array):
        self.in_features = validate_pos_int(in_features)
        self.out_features = validate_pos_int(out_features)
        self.weight = jr.normal(key, (self.out_features, self.in_features))

    def __call__(self, input: jax.Array) -> jax.Array:
        """Embeds the input.

        Args:
            input: integer index input of subdtype integer.

        Returns:
            Embedding of the input.
        """
        if not jnp.issubdtype(input.dtype, jnp.integer):
            raise TypeError(f"{input.dtype=} is not a subdtype of integer")

        return self.weight.T[input]


def scan_linear(
    input: jax.Array,
    weight: Batched[jax.Array],
    bias: Batched[jax.Array] | None,
    act: ActivationFunctionType,
) -> jax.Array:
    # reduce the ``jaxpr`` size by using ``scan``
    # for the intermediate layers in MLP. can lower the compilation time
    if bias is None:

        def scan_func(input: jax.Array, weight: Batched[jax.Array]):
            return act(linear(input, weight, None)), None

        output, _ = jax.lax.scan(scan_func, input, weight)
        return output

    def scan_func(input: jax.Array, weight_bias: Batched[jax.Array]):
        weight, bias = weight_bias[..., :-1], weight_bias[..., -1]
        return act(linear(input, weight, bias)), None

    weight_bias = jnp.concatenate([weight, bias[:, :, None]], axis=-1)
    output, _ = jax.lax.scan(scan_func, input, weight_bias)
    return output


def infer_in_features(_1, x, **_2) -> int:
    return x.shape[-1]


updates = dict(in_features=infer_in_features)


class MLP(sk.TreeClass):
    """Multi-layer perceptron.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        hidden_features: Number of hidden units in each hidden layer.
        num_hidden_layers: Number of hidden layers including the output layer.
        key: Random number generator key.
        act: Activation function. Defaults to ``tanh``.
        weight_init: Weight initialization function. Defaults to ``glorot_uniform``.
        bias_init: Bias initialization function. Defaults to ``zeros``.
        dtype: dtype of the weight and biases. ``float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> key = jr.PRNGKey(0)
        >>> layer = sk.nn.MLP(1, 2, hidden_features=4, num_hidden_layers=2, key=key)
        >>> input = jnp.ones((3, 1))
        >>> layer(input).shape
        (3, 2)

    Note:
        - :class:`.MLP` with ``in_features=1``, ``out_features=2``, ``hidden_features=4``,
          ``num_hidden_layers=2`` is equivalent to ``[1, 4, 4, 2]`` which has one
          input layer (1, 4), one intermediate  layer (4, 4), and one output
          layer (4, 2) = ``num_hidden_layers + 1``

    Note:
        :class:`.MLP` supports lazy initialization, meaning that the weight and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> key = jr.PRNGKey(0)
        >>> lazy = sk.nn.MLP(None, 1, num_hidden_layers=2, hidden_features=10, key=key)
        >>> input = jnp.ones([1, 10])
        >>> _, material = sk.value_and_tree(lambda lazy: lazy(input))(lazy)
        >>> material.in_features
        10

    Note:
        :class:`.MLP` uses ``jax.lax.scan`` to reduce the ``jaxpr`` size.
        Leading to faster compilation times and smaller ``jaxpr`` size.

        >>> import serket as sk
        >>> import jax
        >>> import jax.numpy as jnp
        >>> # 10 hidden layers
        >>> mlp1 = sk.nn.MLP(1, 2, 5, 10, key=jax.random.PRNGKey(0))
        >>> # 50 hidden layers
        >>> mlp2 = sk.nn.MLP(1, 2, 5, 50, key=jax.random.PRNGKey(0))
        >>> jaxpr1 = jax.make_jaxpr(mlp1)(jnp.ones([10, 1]))
        >>> jaxpr2 = jax.make_jaxpr(mlp2)(jnp.ones([10, 1]))
        >>> # same number of equations irrespective of the number of hidden layers
        >>> assert len(jaxpr1.jaxpr.eqns) == len(jaxpr2.jaxpr.eqns)
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        num_hidden_layers: int,
        *,
        key: jax.Array,
        act: ActivationType = "tanh",
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        dtype: DType = jnp.float32,
    ):
        if hidden_features < 1:
            raise ValueError(f"`{hidden_features=}` must be positive.")

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.num_hidden_layers = num_hidden_layers

        in_key, mid_key, out_key = jr.split(key, 3)
        self.act = resolve_act(act)

        in_weight_shape = (hidden_features, in_features)
        k1, k2 = jr.split(in_key)
        self.in_weight = resolve_init(weight_init)(k1, in_weight_shape, dtype)
        self.in_bias = resolve_init(bias_init)(k2, (hidden_features,), dtype)

        k3, k4 = jr.split(mid_key)
        mid_weight_shape = (num_hidden_layers, hidden_features, hidden_features)
        self.mid_weight = resolve_init(weight_init)(k3, mid_weight_shape, dtype)
        mid_bias_shape = (num_hidden_layers, hidden_features)
        self.mid_bias = resolve_init(bias_init)(k4, mid_bias_shape, dtype)

        k5, k6 = jr.split(out_key)
        out_weight_shape = (out_features, hidden_features)
        self.out_weight = resolve_init(weight_init)(k5, out_weight_shape, dtype)
        self.out_bias = resolve_init(bias_init)(k6, (out_features,), dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    def __call__(self, input: jax.Array) -> jax.Array:
        input = self.act(linear(input, self.in_weight, self.in_bias))
        input = scan_linear(input, self.mid_weight, self.mid_bias, self.act)
        return linear(input, self.out_weight, self.out_bias)
