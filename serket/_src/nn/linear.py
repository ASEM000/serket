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
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr

import serket as sk
from serket._src.nn.activation import (
    ActivationFunctionType,
    ActivationType,
    resolve_activation,
)
from serket._src.nn.initialization import resolve_init
from serket._src.utils.convert import tuplify
from serket._src.utils.lazy import maybe_lazy_call, maybe_lazy_init
from serket._src.utils.typing import DType, InitType
from serket._src.utils.validate import validate_pos_int

T = TypeVar("T")
PyTree = Any


class Batched(Generic[T]):
    pass


def linear(
    input: jax.Array,
    weight: Any,
    bias: jax.Array | None,
    in_axis: tuple[int, ...],
    out_axis: int,
) -> jax.Array:
    """A@B + C.

    Args:
        input: input array.
        weight: weight array. In the shape of (out_features, in_feature_1, in_feature_2, ...)
        bias: bias array. In the shape of (out_features,) or ``None`` for no bias.
        in_axis: which axes in the input to apply the linear layer to. ``tuple`` of ``ints``
            corresponding to the (in_feature_1, in_feature_2, ...)
        out_axis: the axis to put the result. accepts ``in`` values.
    """
    in_axis = [axis if axis >= 0 else axis + input.ndim for axis in in_axis]
    lhs = "".join(str(axis) for axis in range(input.ndim))  # 0, 1, 2, 3
    rhs = "F" + "".join(str(axis) for axis in in_axis)  # F, 1, 2, 3
    out = "".join(str(axis) for axis in range(input.ndim) if axis not in in_axis)
    out_axis = out_axis if out_axis >= 0 else out_axis + len(out) + 1
    out = out[:out_axis] + "F" + out[out_axis:]

    try:
        einsum_pattern = f"{lhs},{rhs}->{out}"
        result = jnp.einsum(einsum_pattern, input, weight)
    except ValueError as e:
        raise type(e)(f"{einsum_pattern=}, {e}")

    if bias is None:
        return result

    with jax.ensure_compile_time_eval():
        broadcast_shape = list(range(result.ndim))
        del broadcast_shape[out_axis]
    bias = jnp.expand_dims(bias, axis=broadcast_shape)
    return result + bias


def is_lazy_call(instance, *_1, **_2) -> bool:
    return getattr(instance, "in_features", False) is None


def is_lazy_init(_1, in_features, *_2, **_3) -> bool:
    return in_features is None


def infer_in_features(instance, x, **__) -> tuple[int, ...]:
    in_axis = getattr(instance, "in_axis", ())
    return tuple(x.shape[i] for i in tuplify(in_axis))


updates = dict(in_features=infer_in_features)


class Linear(sk.TreeClass):
    """Apply a Linear Layer to input.

    Args:
        in_features: number of input features corresponding to in_axis
        out_features: number of output features
        key: key to use for initializing the weights.
        in_axis: axes to apply the linear layer to. Accepts int or tuple of ints.
        out_axis: result axis. Defaults to -1.
        weight_init: weight initialization function. Defaults to ``glorot_uniform``.
        bias_init: bias initialization function. Defaults to ``zeros``.
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

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
        >>> in_features = (1, 2)  # number of input features corresponding to ``in_axis``
        >>> in_axis = (0, 1)  # which axes to apply the linear layer to
        >>> out_features = 5
        >>> out_axis = 0 # which axis to put the result
        >>> key = jr.PRNGKey(0)
        >>> layer = sk.nn.Linear(in_features, out_features, in_axis=in_axis, out_axis=out_axis, key=key)
        >>> layer(input).shape
        (5, 3, 4)

    Note:
        :class:`.Linear` supports lazy initialization, meaning that the weights and
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
        in_features: int | tuple[int, ...] | None,
        out_features: int,
        *,
        key: jax.Array,
        in_axis: int | tuple[int, ...] = -1,
        out_axis: int = -1,
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        dtype: DType = jnp.float32,
    ):

        in_features = tuplify(in_features)
        in_axis = tuplify(in_axis)

        if len(in_axis) != len(in_features):
            raise ValueError(f"{len(in_axis)=} != {len(in_features)=}")

        # arrange the in_features by the in_axis
        def compare(ik):
            return in_axis[ik[0]]

        _, in_features = zip(*sorted(enumerate(in_features), key=compare))

        self.in_features = in_features
        self.out_features = out_features
        self.in_axis = sorted(in_axis)
        self.out_axis = out_axis
        self.weight_init = weight_init
        self.bias_init = bias_init

        if not (all(isinstance(i, int) for i in self.in_features)):
            raise TypeError(f"Expected tuple of ints for {self.in_features=}")

        if not (all(isinstance(i, int) for i in self.in_axis)):
            raise TypeError(f"Expected tuple of ints for {self.in_axis=}")

        k1, k2 = jr.split(key)

        weight_shape = (out_features, *in_features)
        self.weight = resolve_init(weight_init)(k1, weight_shape, dtype)
        self.bias = resolve_init(bias_init)(k2, (self.out_features,), dtype)

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
        key: random key to initialize the weights.

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
            return act(input @ weight.T), None

        output, _ = jax.lax.scan(scan_func, input, weight)
        return output

    def scan_func(input: jax.Array, weight_bias: Batched[jax.Array]):
        weight, bias = weight_bias[..., :-1], weight_bias[..., -1]
        return act(input @ weight.T + bias), None

    weight_bias = jnp.concatenate([weight, bias[:, :, None]], axis=-1)
    output, _ = jax.lax.scan(scan_func, input, weight_bias)
    return output


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
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

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
        :class:`.MLP` supports lazy initialization, meaning that the weights and
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
        >>> material.in_linear.in_features
        (10,)

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

        keys = jr.split(key, num_hidden_layers + 1)
        self.act = resolve_activation(act)
        kwargs = dict(weight_init=weight_init, bias_init=bias_init, dtype=dtype)

        @jax.vmap
        def batched_linear(key: jax.Array) -> Batched[Linear]:
            layer = Linear(hidden_features, hidden_features, key=key, **kwargs)
            return sk.tree_mask(layer)

        self.in_linear = Linear(in_features, hidden_features, key=keys[0], **kwargs)
        self.mid_linear: Batched[Linear] = sk.tree_unmask(batched_linear(keys[1:-1]))
        self.out_linear = Linear(hidden_features, out_features, key=keys[-1], **kwargs)

    def __call__(self, input: jax.Array) -> jax.Array:
        input = self.act(self.in_linear(input))
        weight_h, bias_h = self.mid_linear.weight, self.mid_linear.bias
        input = scan_linear(input, weight_h, bias_h, self.act)
        return self.out_linear(input)
