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

from __future__ import annotations

import functools as ft
from typing import Any, Generic, Sequence, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr

import serket as sk
from serket._src.nn.activation import (
    ActivationFunctionType,
    ActivationType,
    resolve_activation,
)
from serket._src.nn.initialization import DType, InitType, resolve_init
from serket._src.utils import maybe_lazy_call, maybe_lazy_init, positive_int_cb

T = TypeVar("T")


class Batched(Generic[T]):
    pass


PyTree = Any


def is_lazy_call(instance, *_, **__) -> bool:
    return getattr(instance, "in_features", False) is None


def is_lazy_init(_, in_features, *__, **___) -> bool:
    return in_features is None


def infer_multilinear_in_features(_, *x, **__) -> int | tuple[int, ...]:
    return x[0].shape[-1] if len(x) == 1 else tuple(xi.shape[-1] for xi in x)


updates = dict(in_features=infer_multilinear_in_features)


def multilinear(
    arrays: tuple[jax.Array, ...],
    weight: jax.Array,
    bias: jax.Array | None,
) -> jax.Array:
    """Apply a linear layer to multiple inputs"""

    def generate_einsum_string(degree: int) -> str:
        alpha = "".join(map(str, range(degree + 1)))
        xs_string = [f"...{i}" for i in alpha[:degree]]
        output_string = ",".join(xs_string)
        output_string += f",{alpha[:degree+1]}->...{alpha[degree]}"
        return output_string

    einsum_string = generate_einsum_string(len(arrays))
    out = jnp.einsum(einsum_string, *arrays, weight)
    return out if bias is None else (out + bias)


def general_linear(
    input: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None,
    axes: tuple[int, ...],
) -> jax.Array:
    """Apply a linear layer to input at axes defined by ``axes``"""

    # ensure negative axes
    def generate_einsum_string(*axes: tuple[int, ...]) -> str:
        axes = sorted(axes)
        total_axis = abs(min(axes))  # get the total number of axes
        alpha = "".join(map(str, range(total_axis + 1)))
        array_str = "..." + alpha[:total_axis]
        weight_str = "".join([array_str[axis] for axis in axes]) + alpha[total_axis]
        result_string = "".join([ai for ai in array_str if ai not in weight_str])
        result_string += alpha[total_axis]
        return f"{array_str},{weight_str}->{result_string}"

    axes = map(lambda i: i if i < 0 else i - input.ndim, axes)
    einsum_string = generate_einsum_string(*axes)
    out = jnp.einsum(einsum_string, input, weight)
    return out if bias is None else (out + bias)


class Linear(sk.TreeClass):
    """Linear layer with arbitrary number of inputs applied to last axis of each input

    Args:
        in_features: number of input features for each input. accepts a tuple of ints
            or a single int for single input. If ``None``, the layer is lazily
            initialized.
        out_features: number of output features.
        key: key for the random number generator.
        weight_init: function to initialize the weights. Defaults to ``glorot_uniform``.
        bias_init: function to initialize the bias. Defaults to ``zeros``.
        dtype: dtype of the weights and bias. defaults to ``jnp.float32``.

    Example:
        Linear layer example

        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> input = jnp.ones((1, 5))
        >>> layer = sk.nn.Linear(5, 6, key=jr.PRNGKey(0))
        >>> layer(input).shape
        (1, 6)

    Example:
        Bilinear layer example

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import serket as sk
        >>> array_1 = jnp.ones((1, 5))
        >>> array_2 = jnp.ones((1, 6))
        >>> layer = sk.nn.Linear((5,6), 7, key=jr.PRNGKey(0))
        >>> layer(array_1, array_2).shape
        (1, 7)

    Note:
        :class:`.Linear` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> k1, k2 = jr.split(jr.PRNGKey(0))
        >>> @sk.autoinit
        ... class Linears(sk.TreeClass):
        ...    l1: sk.nn.Linear = sk.nn.Linear(None, 32, key=k1)
        ...    l2: sk.nn.Linear = sk.nn.Linear((32,), 10, key=k2)
        ...    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x, y)))
        >>> lazy_linears = Linears()
        >>> x = jnp.ones([100, 28])
        >>> y = jnp.ones([100, 56])
        >>> _, material_linears = lazy_linears.at["__call__"](x, y)
        >>> material_linears.l1.in_features
        (28, 56)

    Note:
        The difference between :class:`.Linear` and :class:`.GeneralLinear` is that
        :class:`.Linear` applies the linear layer to the last axis of each input
        for possibly multiple inputs, while :class:`.GeneralLinear` applies the
        linear layer to the axes specified by ``in_axes`` of a single input.
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int | tuple[int, ...] | None,
        out_features: int,
        key: jax.Array,
        *,
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        dtype: DType = jnp.float32,
    ):
        if not isinstance(in_features, (int, tuple, type(None))):
            raise TypeError(f"{in_features=} must be `None`, `tuple` or `int`")
        if isinstance(in_features, int):
            in_features = (in_features,)
        self.in_features = in_features
        self.out_features = out_features
        self.weight_init = weight_init
        self.bias_init = bias_init
        k1, k2 = jr.split(key)
        weight_shape = (*self.in_features, self.out_features)
        self.weight = resolve_init(weight_init)(k1, weight_shape, dtype)
        self.bias = resolve_init(bias_init)(k2, (out_features,), dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    def __call__(self, *arrays) -> jax.Array:
        if len(arrays) != len(self.in_features):
            raise ValueError(f"{len(arrays)=} != {len(self.in_features)=}")
        return multilinear(arrays, self.weight, self.bias)


def infer_in_features(instance, x, **__) -> tuple[int, ...]:
    in_axes = getattr(instance, "in_axes", ())
    return tuple(x.shape[i] for i in in_axes)


updates = dict(in_features=infer_in_features)


class GeneralLinear(sk.TreeClass):
    """Apply a Linear Layer to input at in_axes

    Args:
        in_features: number of input features corresponding to in_axes
        out_features: number of output features
        key: key to use for initializing the weights.
        in_axes: axes to apply the linear layer to
        weight_init: weight initialization function. Defaults to ``glorot_uniform``.
        bias_init: bias initialization function. Defaults to ``zeros``.
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        Apply linear layer to first and second axes of input

        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> x = jnp.ones([1, 2, 3, 4])
        >>> in_features = (1, 2)
        >>> out_features = 5
        >>> in_axes = (0, 1)
        >>> key = jr.PRNGKey(0)
        >>> layer = sk.nn.GeneralLinear(in_features, out_features, in_axes=in_axes, key=key)
        >>> layer(x).shape
        (3, 4, 5)

    Note:
        :class:`.GeneralLinear` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` to call the layer with an input of known shape.

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> lazy_linear = sk.nn.GeneralLinear(None, 12, in_axes=(0, 2), key=jr.PRNGKey(0))
        >>> _, material_linear = lazy_linear.at['__call__'](jnp.ones((10, 5, 4)))
        >>> material_linear.in_features
        (10, 4)
        >>> material_linear(jnp.ones((10, 5, 4))).shape
        (5, 12)

    Note:
        The difference between :class:`.Linear` and :class:`.GeneralLinear` is that
        :class:`.Linear` applies the linear layer to the last axis of each input
        for possibly multiple inputs, while :class:`.GeneralLinear` applies the
        linear layer to the axes specified by ``in_axes`` of a single input.
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: tuple[int, ...] | None,
        out_features: int,
        *,
        key: jax.Array,
        in_axes: tuple[int, ...],
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        dtype: DType = jnp.float32,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.in_axes = in_axes
        self.weight_init = weight_init
        self.bias_init = bias_init

        if not (all(isinstance(i, int) for i in in_features)):
            raise TypeError(f"Expected tuple of ints for {in_features=}")

        if not (all(isinstance(i, int) for i in in_axes)):
            raise TypeError(f"Expected tuple of ints for {in_axes=}")

        if len(in_axes) != len(in_features):
            raise ValueError(f"{len(in_axes)=} != {len(in_features)=}")

        k1, k2 = jr.split(key)
        weight_shape = (*in_features, out_features)
        self.weight = resolve_init(weight_init)(k1, weight_shape, dtype)
        self.bias = resolve_init(bias_init)(k2, (self.out_features,), dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    def __call__(self, input: jax.Array) -> jax.Array:
        return general_linear(input, self.weight, self.bias, self.in_axes)


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
        >>> table = sk.nn.Embedding(10, 3, key=jr.PRNGKey(0))
        >>> # take the last word in the vocab
        >>> table(jnp.array([9]))
        Array([[0.43810904, 0.35078037, 0.13254273]], dtype=float32)
    """

    def __init__(self, in_features: int, out_features: int, key: jax.Array):
        self.in_features = positive_int_cb(in_features)
        self.out_features = positive_int_cb(out_features)
        self.weight = jr.uniform(key, (self.in_features, self.out_features))

    def __call__(self, input: jax.Array) -> jax.Array:
        """Embeds the input.

        Args:
            input: integer index input of subdtype integer.

        Returns:
            Embedding of the input.

        """
        if not jnp.issubdtype(input.dtype, jnp.integer):
            raise TypeError(f"{input.dtype=} is not a subdtype of integer")

        return jnp.take(self.weight, input, axis=0)


class FNN(sk.TreeClass):
    """Fully connected neural network

    Args:
        layers: Sequence of layer sizes
        key: Random number generator key.
        act: Activation function. Defaults to ``tanh``.
        weight_init: Weight initializer function. Defaults to ``glorot_uniform``.
        bias_init: Bias initializer function. Defaults to ``zeros``.
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> fnn = sk.nn.FNN([10, 5, 2], key=jr.PRNGKey(0))
        >>> fnn(jnp.ones((3, 10))).shape
        (3, 2)

    Note:
        - layers argument yields ``len(layers) - 1`` linear layers with required
          ``len(layers)-2`` activation functions, for example, ``layers=[10, 5, 2]``
          yields 2 linear layers with weight shapes (10, 5) and (5, 2)
          and single activation function is applied between them.
        - :class:`.FNN` uses python ``for`` loop to apply layers and activation functions.

    Note:
        :class:`.FNN` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, add ``None`` as the the first element of the
        ``layers`` argument and use the ``.at["__call__"]`` attribute
        to call the layer with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy_fnn = sk.nn.FNN([None, 10, 2, 1], key=jr.PRNGKey(0))
        >>> _, material_fnn = lazy_fnn.at['__call__'](jnp.ones([1, 10]))
        >>> material_fnn.linear_0.in_features
        (10,)
    """

    def __init__(
        self,
        layers: Sequence[int],
        *,
        key: jax.Array,
        act: ActivationType = "tanh",
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        dtype: DType = jnp.float32,
    ):
        keys = jr.split(key, len(layers) - 1)
        self.act = resolve_activation(act)

        for i, (di, do, ki) in enumerate(zip(layers[:-1], layers[1:], keys)):
            layer = Linear(
                in_features=di,
                out_features=do,
                key=ki,
                weight_init=weight_init,
                bias_init=bias_init,
                dtype=dtype,
            )
            setattr(self, f"linear_{i}", layer)

    def __call__(self, input: jax.Array) -> jax.Array:
        vs = vars(self)
        *layers, last = [vs[k] for k in vs if k.startswith("linear_")]
        for li in layers:
            input = self.act(li(input))
        return last(input)


def _scan_linear(
    input: jax.Array,
    weight: Batched[jax.Array],
    bias: Batched[jax.Array] | None,
    act: ActivationFunctionType,
) -> jax.Array:
    if bias is None:

        def scan_func(x: jax.Array, weight: Batched[jax.Array]):
            return act(x @ weight), None

        input, _ = jax.lax.scan(scan_func, input, weight)
        return input

    def scan_func(x: jax.Array, weight_bias: Batched[jax.Array]):
        weight, bias = weight_bias[..., :-1], weight_bias[..., -1]
        return act(x @ weight + bias), None

    weight_bias = jnp.concatenate([weight, bias[:, :, None]], axis=-1)
    input, _ = jax.lax.scan(scan_func, input, weight_bias)
    return input


class MLP(sk.TreeClass):
    """Multi-layer perceptron.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        key: Random number generator key.
        hidden_features: Number of hidden units in each hidden layer.
        num_hidden_layers: Number of hidden layers including the output layer.
        act: Activation function. Defaults to ``tanh``.
        weight_init: Weight initialization function. Defaults to ``glorot_uniform``.
        bias_init: Bias initialization function. Defaults to ``zeros``.
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> mlp = sk.nn.MLP(1, 2, hidden_features=4, num_hidden_layers=2, key=jr.PRNGKey(0))
        >>> mlp(jnp.ones((3, 1))).shape
        (3, 2)

    Note:
        - :class:`.MLP` with ``in_features=1``, ``out_features=2``, ``hidden_features=4``,
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
        >>> import jax.random as jr
        >>> import serket as sk
        >>> import numpy.testing as npt
        >>> fnn = sk.nn.FNN([1] + [4] * 100 + [2], key=jr.PRNGKey(0))
        >>> mlp = sk.nn.MLP(1, 2, hidden_features=4, num_hidden_layers=100, key=jr.PRNGKey(0))
        >>> x = jnp.ones((100, 1))
        >>> fnn_jaxpr = jax.make_jaxpr(fnn)(x)
        >>> mlp_jaxpr = jax.make_jaxpr(mlp)(x)
        >>> npt.assert_allclose(fnn(x), mlp(x), atol=1e-6)
        >>> len(fnn_jaxpr.jaxpr.eqns)
        403
        >>> len(mlp_jaxpr.jaxpr.eqns)
        10

    Note:
        :class:`.MLP` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy_mlp = sk.nn.MLP(None, 1, num_hidden_layers=2, hidden_features=10, key=jr.PRNGKey(0))
        >>> _, material_mlp = lazy_mlp.at['__call__'](jnp.ones([1, 10]))
        >>> material_mlp.linear_i.in_features
        (10,)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: jax.Array,
        hidden_features: int,
        num_hidden_layers: int,
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
            return sk.tree_mask(
                Linear(hidden_features, hidden_features, key=key, **kwargs)
            )

        self.linear_i = Linear(in_features, hidden_features, key=keys[0], **kwargs)
        self.linear_h = sk.tree_unmask(batched_linear(keys[1:-1]))
        self.linear_o = Linear(hidden_features, out_features, key=keys[-1], **kwargs)

    def __call__(self, input: jax.Array) -> jax.Array:
        input = self.act(self.linear_i(input))
        weight_h, bias_h = self.linear_h.weight, self.linear_h.bias
        input = _scan_linear(input, weight_h, bias_h, self.act)
        return self.linear_o(input)
