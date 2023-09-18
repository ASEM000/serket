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


@ft.lru_cache(maxsize=None)
def _multilinear_einsum_string(degree: int) -> str:
    # Generate einsum string for a linear layer of degree n
    # Example:
    #     >>> _multilinear_einsum_string(1)
    #     '...a,ab->....b'
    #     >>> _multilinear_einsum_string(2)
    #     '...a,...b,abc->....c'
    alpha = "".join(map(str, range(degree + 1)))
    xs_string = [f"...{i}" for i in alpha[:degree]]
    output_string = ",".join(xs_string)
    output_string += f",{alpha[:degree+1]}->...{alpha[degree]}"
    return output_string


@ft.lru_cache(maxsize=None)
def _general_linear_einsum_string(*axes: tuple[int, ...]) -> str:
    # Return the einsum string for a general linear layer.
    # Example:
    #     # apply linear layer to last axis
    #     >>> _general_linear_einsum_string(-1)
    #     '...0,01->...1'

    #     # apply linear layer to last two axes
    #     >>> _general_linear_einsum_string(-1,-2)
    #     '...01,012->...2'

    #     # apply linear layer to second last axis
    #     >>> _general_linear_einsum_string(-2)
    #     '...01,02->...12'

    #     # apply linear layer to last and third last axis
    #     >>> _general_linear_einsum_string(-1,-3)
    #     '...012,023->...13'

    if not all([i < 0 for i in axes]):
        raise ValueError("axes should be negative")

    axes = sorted(axes)
    total_axis = abs(min(axes))  # get the total number of axes
    alpha = "".join(map(str, range(total_axis + 1)))
    input_string = "..." + alpha[:total_axis]
    weight_string = "".join([input_string[axis] for axis in axes]) + alpha[total_axis]
    result_string = "".join([ai for ai in input_string if ai not in weight_string])
    result_string += alpha[total_axis]
    return f"{input_string},{weight_string}->{result_string}"


class Multilinear(sk.TreeClass):
    """Linear layer with arbitrary number of inputs applied to last axis of each input

    Args:
        in_features: number of input features for each input
        out_features: number of output features
        key: key for the random number generator.
        weight_init: function to initialize the weights. Defaults to ``glorot_uniform``.
        bias_init: function to initialize the bias. Defaults to ``zeros``.
        dtype: dtype of the weights and bias. defaults to ``jnp.float32``.

    Example:
        >>> # Bilinear layer
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import serket as sk
        >>> layer = sk.nn.Multilinear((5,6), 7, key=jr.PRNGKey(0))
        >>> layer(jnp.ones((1,5)), jnp.ones((1,6))).shape
        (1, 7)

        >>> # Trilinear layer
        >>> layer = sk.nn.Multilinear((5,6,7), 8, key=jr.PRNGKey(0))
        >>> layer(jnp.ones((1,5)), jnp.ones((1,6)), jnp.ones((1,7))).shape
        (1, 8)

    Note:
        :class:`.Multilinear` supports lazy initialization, meaning that the weights and
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
        ...    l1: sk.nn.Multilinear = sk.nn.Multilinear(None, 32, key=k1)
        ...    l2: sk.nn.Multilinear = sk.nn.Multilinear((32,), 10, key=k2)
        ...    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x, y)))
        >>> lazy_linears = Linears()
        >>> x = jnp.ones([100, 28])
        >>> y = jnp.ones([100, 56])
        >>> _, materialized_linears = lazy_linears.at["__call__"](x, y)
        >>> materialized_linears.l1.in_features
        (28, 56)
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: tuple[int, ...] | None,
        out_features: int,
        key: jr.KeyArray,
        *,
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        dtype: DType = jnp.float32,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.weight_init = weight_init
        self.bias_init = bias_init

        if not isinstance(in_features, (tuple, int)):
            raise ValueError(f"Expected tuple or int for {in_features=}.")

        k1, k2 = jr.split(key)
        weight_shape = (*in_features, out_features)
        self.weight = resolve_init(weight_init)(k1, weight_shape, dtype)
        self.bias = resolve_init(bias_init)(k2, (out_features,), dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    def __call__(self, *x) -> jax.Array:
        einsum_string = _multilinear_einsum_string(len(self.in_features))
        x = jnp.einsum(einsum_string, *x, self.weight)
        return x if self.bias is None else (x + self.bias)


class Linear(Multilinear):
    """Linear layer with 1 input applied to last axis of input

    Args:
        in_features: number of input features
        out_features: number of output features
        key: key for the random number generator.
        weight_init: function to initialize the weights. Defaults to ``glorot_uniform``.
        bias_init: function to initialize the bias. Defaults to ``zeros``.
        dtype: data type of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> layer = sk.nn.Linear(5, 6, key=jr.PRNGKey(0))
        >>> layer(jnp.ones((1,5))).shape
        (1, 6)

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
        ...    l1: sk.nn.Linear = sk.nn.Linear(None, 32,key=k1)
        ...    l2: sk.nn.Linear = sk.nn.Linear(32, 10,key=k2)
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> lazy_linears = Linears()
        >>> x = jnp.ones((100, 28, 28)).reshape(100, -1)
        >>> _, materialized_linears = lazy_linears.at["__call__"](x)
        >>> materialized_linears.l1.in_features
        (784,)
    """

    def __init__(
        self,
        in_features: int | None,
        out_features: int,
        *,
        key: jr.KeyArray,
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        dtype: DType = jnp.float32,
    ):
        super().__init__(
            in_features if in_features is None else (in_features,),
            out_features,
            key=key,
            weight_init=weight_init,
            bias_init=bias_init,
            dtype=dtype,
        )


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
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> x = jnp.ones([1, 2, 3, 4])
        >>> layer = sk.nn.GeneralLinear(in_features=(1, 2), in_axes=(0, 1), out_features=5, key=jr.PRNGKey(0))
        >>> layer(x).shape
        (3, 4, 5)

    Note:
        This layer is similar to to ``flax`` linen's ``DenseGeneral``, the difference
        is that this layer uses einsum to apply the linear layer to the specified axes.

    Note:
        :class:`.GeneralLinear` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> lazy_linear = sk.nn.GeneralLinear(None, 12, in_axes=(0, 2), key=jr.PRNGKey(0))
        >>> _, materialized_linear = lazy_linear.at['__call__'](jnp.ones((10, 5, 4)))
        >>> materialized_linear.in_features
        (10, 4)
        >>> materialized_linear(jnp.ones((10, 5, 4))).shape
        (5, 12)
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: tuple[int, ...] | None,
        out_features: int,
        *,
        key: jr.KeyArray,
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
    def __call__(self, x: jax.Array) -> jax.Array:
        # ensure negative axes
        axes = map(lambda i: i if i < 0 else i - x.ndim, self.in_axes)
        einsum_string = _general_linear_einsum_string(*axes)
        x = jnp.einsum(einsum_string, x, self.weight)
        return x


class Identity(sk.TreeClass):
    """Identity layer. Returns the input."""

    def __call__(self, x: jax.Array, **_) -> jax.Array:
        return x


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

    def __init__(self, in_features: int, out_features: int, key: jr.KeyArray):
        self.in_features = positive_int_cb(in_features)
        self.out_features = positive_int_cb(out_features)
        self.weight = jr.uniform(key, (self.in_features, self.out_features))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Embeds the input.

        Args:
            x: integer index array of subdtype integer.

        Returns:
            Embedding of the input.

        """
        if not jnp.issubdtype(x.dtype, jnp.integer):
            raise TypeError(f"{x.dtype=} is not a subdtype of integer")

        return jnp.take(self.weight, x, axis=0)


class FNN(sk.TreeClass):
    """Fully connected neural network

    Args:
        layers: Sequence of layer sizes
        key: Random number generator key.
        act: a single Activation function to be applied between layers or
            ``len(layers)-2`` Sequence of activation functions applied between
            layers. Defaults to ``tanh``.
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
        >>> _, materialized_fnn = lazy_fnn.at['__call__'](jnp.ones([1, 10]))
        >>> materialized_fnn.layers[0].in_features
        (10,)
    """

    def __init__(
        self,
        layers: Sequence[int],
        *,
        key: jr.KeyArray,
        act: ActivationType | Sequence[ActivationType] = "tanh",
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        dtype: DType = jnp.float32,
    ):
        keys = jr.split(key, len(layers) - 1)
        num_hidden_layers = len(layers) - 2

        if isinstance(act, tuple):
            if len(act) != (num_hidden_layers):
                raise ValueError(f"{len(act)=} != {(num_hidden_layers)=}")

            self.act = tuple(resolve_activation(act) for act in act)
        else:
            self.act = resolve_activation(act)

        self.layers = tuple(
            Linear(
                in_features=di,
                out_features=do,
                key=ki,
                weight_init=weight_init,
                bias_init=bias_init,
                dtype=dtype,
            )
            for (ki, di, do) in (zip(keys, layers[:-1], layers[1:]))
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        *layers, last = self.layers

        if isinstance(self.act, tuple):
            for ai, li in zip(self.act, layers):
                x = ai(li(x))
        else:
            for li in layers:
                x = self.act(li(x))

        return last(x)


def _scan_batched_layer_with_single_activation(
    x: Batched[jax.Array],
    layer: Batched[Linear],
    act: ActivationFunctionType,
) -> jax.Array:
    if layer.bias is None:

        def scan_func(x: jax.Array, bias: Batched[jax.Array]):
            return act(x + bias), None

        x, _ = jax.lax.scan(scan_func, x, layer.weight)
        return x

    def scan_func(x: jax.Array, weight_bias: Batched[jax.Array]):
        weight, bias = weight_bias[..., :-1], weight_bias[..., -1]
        return act(x @ weight + bias), None

    weight_bias = jnp.concatenate([layer.weight, layer.bias[:, :, None]], axis=-1)
    x, _ = jax.lax.scan(scan_func, x, weight_bias)
    return x


def _scan_batched_layer_with_multiple_activations(
    x: Batched[jax.Array],
    layer: Batched[Linear],
    act: Sequence[ActivationFunctionType],
) -> jax.Array:
    if layer.bias is None:

        def scan_func(x_index: tuple[jax.Array, int], weight: Batched[jax.Array]):
            x, index = x_index
            x = jax.lax.switch(index, act, x @ weight)
            return (x, index + 1), None

        (x, _), _ = jax.lax.scan(scan_func, (x, 0), layer.weight)
        return x

    def scan_func(x_index: jax.Array, weight_bias: Batched[jax.Array]):
        x, index = x_index
        weight, bias = weight_bias[..., :-1], weight_bias[..., -1]
        x = jax.lax.switch(index, act, x @ weight + bias)
        return [x, index + 1], None

    weight_bias = jnp.concatenate([layer.weight, layer.bias[:, :, None]], axis=-1)
    (x, _), _ = jax.lax.scan(scan_func, [x, 0], weight_bias)
    return x


class MLP(sk.TreeClass):
    """Multi-layer perceptron.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        key: Random number generator key.
        hidden_size: Number of hidden units in each hidden layer.
        num_hidden_layers: Number of hidden layers including the output layer.
        act: Activation function. Defaults to ``tanh``.
        weight_init: Weight initialization function. Defaults to ``glorot_uniform``.
        bias_init: Bias initialization function. Defaults to ``zeros``.
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> mlp = sk.nn.MLP(1, 2, hidden_size=4, num_hidden_layers=2, key=jr.PRNGKey(0))
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
        >>> import jax.random as jr
        >>> import serket as sk
        >>> import numpy.testing as npt
        >>> fnn = sk.nn.FNN([1] + [4] * 100 + [2], key=jr.PRNGKey(0))
        >>> mlp = sk.nn.MLP(1, 2, hidden_size=4, num_hidden_layers=100, key=jr.PRNGKey(0))
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
        >>> lazy_mlp = sk.nn.MLP(None, 1, num_hidden_layers=2, hidden_size=10, key=jr.PRNGKey(0))
        >>> _, materialized_mlp = lazy_mlp.at['__call__'](jnp.ones([1, 10]))
        >>> materialized_mlp.layers[0].in_features
        (10,)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: jr.KeyArray,
        hidden_size: int,
        num_hidden_layers: int,
        act: ActivationType | tuple[ActivationType, ...] = "tanh",
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        dtype: DType = jnp.float32,
    ):
        if hidden_size < 1:
            raise ValueError(f"`{hidden_size=}` must be positive.")

        keys = jr.split(key, num_hidden_layers + 1)

        if isinstance(act, tuple):
            if len(act) != (num_hidden_layers):
                raise ValueError(f"{len(act)=} != {(num_hidden_layers)=}")
            self.act = tuple(resolve_activation(act) for act in act)
        else:
            self.act = resolve_activation(act)

        kwargs = dict(weight_init=weight_init, bias_init=bias_init, dtype=dtype)

        @jax.vmap
        def batched_linear(key: jr.KeyArray) -> Batched[Linear]:
            return sk.tree_mask(Linear(hidden_size, hidden_size, key=key, **kwargs))

        self.layers: tuple[Linear, Batched[Linear], Linear] = (
            Linear(in_features, hidden_size, key=keys[0], **kwargs),
            sk.tree_unmask(batched_linear(keys[1:-1])),
            Linear(hidden_size, out_features, key=keys[-1], **kwargs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        l0, lm, lh = self.layers

        if isinstance(self.act, tuple):
            a0, *ah = self.act
            x = a0(l0(x))
            x = _scan_batched_layer_with_multiple_activations(x, lm, ah)
            return lh(x)

        a0 = self.act
        x = a0(l0(x))
        x = _scan_batched_layer_with_single_activation(x, lm, a0)
        return lh(x)
