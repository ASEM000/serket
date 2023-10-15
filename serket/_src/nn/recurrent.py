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

import abc
import functools as ft
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

import serket as sk
from serket._src.custom_transform import tree_state
from serket._src.nn.activation import ActivationType, resolve_activation
from serket._src.nn.convolution import (
    Conv1D,
    Conv2D,
    Conv3D,
    FFTConv1D,
    FFTConv2D,
    FFTConv3D,
)
from serket._src.nn.initialization import DType, InitType
from serket._src.utils import (
    DilationType,
    KernelSizeType,
    PaddingType,
    StridesType,
    maybe_lazy_call,
    maybe_lazy_init,
    positive_int_cb,
    validate_axis_shape,
    validate_spatial_nd,
)

"""Defines RNN related classes."""


def is_lazy_call(instance, *_, **__) -> bool:
    return instance.in_features is None


def is_lazy_init(_, in_features: int | None, *__, **___) -> bool:
    return in_features is None


def infer_in_features(_, x: jax.Array, *__, **___) -> int:
    return x.shape[0]


updates = dict(in_features=infer_in_features)


@sk.autoinit
class RNNState(sk.TreeClass):
    hidden_state: jax.Array


class RNNCell(sk.TreeClass):
    """Abstract class for RNN cells.

    Subclasses should
        - Implement `__call__` method that accept an input and a state and return a new state.
        - Define state rule using :func:`serket.tree_state` decorator.

    Subclassed classes can by used with `ScanRNN` to scan the RNN over a sequence
    of inputs. for example, check out `SimpleRNNCell`.
    """

    @abc.abstractmethod
    def __call__(self, x: jax.Array, state: RNNState) -> RNNState:
        ...

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        # 0 for non-spatial, 1 for 1D, 2 for 2D, 3 for 3D etc.
        ...


class SimpleRNNState(RNNState):
    ...


class SimpleRNNCell(RNNCell):
    """Vanilla RNN cell that defines the update rule for the hidden state

    Args:
        in_features: the number of input features
        hidden_features: the number of hidden features
        key: the key to use to initialize the weights
        weight_init: the function to use to initialize the weights
        bias_init: the function to use to initialize the bias
        recurrent_weight_init: the function to use to initialize the recurrent weights
        act: the activation function to use for the hidden state update
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> # 10-dimensional input, 20-dimensional hidden state
        >>> cell = sk.nn.SimpleRNNCell(10, 20, key=jr.PRNGKey(0))
        >>> # 20-dimensional hidden state
        >>> rnn_state = sk.tree_state(cell)
        >>> x = jnp.ones((10,)) # 10 features
        >>> result = cell(x, rnn_state)
        >>> result.hidden_state.shape  # 20 features
        (20,)

    Note:
        :class:`.SimpleRNNCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy_cell = sk.nn.SimpleRNNCell(None, 20, key=jr.PRNGKey(0))
        >>> lazy_layer = sk.nn.ScanRNN(lazy_cell, return_sequences=True)
        >>> x = jnp.ones((5, 10)) # 5 timesteps, 10 features
        >>> _, materialized_layer = lazy_layer.at["__call__"](x)
        >>> materialized_layer(x).shape
        (5, 20)

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNNCell.
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        key: jax.Array,
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        recurrent_weight_init: InitType = "orthogonal",
        act: ActivationType = jax.nn.tanh,
        dtype: DType = jnp.float32,
    ):
        k1, k2 = jr.split(key, 2)

        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act = resolve_activation(act)

        i2h = sk.nn.Linear(
            in_features,
            hidden_features,
            weight_init=weight_init,
            bias_init=bias_init,
            key=k1,
            dtype=dtype,
        )

        h2h = sk.nn.Linear(
            hidden_features,
            hidden_features,
            weight_init=recurrent_weight_init,
            bias_init=None,
            key=k2,
            dtype=dtype,
        )

        self.in_hidden_to_hidden_weight = jnp.concatenate([i2h.weight, h2h.weight])
        self.in_hidden_to_hidden_bias = i2h.bias

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, array: jax.Array, state: SimpleRNNState) -> SimpleRNNState:
        if not isinstance(state, SimpleRNNState):
            raise TypeError(f"Expected {state=} to be an instance of `SimpleRNNState`")

        ih = jnp.concatenate([array, state.hidden_state], axis=-1)
        h = ih @ self.in_hidden_to_hidden_weight + self.in_hidden_to_hidden_bias
        return SimpleRNNState(self.act(h))

    spatial_ndim: int = 0


class DenseState(RNNState):
    ...


class DenseCell(RNNCell):
    """No hidden state cell that applies a dense(Linear+activation) layer to the input

    Args:
        in_features: the number of input features
        hidden_features: the number of hidden features
        key: the key to use to initialize the weights
        weight_init: the function to use to initialize the weights
        bias_init: the function to use to initialize the bias
        act: the activation function to use for the hidden state update,
            use `None` for no activation
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> # 10-dimensional input, 20-dimensional hidden state
        >>> cell = sk.nn.DenseCell(10, 20, key=jr.PRNGKey(0))
        >>> # 20-dimensional hidden state
        >>> dummy_state = sk.tree_state(cell)
        >>> x = jnp.ones((10,)) # 10 features
        >>> result = cell(x, dummy_state)
        >>> result.hidden_state.shape  # 20 features
        (20,)

    Note:
        :class:`.DenseCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy_cell = sk.nn.DenseCell(None, 20, key=jr.PRNGKey(0))
        >>> lazy_layer = sk.nn.ScanRNN(lazy_cell, return_sequences=True)
        >>> x = jnp.ones((5, 10)) # 5 timesteps, 10 features
        >>> _, materialized_layer = lazy_layer.at["__call__"](x)
        >>> materialized_layer(x).shape
        (5, 20)
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        act: ActivationType = jax.nn.tanh,
        key: jax.Array,
        dtype: DType = jnp.float32,
    ):
        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act = resolve_activation(act)

        self.in_to_hidden = sk.nn.Linear(
            in_features,
            hidden_features,
            weight_init=weight_init,
            bias_init=bias_init,
            key=key,
            dtype=dtype,
        )

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, array: jax.Array, state: DenseState) -> DenseState:
        if not isinstance(state, DenseState):
            raise TypeError(f"Expected {state=} to be an instance of `DenseState`")

        h = self.act(self.in_to_hidden(array))
        return DenseState(h)

    spatial_ndim: int = 0


@sk.autoinit
class LSTMState(RNNState):
    cell_state: jax.Array


class LSTMCell(RNNCell):
    """LSTM cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: the number of input features
        hidden_features: the number of hidden features
        weight_init: the function to use to initialize the weights
        bias_init: the function to use to initialize the bias
        recurrent_weight_init: the function to use to initialize the recurrent weights
        act: the activation function to use for the hidden state update
        recurrent_act: the activation function to use for the cell state update
        key: the key to use to initialize the weights
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> # 10-dimensional input, 20-dimensional hidden state
        >>> cell = sk.nn.LSTMCell(10, 20, key=jr.PRNGKey(0))
        >>> # 20-dimensional hidden state
        >>> dummy_state = sk.tree_state(cell)
        >>> x = jnp.ones((10,)) # 10 features
        >>> result = cell(x, dummy_state)
        >>> result.hidden_state.shape  # 20 features
        (20,)

    Note:
        :class:`.LSTMCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy_cell = sk.nn.LSTMCell(None, 20, key=jr.PRNGKey(0))
        >>> lazy_layer = sk.nn.ScanRNN(lazy_cell, return_sequences=True)
        >>> x = jnp.ones((5, 10)) # 5 timesteps, 10 features
        >>> _, materialized_layer = lazy_layer.at["__call__"](x)
        >>> materialized_layer(x).shape
        (5, 20)

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTMCell
        - https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        key: jax.Array,
        weight_init: str | Callable = "glorot_uniform",
        bias_init: str | Callable | None = "zeros",
        recurrent_weight_init: str | Callable = "orthogonal",
        act: str | Callable[[Any], Any] | None = "tanh",
        recurrent_act: ActivationType | None = "sigmoid",
        dtype: DType = jnp.float32,
    ):
        k1, k2 = jr.split(key, 2)

        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act = resolve_activation(act)
        self.recurrent_act = resolve_activation(recurrent_act)

        i2h = sk.nn.Linear(
            in_features,
            hidden_features * 4,
            weight_init=weight_init,
            bias_init=bias_init,
            key=k1,
            dtype=dtype,
        )

        h2h = sk.nn.Linear(
            hidden_features,
            hidden_features * 4,
            weight_init=recurrent_weight_init,
            bias_init=None,
            key=k2,
            dtype=dtype,
        )

        self.in_hidden_to_hidden_weight = jnp.concatenate([i2h.weight, h2h.weight])
        self.in_hidden_to_hidden_bias = i2h.bias

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, array: jax.Array, state: LSTMState) -> LSTMState:
        if not isinstance(state, LSTMState):
            raise TypeError(f"Expected {state=} to be an instance of `LSTMState`")

        h, c = state.hidden_state, state.cell_state
        ih = jnp.concatenate([array, h], axis=-1)
        h = ih @ self.in_hidden_to_hidden_weight + self.in_hidden_to_hidden_bias
        i, f, g, o = jnp.split(h, 4, axis=-1)
        i = self.recurrent_act(i)
        f = self.recurrent_act(f)
        g = self.act(g)
        o = self.recurrent_act(o)
        c = f * c + i * g
        h = o * self.act(c)
        return LSTMState(h, c)

    spatial_ndim: int = 0


class GRUState(RNNState):
    ...


class GRUCell(RNNCell):
    """GRU cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: the number of input features
        hidden_features: the number of hidden features
        key: the key to use to initialize the weights
        weight_init: the function to use to initialize the weights
        bias_init: the function to use to initialize the bias
        recurrent_weight_init: the function to use to initialize the recurrent weights
        act: the activation function to use for the hidden state update
        recurrent_act: the activation function to use for the cell state update
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> # 10-dimensional input, 20-dimensional hidden state
        >>> cell = sk.nn.GRUCell(10, 20, key=jr.PRNGKey(0))
        >>> # 20-dimensional hidden state
        >>> dummy_state = sk.tree_state(cell)
        >>> x = jnp.ones((10,)) # 10 features
        >>> result = cell(x, dummy_state)
        >>> result.hidden_state.shape  # 20 features
        (20,)

    Note:
        :class:`.GRUCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy_cell = sk.nn.GRUCell(None, 20, key=jr.PRNGKey(0))
        >>> lazy_layer = sk.nn.ScanRNN(lazy_cell, return_sequences=True)
        >>> x = jnp.ones((5, 10)) # 5 timesteps, 10 features
        >>> _, materialized_layer = lazy_layer.at["__call__"](x)
        >>> materialized_layer(x).shape
        (5, 20)

    Reference:
        - https://keras.io/api/layers/recurrent_layers/gru/
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        key: jax.Array,
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        recurrent_weight_init: InitType = "orthogonal",
        act: ActivationType | None = "tanh",
        recurrent_act: ActivationType | None = "sigmoid",
        dtype: DType = jnp.float32,
    ):
        k1, k2 = jr.split(key, 2)

        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act = resolve_activation(act)
        self.recurrent_act = resolve_activation(recurrent_act)

        self.in_to_hidden = sk.nn.Linear(
            in_features,
            hidden_features * 3,
            weight_init=weight_init,
            bias_init=bias_init,
            key=k1,
            dtype=dtype,
        )

        self.hidden_to_hidden = sk.nn.Linear(
            hidden_features,
            hidden_features * 3,
            weight_init=recurrent_weight_init,
            bias_init=None,
            key=k2,
            dtype=dtype,
        )

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, array: jax.Array, state: GRUState) -> GRUState:
        if not isinstance(state, GRUState):
            raise TypeError(f"Expected {state=} to be an instance of `GRUState`")

        h = state.hidden_state
        xe, xu, xo = jnp.split(self.in_to_hidden(array), 3, axis=-1)
        he, hu, ho = jnp.split(self.hidden_to_hidden(h), 3, axis=-1)
        e = self.recurrent_act(xe + he)
        u = self.recurrent_act(xu + hu)
        o = self.act(xo + (e * ho))
        h = (1 - u) * o + u * h
        return GRUState(hidden_state=h)

    spatial_ndim: int = 0


@sk.autoinit
class ConvLSTMNDState(RNNState):
    cell_state: jax.Array


class ConvLSTMNDCell(RNNCell):
    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        kernel_size: KernelSizeType,
        *,
        key: jax.Array,
        strides: StridesType = 1,
        padding: PaddingType = "same",
        dilation: DilationType = 1,
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        recurrent_weight_init: InitType = "orthogonal",
        act: ActivationType | None = "tanh",
        recurrent_act: ActivationType | None = "hard_sigmoid",
        dtype: DType = jnp.float32,
    ):
        k1, k2 = jr.split(key, 2)

        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act = resolve_activation(act)
        self.recurrent_act = resolve_activation(recurrent_act)

        self.in_to_hidden = self.convolution_layer(
            in_features,
            hidden_features * 4,
            kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            weight_init=weight_init,
            bias_init=bias_init,
            key=k1,
            dtype=dtype,
        )

        self.hidden_to_hidden = self.convolution_layer(
            hidden_features,
            hidden_features * 4,
            kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            weight_init=recurrent_weight_init,
            bias_init=None,
            key=k2,
            dtype=dtype,
        )

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, array: jax.Array, state: ConvLSTMNDState) -> ConvLSTMNDState:
        if not isinstance(state, ConvLSTMNDState):
            raise TypeError(f"Expected {state=} to be an instance of ConvLSTMNDState.")

        h, c = state.hidden_state, state.cell_state
        h = self.in_to_hidden(array) + self.hidden_to_hidden(h)
        i, f, g, o = jnp.split(h, 4, axis=0)
        i = self.recurrent_act(i)
        f = self.recurrent_act(f)
        g = self.act(g)
        o = self.recurrent_act(o)
        c = f * c + i * g
        h = o * self.act(c)
        return ConvLSTMNDState(h, c)

    @property
    @abc.abstractmethod
    def convolution_layer(self):
        ...

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class ConvLSTM1DCell(ConvLSTMNDCell):
    """1D Convolution LSTM cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        key: PRNG key
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act: Activation function
        recurrent_act: Recurrent activation function
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.ConvLSTM1DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4)

    Note:
        :class:`.ConvLSTM1DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy_cell = sk.nn.ConvLSTM1DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(lazy_cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4)

    Reference:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D
    """

    spatial_ndim: int = 1
    convolution_layer = Conv1D


class FFTConvLSTM1DCell(ConvLSTMNDCell):
    """1D FFT Convolution LSTM cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        key: PRNG key
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act: Activation function
        recurrent_act: Recurrent activation function
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.FFTConvLSTM1DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4)

    Note:
        :class:`.FFTConvLSTM1DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.FFTConvLSTM1DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4)

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D
    """

    spatial_ndim: int = 1
    convolution_layer = FFTConv1D


class ConvLSTM2DCell(ConvLSTMNDCell):
    """2D Convolution LSTM cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        key: random key to use to initialize weights.
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act: Activation function
        recurrent_act: Recurrent activation function
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.ConvLSTM2DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4, 4)

    Note:
        :class:`.ConvLSTM2DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.ConvLSTM2DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4, 4)

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM2D
    """

    spatial_ndim: int = 2
    convolution_layer = Conv2D


class FFTConvLSTM2DCell(ConvLSTMNDCell):
    """2D FFT Convolution LSTM cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        key: random key to initialize weights.
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act: Activation function
        recurrent_act: Recurrent activation function
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.FFTConvLSTM2DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4, 4)

    Note:
        :class:`.FFTConvLSTM2DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.FFTConvLSTM2DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4, 4)

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM2D
    """

    spatial_ndim: int = 2
    convolution_layer = FFTConv2D


class ConvLSTM3DCell(ConvLSTMNDCell):
    """3D Convolution LSTM cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        key: random key to initialize weights.
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act: Activation function
        recurrent_act: Recurrent activation function
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.ConvLSTM3DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4, 4, 4)

    Note:
        :class:`.ConvLSTM3DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.ConvLSTM3DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4, 4, 4)

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM3D
    """

    spatial_ndim: int = 3
    convolution_layer = Conv3D


class FFTConvLSTM3DCell(ConvLSTMNDCell):
    """3D FFT Convolution LSTM cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        key: random key to initialize weights.
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act: Activation function
        recurrent_act: Recurrent activation function
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.FFTConvLSTM3DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4, 4, 4)

    Note:
        :class:`.FFTConvLSTM3DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.FFTConvLSTM3DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4, 4, 4)

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM3D
    """

    spatial_ndim: int = 3
    convolution_layer = FFTConv3D


class ConvGRUNDState(RNNState):
    ...


class ConvGRUNDCell(RNNCell):
    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        key: jax.Array,
        strides: StridesType = 1,
        padding: PaddingType = "same",
        dilation: DilationType = 1,
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        recurrent_weight_init: InitType = "orthogonal",
        act: ActivationType | None = "tanh",
        recurrent_act: ActivationType | None = "sigmoid",
        dtype: DType = jnp.float32,
    ):
        k1, k2 = jr.split(key, 2)

        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act = resolve_activation(act)
        self.recurrent_act = resolve_activation(recurrent_act)

        self.in_to_hidden = self.convolution_layer(
            in_features,
            hidden_features * 3,
            kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            weight_init=weight_init,
            bias_init=bias_init,
            key=k1,
            dtype=dtype,
        )

        self.hidden_to_hidden = self.convolution_layer(
            hidden_features,
            hidden_features * 3,
            kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            weight_init=recurrent_weight_init,
            bias_init=None,
            key=k2,
            dtype=dtype,
        )

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, array: jax.Array, state: ConvGRUNDState) -> ConvGRUNDState:
        if not isinstance(state, ConvGRUNDState):
            raise TypeError(f"Expected {state=} to be an instance of `GRUState`")

        h = state.hidden_state
        xe, xu, xo = jnp.split(self.in_to_hidden(array), 3, axis=0)
        he, hu, ho = jnp.split(self.hidden_to_hidden(h), 3, axis=0)
        e = self.recurrent_act(xe + he)
        u = self.recurrent_act(xu + hu)
        o = self.act(xo + (e * ho))
        h = (1 - u) * o + u * h
        return ConvGRUNDState(hidden_state=h)

    @property
    @abc.abstractmethod
    def convolution_layer(self):
        ...

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class ConvGRU1DCell(ConvGRUNDCell):
    """1D Convolution GRU cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        key: random key to initialize weights.
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act: Activation function
        recurrent_act: Recurrent activation function
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.ConvGRU1DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4)

    Note:
        :class:`.ConvGRU1DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.ConvGRU1DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4)
    """

    spatial_ndim: int = 1
    convolution_layer = Conv1D


class FFTConvGRU1DCell(ConvGRUNDCell):
    """1D FFT Convolution GRU cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        key: random key to initialize weights.
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act: Activation function
        recurrent_act: Recurrent activation function
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.FFTConvGRU1DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4)

    Note:
        :class:`.FFTConvGRU1DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy_cell = sk.nn.FFTConvGRU1DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4)
    """

    spatial_ndim: int = 1
    convolution_layer = FFTConv1D


class ConvGRU2DCell(ConvGRUNDCell):
    """2D Convolution GRU cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        key: random key to initialize weights.
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act: Activation function
        recurrent_act: Recurrent activation function
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.ConvGRU2DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4, 4)

    Note:
        :class:`.ConvGRU2DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy_cell = sk.nn.ConvGRU2DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(lazy_cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4, 4)
    """

    spatial_ndim: int = 2
    convolution_layer = Conv2D


class FFTConvGRU2DCell(ConvGRUNDCell):
    """2D FFT Convolution GRU cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        key: random key to initialize weights.
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act: Activation function
        recurrent_act: Recurrent activation function
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.FFTConvGRU2DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4, 4)

    Note:
        :class:`.FFTConvGRU2DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy_cell = sk.nn.FFTConvGRU2DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(lazy_cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4, 4)
    """

    spatial_ndim: int = 2
    convolution_layer = FFTConv2D


class ConvGRU3DCell(ConvGRUNDCell):
    """3D Convolution GRU cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        key: random key to initialize weights.
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act: Activation function
        recurrent_act: Recurrent activation function
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy_cell = sk.nn.ConvGRU3DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(lazy_cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4, 4, 4)

    Note:
        :class:`.ConvGRU3DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy_cell = sk.nn.ConvGRU3DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(lazy_cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4, 4, 4)
    """

    spatial_ndim: int = 3
    convolution_layer = Conv3D


class FFTConvGRU3DCell(ConvGRUNDCell):
    """3D Convolution GRU cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        key: random key to initialize weights.
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act: Activation function
        recurrent_act: Recurrent activation function
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.FFTConvGRU3DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4, 4, 4)

    Note:
        :class:`.FFTConvGRU3DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy_cell = sk.nn.FFTConvGRU3DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> x = jnp.ones((1, 10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(lazy_cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4, 4, 4)
    """

    spatial_ndim: int = 3
    convolution_layer = FFTConv3D


# Scanning API


def materialize_cell(instance, x: jax.Array, state=None, **__) -> RNNCell:
    # in case of lazy initialization, we need to materialize the cell
    # before it can be passed to the scan function
    cell = instance.cell
    state = state if state is not None else sk.tree_state(instance, array=x)
    state = split_state(state, 2) if instance.backward_cell is not None else [state]
    _, cell = cell.at["__call__"](x[0], state[0])
    return cell


def materialize_backward_cell(instance, x, state=None, **__) -> RNNCell | None:
    if instance.backward_cell is None:
        return None
    cell = instance.cell
    state = state if state is not None else sk.tree_state(instance, array=x)
    state = split_state(state, 2) if instance.backward_cell is not None else [state]
    _, cell = cell.at["__call__"](x[0], state[-1])
    return cell


def is_lazy_init(_, cell, backward_cell=None, **__) -> bool:
    lhs = cell.in_features is None
    rhs = getattr(backward_cell, "in_features", False) is None
    return lhs or rhs


def is_lazy_call(instance, x, state=None, **_) -> bool:
    lhs = instance.cell.in_features is None
    rhs = getattr(instance.backward_cell, "in_features", False) is None
    return lhs or rhs


updates = dict(cell=materialize_cell, backward_cell=materialize_backward_cell)


def split_state(state: RNNState, splits: int) -> list[RNNState]:
    flat_arrays: list[jax.Array] = jtu.tree_leaves(state)
    return [type(state)(*x) for x in zip(*(jnp.split(x, splits) for x in flat_arrays))]


def concat_state(states: list[RNNState]) -> RNNState:
    # undo the split
    return (
        states[0]
        if len(states) == 1
        else jax.tree_map(lambda *x: jnp.concatenate([*x]), *states)
    )


def accumulate_scan(
    cell: RNNCell,
    array: jax.Array,
    state: RNNState,
    reverse: bool = False,
) -> tuple[jax.Array, RNNState]:
    """Scans a RNN cell over a sequence. Accumulates the output for each timestep."""

    def scan_func(carry, array):
        state = cell(array, state=carry)
        return state, state

    array = jnp.flip(array, axis=0) if reverse else array  # flip over time axis
    result = jax.lax.scan(scan_func, state, array)[1].hidden_state
    carry, result = jax.lax.scan(scan_func, state, array)
    result = result.hidden_state
    result = jnp.flip(result, axis=-1) if reverse else result
    return result, carry


def unaccumulate_scan(
    cell: RNNCell,
    array: jax.Array,
    state: RNNState,
    reverse: bool = False,
) -> jax.Array:
    """Scans a RNN cell over a sequence. Returns the output for the last timestep."""

    def scan_func(carry, x):
        state = cell(x, state=carry)
        return state, None

    array = jnp.flip(array, axis=0) if reverse else array
    carry, _ = jax.lax.scan(scan_func, state, array)
    result = carry.hidden_state
    return result, carry


def scan_unidirectional_rnn(
    cell: RNNCell,
    backward_cell: None,
    array: jax.Array,
    state: RNNState,
    return_sequences: bool,
    return_state: bool,
) -> jax.Array | tuple[jax.Array, RNNState]:
    """Scans a unidirectional RNN cell over a sequence.

    Args:
        cell: the RNN cell to scan.
        backward_cell: placeholder for consistency with bidirectional RNN. ignored.
        array: the input sequence. shape: ``[time, features]``.
        state: the initial state of the RNN cell.
        return_sequences: whether to return the output for each timestep.
        return_state: whether to return the final state of the RNN cell(s).
    """
    del backward_cell  # for consistency with bidirectional RNN
    scan_func = accumulate_scan if return_sequences else unaccumulate_scan
    result, state = scan_func(cell, array, state)
    return (result, state) if return_state else result


def scan_bidirectional_rnn(
    cell: RNNCell,
    backward_cell: RNNCell,
    array: jax.Array,
    state: RNNState,
    return_sequences: bool,
    return_state: bool,
) -> jax.Array | tuple[jax.Array, RNNState]:
    """Scans a bidirectional RNN cell over a sequence.

    Args:
        cell: the forward RNN cell to scan.
        backward_cell: the backward RNN cell to scan.
        array: the input sequence. shape: ``[time, features]``.
        state: the initial state of the RNN cell.
        return_sequences: whether to return the output for each timestep.
        return_state: whether to return the final state of the RNN cell(s).
    """
    lhs_state, rhs_state = split_state(state, splits=2)
    scan_func = accumulate_scan if return_sequences else unaccumulate_scan
    lhs_result, lhs_state = scan_func(cell, array, lhs_state, False)
    rhs_result, rhs_state = scan_func(backward_cell, array, rhs_state, True)
    concat_axis = int(return_sequences)
    result = jnp.concatenate((lhs_result, rhs_result), axis=concat_axis)
    state: RNNState = concat_state((lhs_state, rhs_state))
    return (result, state) if return_state else result


def check_cells(*cells: Any) -> None:
    """Checks that the cells are compatible with each other."""
    cell0, *cells = cells
    for cell in cells:
        if not isinstance(cell, RNNCell):
            raise TypeError(f"{cell=} to be an instance of `RNNCell`.")
        if cell0.in_features != cell.in_features:
            raise ValueError(f"{cell0.in_features=} != {cell.in_features=}")
        if cell0.hidden_features != cell.hidden_features:
            raise ValueError(f"{cell0.hidden_features=} != {cell.hidden_features=}")


class ScanRNN(sk.TreeClass):
    """Scans RNN cell over a sequence.

    Args:
        cell: the RNN cell to scan.
        backward_cell: (optional) the backward RNN cell to scan in case of bidirectional RNN.
        return_sequences: whether to return the output for each timestep.
        return_state: whether to return the final state of the RNN cell(s).

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> # 10-dimensional input, 20-dimensional hidden state
        >>> cell = sk.nn.SimpleRNNCell(10, 20, key=jr.PRNGKey(0))
        >>> rnn = sk.nn.ScanRNN(cell, return_state=True)
        >>> x = jnp.ones((5, 10)) # 5 timesteps, 10 features
        >>> result, state = rnn(x)  # 20 features
        >>> print(result.shape)
        (20,)

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> cell = sk.nn.SimpleRNNCell(10, 20, key=jr.PRNGKey(0))
        >>> rnn = sk.nn.ScanRNN(cell, return_sequences=True, return_state=True)
        >>> x = jnp.ones((5, 10)) # 5 timesteps, 10 features
        >>> result, state = rnn(x)  # 5 timesteps, 20 features
        >>> print(result.shape)
        (5, 20)
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        cell: RNNCell,
        backward_cell: RNNCell | None = None,
        *,
        return_sequences: bool = False,
        return_state: bool = False,
    ):
        if not isinstance(cell, RNNCell):
            raise TypeError(f"Expected {cell=} to be an instance of `RNNCell`.")

        if backward_cell is not None:
            check_cells(cell, backward_cell)

        self.cell = cell
        self.backward_cell = backward_cell
        self.return_sequences = return_sequences
        self.return_state = return_state

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    def __call__(
        self,
        array: jax.Array,
        state: RNNState | None = None,
    ) -> jax.Array | tuple[jax.Array, RNNState]:
        """Scans the RNN cell over a sequence.

        Args:
            array: the input sequence.
            state: the initial state. if None, state is initialized by the rule
                defined using :func:`.tree_state`.

        Returns:
            return the result and state if ``return_state`` is True. otherwise,
            return only the result.
        """

        if not isinstance(state, (RNNState, type(None))):
            raise TypeError(f"Expected state to be an instance of RNNState, {state=}")

        if array.ndim != self.cell.spatial_ndim + 2:
            raise ValueError(
                f"Expected x to have {(self.cell.spatial_ndim + 2)=} dimensions corresponds to "
                f"(timesteps, in_features, {','.join('...'*self.cell.spatial_ndim)}),"
                f" got {array.ndim=}"
            )

        if self.cell.in_features != array.shape[1]:
            raise ValueError(
                f"Expected x to have shape (timesteps, {self.cell.in_features},"
                f"{'*'*self.cell.spatial_ndim}), got {array.shape=}"
            )

        args = (
            self.cell,
            self.backward_cell,
            array,
            tree_state(self, array=array) if state is None else state,
            self.return_sequences,
            self.return_state,
        )
        return (
            scan_unidirectional_rnn(*args)
            if self.backward_cell is None
            else scan_bidirectional_rnn(*args)
        )


# register state handlers


@tree_state.def_state(SimpleRNNCell)
def _(cell: SimpleRNNCell, **_) -> SimpleRNNState:
    return SimpleRNNState(jnp.zeros([cell.hidden_features]))


@tree_state.def_state(DenseCell)
def _(cell: DenseCell, **_) -> DenseState:
    return DenseState(jnp.empty([cell.hidden_features]))


@tree_state.def_state(LSTMCell)
def _(cell: LSTMCell, **_) -> LSTMState:
    shape = [cell.hidden_features]
    return LSTMState(jnp.zeros(shape), jnp.zeros(shape))


@tree_state.def_state(GRUCell)
def _(cell: GRUCell, **_) -> GRUState:
    return GRUState(jnp.zeros([cell.hidden_features]))


def _check_rnn_cell_tree_state_input(cell: RNNCell, array):
    if not (hasattr(array, "ndim") and hasattr(array, "shape")):
        raise TypeError(
            f"Expected {array=} to have `ndim` and `shape` attributes."
            f"To initialize the `{type(cell).__name__}` state.\n"
            "Pass a single sample array to `tree_state(..., array=)`."
        )

    if array.ndim != cell.spatial_ndim + 1:
        raise ValueError(
            f"{array.ndim=} != {(cell.spatial_ndim + 1)=}."
            f"Expected input to have `shape` (in_features, {'...'*cell.spatial_dim})."
            "Pass a single sample array to `tree_state"
        )

    if len(spatial_dim := array.shape[1:]) != cell.spatial_ndim:
        raise ValueError(f"{len(spatial_dim)=} != {cell.spatial_ndim=}.")

    return array


@tree_state.def_state(ConvLSTMNDCell)
def _(cell: ConvLSTMNDCell, array, **_) -> ConvLSTMNDState:
    array = _check_rnn_cell_tree_state_input(cell, array)
    shape = (cell.hidden_features, *array.shape[1:])
    zeros = jnp.zeros(shape).astype(array.dtype)
    return ConvLSTMNDState(zeros, zeros)


@tree_state.def_state(ConvGRUNDCell)
def _(cell: ConvGRUNDCell, *, array: Any, **_) -> ConvGRUNDState:
    array = _check_rnn_cell_tree_state_input(cell, array)
    shape = (cell.hidden_features, *array.shape[1:])
    return ConvGRUNDState(jnp.zeros(shape).astype(array.dtype))


@tree_state.def_state(ScanRNN)
def _(rnn: ScanRNN, array: jax.Array | None = None, **_) -> RNNState:
    # the idea here is to combine the state of the forward and backward cells
    # if backward cell exists. to have single state input for `ScanRNN` and
    # single state output not to complicate the ``__call__`` signature on the
    # user side.
    array = [None] if array is None else array
    # non-spatial cells don't need an input instead
    # pass `None` to `tree_state`
    # otherwise pass the a single time step input to the cells
    return (
        tree_state(rnn.cell, array=array[0])
        if rnn.backward_cell is None
        else concat_state(tree_state((rnn.cell, rnn.backward_cell), array=array[0]))
    )
