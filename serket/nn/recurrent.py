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
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

import serket as sk
from serket.nn.activation import ActivationType, resolve_activation
from serket.nn.custom_transform import tree_state
from serket.nn.initialization import DType, InitType
from serket.nn.utils import (
    DilationType,
    KernelSizeType,
    PaddingType,
    StridesType,
    maybe_lazy_call,
    maybe_lazy_init,
    positive_int_cb,
    validate_axis_shape,
    validate_spatial_ndim,
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

    Subclasses must implement:
        - `__call__` should take in an `input` and a `state` and return a new `state`.
        - `init_state` should take in a `spatial_shape` and return an initial `state`.
        - `spatial_ndim` should return the spatial dimensionality of the RNN.
            0 for non-spatial, 1 for 1D, 2 for 2D, 3 for 3D etc.

    Subclassed classes can by used with `ScanRNN` to scan the RNN over a sequence
    of inputs. for example, check out `SimpleRNNCell`.
    """

    @abc.abstractclassmethod
    def __call__(self, x: jax.Array, state: RNNState, **k) -> RNNState:
        ...

    @property
    @abc.abstractclassmethod
    def spatial_ndim(self) -> int:
        # 0 for non-spatial, 1 for 1D, 2 for 2D, 3 for 3D etc.
        ...


class SimpleRNNState(RNNState):
    ...


class SimpleRNNCell(RNNCell):
    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        recurrent_weight_init: InitType = "orthogonal",
        act_func: ActivationType = jax.nn.tanh,
        key: jr.KeyArray = jr.PRNGKey(0),
        dtype: DType = jnp.float32,
    ):
        """Vanilla RNN cell that defines the update rule for the hidden state

        Args:
            in_features: the number of input features
            hidden_features: the number of hidden features
            weight_init: the function to use to initialize the weights
            bias_init: the function to use to initialize the bias
            recurrent_weight_init: the function to use to initialize the recurrent weights
            act_func: the activation function to use for the hidden state update
            key: the key to use to initialize the weights
            dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

        Example:
            >>> import serket as sk
            >>> import jax.numpy as jnp
            >>> cell = sk.nn.SimpleRNNCell(10, 20) # 10-dimensional input, 20-dimensional hidden state
            >>> rnn_state = sk.tree_state(cell)  # 20-dimensional hidden state
            >>> x = jnp.ones((10,)) # 10 features
            >>> result = cell(x, rnn_state)
            >>> result.hidden_state.shape  # 20 features
            (20,)

        Note:
            :class:`.SimpleRNNCell` supports lazy initialization, meaning that the
            weights and biases are not initialized until the first call to the layer.
            This is useful when the input shape is not known at initialization time.

            To use lazy initialization, pass ``None`` as the ``in_features`` argument
            and use the ``.at["calling_method_name"]`` attribute to call the layer
            with an input of known shape.

            >>> import serket as sk
            >>> import jax.numpy as jnp
            >>> lazy_layer = sk.nn.ScanRNN(sk.nn.SimpleRNNCell(None, 20), return_sequences=True)
            >>> x = jnp.ones((5, 10)) # 5 timesteps, 10 features
            >>> _, materialized_layer = lazy_layer.at["__call__"](x)
            >>> materialized_layer(x).shape
            (5, 20)

        Reference:
            - https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNNCell.
        """
        k1, k2 = jr.split(key, 2)

        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act_func = resolve_activation(act_func)

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
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, state: SimpleRNNState, **k) -> SimpleRNNState:
        if not isinstance(state, SimpleRNNState):
            raise TypeError(f"Expected {state=} to be an instance of `SimpleRNNState`")

        ih = jnp.concatenate([x, state.hidden_state], axis=-1)
        h = ih @ self.in_hidden_to_hidden_weight + self.in_hidden_to_hidden_bias
        return SimpleRNNState(self.act_func(h))

    @property
    def spatial_ndim(self) -> int:
        return 0


class DenseState(RNNState):
    ...


class DenseCell(RNNCell):
    """No hidden state cell that applies a dense(Linear+activation) layer to the input

    Args:
        in_features: the number of input features
        hidden_features: the number of hidden features
        weight_init: the function to use to initialize the weights
        bias_init: the function to use to initialize the bias
        act_func: the activation function to use for the hidden state update,
            use `None` for no activation
        key: the key to use to initialize the weights
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> # 10-dimensional input, 20-dimensional hidden state
        >>> cell = sk.nn.DenseCell(10, 20)
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
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> lazy_layer = sk.nn.ScanRNN(sk.nn.DenseCell(None, 20), return_sequences=True)
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
        act_func: ActivationType = jax.nn.tanh,
        key: jr.KeyArray = jr.PRNGKey(0),
        dtype: DType = jnp.float32,
    ):
        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act_func = resolve_activation(act_func)

        self.in_to_hidden = sk.nn.Linear(
            in_features,
            hidden_features,
            weight_init=weight_init,
            bias_init=bias_init,
            key=key,
            dtype=dtype,
        )

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, state: DenseState, **k) -> DenseState:
        if not isinstance(state, DenseState):
            raise TypeError(f"Expected {state=} to be an instance of `DenseState`")

        h = self.act_func(self.in_to_hidden(x))
        return DenseState(h)

    @property
    def spatial_ndim(self) -> int:
        return 0


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
        act_func: the activation function to use for the hidden state update
        recurrent_act_func: the activation function to use for the cell state update
        key: the key to use to initialize the weights
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> # 10-dimensional input, 20-dimensional hidden state
        >>> cell = sk.nn.LSTMCell(10, 20)
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
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> lazy_layer = sk.nn.ScanRNN(sk.nn.LSTMCell(None, 20), return_sequences=True)
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
        weight_init: str | Callable = "glorot_uniform",
        bias_init: str | Callable | None = "zeros",
        recurrent_weight_init: str | Callable = "orthogonal",
        act_func: str | Callable[[Any], Any] | None = "tanh",
        recurrent_act_func: ActivationType | None = "sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
        dtype: DType = jnp.float32,
    ):
        k1, k2 = jr.split(key, 2)

        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act_func = resolve_activation(act_func)
        self.recurrent_act_func = resolve_activation(recurrent_act_func)

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
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, state: LSTMState, **k) -> LSTMState:
        if not isinstance(state, LSTMState):
            raise TypeError(f"Expected {state=} to be an instance of `LSTMState`")

        h, c = state.hidden_state, state.cell_state
        ih = jnp.concatenate([x, h], axis=-1)
        h = ih @ self.in_hidden_to_hidden_weight + self.in_hidden_to_hidden_bias
        i, f, g, o = jnp.split(h, 4, axis=-1)
        i = self.recurrent_act_func(i)
        f = self.recurrent_act_func(f)
        g = self.act_func(g)
        o = self.recurrent_act_func(o)
        c = f * c + i * g
        h = o * self.act_func(c)
        return LSTMState(h, c)

    @property
    def spatial_ndim(self) -> int:
        return 0


class GRUState(RNNState):
    ...


class GRUCell(RNNCell):
    """GRU cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: the number of input features
        hidden_features: the number of hidden features
        weight_init: the function to use to initialize the weights
        bias_init: the function to use to initialize the bias
        recurrent_weight_init: the function to use to initialize the recurrent weights
        act_func: the activation function to use for the hidden state update
        recurrent_act_func: the activation function to use for the cell state update
        key: the key to use to initialize the weights
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> # 10-dimensional input, 20-dimensional hidden state
        >>> cell = sk.nn.GRUCell(10, 20)
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
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> lazy_layer = sk.nn.ScanRNN(sk.nn.GRUCell(None, 20), return_sequences=True)
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
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        recurrent_weight_init: InitType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
        dtype: DType = jnp.float32,
    ):
        k1, k2 = jr.split(key, 2)

        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act_func = resolve_activation(act_func)
        self.recurrent_act_func = resolve_activation(recurrent_act_func)

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
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, state: GRUState, **k) -> GRUState:
        if not isinstance(state, GRUState):
            raise TypeError(f"Expected {state=} to be an instance of `GRUState`")

        h = state.hidden_state
        xe, xu, xo = jnp.split(self.in_to_hidden(x), 3, axis=-1)
        he, hu, ho = jnp.split(self.hidden_to_hidden(h), 3, axis=-1)
        e = self.recurrent_act_func(xe + he)
        u = self.recurrent_act_func(xu + hu)
        o = self.act_func(xo + (e * ho))
        h = (1 - u) * o + u * h
        return GRUState(hidden_state=h)

    @property
    def spatial_ndim(self) -> int:
        return 0


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
        strides: StridesType = 1,
        padding: PaddingType = "same",
        dilation: DilationType = 1,
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        recurrent_weight_init: InitType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "hard_sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
        dtype: DType = jnp.float32,
    ):
        k1, k2 = jr.split(key, 2)

        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act_func = resolve_activation(act_func)
        self.recurrent_act_func = resolve_activation(recurrent_act_func)

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
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, state: ConvLSTMNDState, **k) -> ConvLSTMNDState:
        if not isinstance(state, ConvLSTMNDState):
            raise TypeError(f"Expected {state=} to be an instance of ConvLSTMNDState.")

        h, c = state.hidden_state, state.cell_state
        h = self.in_to_hidden(x) + self.hidden_to_hidden(h)
        i, f, g, o = jnp.split(h, 4, axis=0)
        i = self.recurrent_act_func(i)
        f = self.recurrent_act_func(f)
        g = self.act_func(g)
        o = self.recurrent_act_func(o)
        c = f * c + i * g
        h = o * self.act_func(c)
        return ConvLSTMNDState(h, c)

    @property
    @abc.abstractmethod
    def convolution_layer(self):
        ...


class ConvLSTM1DCell(ConvLSTMNDCell):
    """1D Convolution LSTM cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.ConvLSTM1DCell(10, 2, 3)
        >>> x = jnp.ones((1, 10, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4)

    Note:
        :class:`.ConvLSTM1DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.ConvLSTM1DCell(None, 2, 3)
        >>> x = jnp.ones((1, 10, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4)

    Reference:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D
    """

    @property
    def spatial_ndim(self) -> int:
        return 1

    @property
    def convolution_layer(self):
        return sk.nn.Conv1D


class FFTConvLSTM1DCell(ConvLSTMNDCell):
    """1D FFT Convolution LSTM cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.FFTConvLSTM1DCell(10, 2, 3)
        >>> x = jnp.ones((1, 10, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4)

    Note:
        :class:`.FFTConvLSTM1DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.FFTConvLSTM1DCell(None, 2, 3)
        >>> x = jnp.ones((1, 10, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4)

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D
    """

    @property
    def spatial_ndim(self) -> int:
        return 1

    @property
    def convolution_layer(self):
        return sk.nn.FFTConv1D


class ConvLSTM2DCell(ConvLSTMNDCell):
    """2D Convolution LSTM cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.ConvLSTM2DCell(10, 2, 3)
        >>> x = jnp.ones((1, 10, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4, 4)

    Note:
        :class:`.ConvLSTM2DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.ConvLSTM2DCell(None, 2, 3)
        >>> x = jnp.ones((1, 10, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4, 4)

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM2D
    """

    @property
    def spatial_ndim(self) -> int:
        return 2

    @property
    def convolution_layer(self):
        return sk.nn.Conv2D


class FFTConvLSTM2DCell(ConvLSTMNDCell):
    """2D FFT Convolution LSTM cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.FFTConvLSTM2DCell(10, 2, 3)
        >>> x = jnp.ones((1, 10, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4, 4)

    Note:
        :class:`.FFTConvLSTM2DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.FFTConvLSTM2DCell(None, 2, 3)
        >>> x = jnp.ones((1, 10, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4, 4)

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM2D
    """

    @property
    def spatial_ndim(self) -> int:
        return 2

    @property
    def convolution_layer(self):
        return sk.nn.FFTConv2D


class ConvLSTM3DCell(ConvLSTMNDCell):
    """3D Convolution LSTM cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.ConvLSTM3DCell(10, 2, 3)
        >>> x = jnp.ones((1, 10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4, 4, 4)

    Note:
        :class:`.ConvLSTM3DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.ConvLSTM3DCell(None, 2, 3)
        >>> x = jnp.ones((1, 10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4, 4, 4)

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM3D
    """

    @property
    def spatial_ndim(self) -> int:
        return 3

    @property
    def convolution_layer(self):
        return sk.nn.Conv3D


class FFTConvLSTM3DCell(ConvLSTMNDCell):
    """3D FFT Convolution LSTM cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.FFTConvLSTM3DCell(10, 2, 3)
        >>> x = jnp.ones((1, 10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4, 4, 4)

    Note:
        :class:`.FFTConvLSTM3DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.FFTConvLSTM3DCell(None, 2, 3)
        >>> x = jnp.ones((1, 10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4, 4, 4)

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM3D
    """

    @property
    def spatial_ndim(self) -> int:
        return 3

    @property
    def convolution_layer(self):
        return sk.nn.FFTConv3D


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
        strides: StridesType = 1,
        padding: PaddingType = "same",
        dilation: DilationType = 1,
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        recurrent_weight_init: InitType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
        dtype: DType = jnp.float32,
    ):
        k1, k2 = jr.split(key, 2)

        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act_func = resolve_activation(act_func)
        self.recurrent_act_func = resolve_activation(recurrent_act_func)

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
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, state: ConvGRUNDState, **k) -> ConvGRUNDState:
        if not isinstance(state, ConvGRUNDState):
            raise TypeError(f"Expected {state=} to be an instance of `GRUState`")

        h = state.hidden_state
        xe, xu, xo = jnp.split(self.in_to_hidden(x), 3, axis=0)
        he, hu, ho = jnp.split(self.hidden_to_hidden(h), 3, axis=0)
        e = self.recurrent_act_func(xe + he)
        u = self.recurrent_act_func(xu + hu)
        o = self.act_func(xo + (e * ho))
        h = (1 - u) * o + u * h
        return ConvGRUNDState(hidden_state=h)

    @property
    @abc.abstractmethod
    def convolution_layer(self):
        ...


class ConvGRU1DCell(ConvGRUNDCell):
    """1D Convolution GRU cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.ConvGRU1DCell(10, 2, 3)
        >>> x = jnp.ones((1, 10, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4)

    Note:
        :class:`.ConvGRU1DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.ConvGRU1DCell(None, 2, 3)
        >>> x = jnp.ones((1, 10, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4)
    """

    @property
    def spatial_ndim(self) -> int:
        return 1

    @property
    def convolution_layer(self):
        return sk.nn.Conv1D


class FFTConvGRU1DCell(ConvGRUNDCell):
    """1D FFT Convolution GRU cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.FFTConvGRU1DCell(10, 2, 3)
        >>> x = jnp.ones((1, 10, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4)

    Note:
        :class:`.FFTConvGRU1DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.FFTConvGRU1DCell(None, 2, 3)
        >>> x = jnp.ones((1, 10, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4)
    """

    @property
    def spatial_ndim(self) -> int:
        return 1

    @property
    def convolution_layer(self):
        return sk.nn.FFTConv1D


class ConvGRU2DCell(ConvGRUNDCell):
    """2D Convolution GRU cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.ConvGRU2DCell(10, 2, 3)
        >>> x = jnp.ones((1, 10, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4, 4)

    Note:
        :class:`.ConvGRU2DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.ConvGRU2DCell(None, 2, 3)
        >>> x = jnp.ones((1, 10, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4, 4)
    """

    @property
    def spatial_ndim(self) -> int:
        return 2

    @property
    def convolution_layer(self):
        return sk.nn.Conv2D


class FFTConvGRU2DCell(ConvGRUNDCell):
    """2D FFT Convolution GRU cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.FFTConvGRU2DCell(10, 2, 3)
        >>> x = jnp.ones((1, 10, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4, 4)

    Note:
        :class:`.FFTConvGRU2DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.FFTConvGRU2DCell(None, 2, 3)
        >>> x = jnp.ones((1, 10, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4, 4)
    """

    @property
    def spatial_ndim(self) -> int:
        return 2

    @property
    def convolution_layer(self):
        return sk.nn.FFTConv2D


class ConvGRU3DCell(ConvGRUNDCell):
    """3D Convolution GRU cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.ConvGRU3DCell(10, 2, 3)
        >>> x = jnp.ones((1, 10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4, 4, 4)

    Note:
        :class:`.ConvGRU3DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.ConvGRU3DCell(None, 2, 3)
        >>> x = jnp.ones((1, 10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4, 4, 4)
    """

    @property
    def spatial_ndim(self) -> int:
        return 3

    @property
    def convolution_layer(self):
        return sk.nn.Conv3D


class FFTConvGRU3DCell(ConvGRUNDCell):
    """3D Convolution GRU cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        dilation: Dilation of the convolutional kernel
        weight_init: Weight initialization function
        bias_init: Bias initialization function
        recurrent_weight_init: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.FFTConvGRU3DCell(10, 2, 3)
        >>> x = jnp.ones((1, 10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> layer(x).shape
        (1, 2, 4, 4, 4)

    Note:
        :class:`.FFTConvGRU3DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> cell = sk.nn.FFTConvGRU3DCell(None, 2, 3)
        >>> x = jnp.ones((1, 10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> layer = sk.nn.ScanRNN(cell, return_sequences=True)  # return time steps results
        >>> _, layer = layer.at["__call__"](x)  # materialize the layer
        >>> layer(x).shape
        (1, 2, 4, 4, 4)
    """

    @property
    def spatial_ndim(self) -> int:
        return 3

    @property
    def convolution_layer(self):
        return sk.nn.FFTConv3D


# Scanning API


def materialize_cell(instance, x: jax.Array, state=None, **__) -> RNNCell:
    # in case of lazy initialization, we need to materialize the cell
    # before it can be passed to the scan function
    cell = instance.cell
    state = state if state is not None else sk.tree_state(instance, x)
    state = split_state(state, 2) if instance.backward_cell is not None else [state]
    _, cell = cell.at["__call__"](x[0], state[0])
    return cell


def materialize_backward_cell(instance, x, state=None, **__) -> RNNCell | None:
    if instance.backward_cell is None:
        return None
    cell = instance.cell
    state = state if state is not None else sk.tree_state(instance, x)
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
        >>> cell = sk.nn.SimpleRNNCell(10, 20) # 10-dimensional input, 20-dimensional hidden state
        >>> rnn = sk.nn.ScanRNN(cell, return_state=True)
        >>> x = jnp.ones((5, 10)) # 5 timesteps, 10 features
        >>> result, state = rnn(x)  # 20 features
        >>> print(result.shape)
        (20,)

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> cell = sk.nn.SimpleRNNCell(10, 20)
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
            # bidirectional
            if not isinstance(backward_cell, RNNCell):
                raise TypeError(f"{backward_cell=} to be an instance of `RNNCell`.")
            if cell.in_features != backward_cell.in_features:
                raise ValueError(f"{cell.in_features=} != {backward_cell.in_features=}")
            if cell.hidden_features != backward_cell.hidden_features:
                raise ValueError(
                    f"{cell.hidden_features=} != {backward_cell.hidden_features=}."
                )

        self.cell = cell
        self.backward_cell = backward_cell
        self.return_sequences = return_sequences
        self.return_state = return_state

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    def __call__(
        self,
        x: jax.Array,
        state: RNNState | None = None,
        **k,
    ) -> jax.Array | tuple[jax.Array, RNNState]:
        """Scans the RNN cell over a sequence.

        Args:
            x: the input sequence.
            state: the initial state. if None, state is initialized by the rule
                defined using :func:`.tree_state`.

        Returns:
            return the result and state if ``return_state`` is True. otherwise,
            return only the result.
        """

        if not isinstance(state, (RNNState, type(None))):
            raise TypeError(f"Expected state to be an instance of RNNState, {state=}")

        if x.ndim != self.cell.spatial_ndim + 2:
            raise ValueError(
                f"Expected x to have {(self.cell.spatial_ndim + 2)=} dimensions corresponds to "
                f"(timesteps, in_features, {','.join('...'*self.cell.spatial_ndim)}),"
                f" got {x.ndim=}"
            )

        if self.cell.in_features != x.shape[1]:
            raise ValueError(
                f"Expected x to have shape (timesteps, {self.cell.in_features},"
                f"{'*'*self.cell.spatial_ndim}), got {x.shape=}"
            )

        state: RNNState = tree_state(self, array=x)  # if state is None else state
        scan_func = _accumulate_scan if self.return_sequences else _no_accumulate_scan

        if self.backward_cell is None:
            result, state = scan_func(x, self.cell, state)
            return (result, state) if self.return_state else result

        # bidirectional
        lhs_state, rhs_state = split_state(state, splits=2)
        lhs_result, lhs_state = scan_func(x, self.cell, lhs_state, False)
        rhs_result, rhs_state = scan_func(x, self.backward_cell, rhs_state, True)
        concat_axis = int(self.return_sequences)
        result = jnp.concatenate((lhs_result, rhs_result), axis=concat_axis)
        state: RNNState = concat_state((lhs_state, rhs_state))
        return (result, state) if self.return_state else result


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


def _accumulate_scan(
    x: jax.Array,
    cell: RNNCell,
    state: RNNState,
    reverse: bool = False,
) -> tuple[jax.Array, RNNState]:
    def scan_func(carry, x):
        state = cell(x, state=carry)
        return state, state

    x = jnp.flip(x, axis=0) if reverse else x  # flip over time axis
    result = jax.lax.scan(scan_func, state, x)[1].hidden_state
    carry, result = jax.lax.scan(scan_func, state, x)
    result = result.hidden_state
    result = jnp.flip(result, axis=-1) if reverse else result
    return result, carry


def _no_accumulate_scan(
    x: jax.Array,
    cell: RNNCell,
    state: RNNState,
    reverse: bool = False,
) -> jax.Array:
    def scan_func(carry, x):
        state = cell(x, state=carry)
        return state, None

    x = jnp.flip(x, axis=0) if reverse else x
    carry, _ = jax.lax.scan(scan_func, state, x)
    result = carry.hidden_state
    return result, carry


# register state handlers


@tree_state.def_state(SimpleRNNCell)
def simple_rnn_init_state(cell: SimpleRNNCell) -> SimpleRNNState:
    return SimpleRNNState(jnp.zeros([cell.hidden_features]))


@tree_state.def_state(DenseCell)
def dense_init_state(cell: DenseCell) -> DenseState:
    return DenseState(jnp.empty([cell.hidden_features]))


@tree_state.def_state(LSTMCell)
def lstm_init_state(cell: LSTMCell) -> LSTMState:
    shape = [cell.hidden_features]
    return LSTMState(jnp.zeros(shape), jnp.zeros(shape))


@tree_state.def_state(GRUCell)
def gru_init_state(cell: GRUCell) -> GRUState:
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
def conv_lstm_init_state(cell: ConvLSTMNDCell, *, array: Any) -> ConvLSTMNDState:
    array = _check_rnn_cell_tree_state_input(cell, array)
    shape = (cell.hidden_features, *array.shape[1:])
    zeros = jnp.zeros(shape).astype(array.dtype)
    return ConvLSTMNDState(zeros, zeros)


@tree_state.def_state(ConvGRUNDCell)
def conv_gru_init_state(cell: ConvGRUNDCell, *, array: Any) -> ConvGRUNDState:
    array = _check_rnn_cell_tree_state_input(cell, array)
    shape = (cell.hidden_features, *array.shape[1:])
    return ConvGRUNDState(jnp.zeros(shape).astype(array.dtype))


@tree_state.def_state(ScanRNN)
def scan_rnn_init_state(rnn: ScanRNN, *, array: jax.Array | None = None) -> RNNState:
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
