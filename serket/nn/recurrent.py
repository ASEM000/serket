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
from jax.util import unzip2

import serket as sk
from serket.nn.activation import ActivationType, resolve_activation
from serket.nn.custom_transform import tree_state
from serket.nn.initialization import InitType
from serket.nn.utils import (
    DilationType,
    KernelSizeType,
    PaddingType,
    StridesType,
    positive_int_cb,
    validate_axis_shape,
    validate_spatial_ndim,
)

"""Defines RNN related classes."""


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
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        recurrent_weight_init_func: InitType = "orthogonal",
        act_func: ActivationType = jax.nn.tanh,
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """Vanilla RNN cell that defines the update rule for the hidden state

        Args:
            in_features: the number of input features
            hidden_features: the number of hidden features
            weight_init_func: the function to use to initialize the weights
            bias_init_func: the function to use to initialize the bias
            recurrent_weight_init_func: the function to use to initialize the recurrent weights
            act_func: the activation function to use for the hidden state update
            key: the key to use to initialize the weights

        Example:
            >>> import serket as sk
            >>> import jax.numpy as jnp
            >>> cell = sk.nn.SimpleRNNCell(10, 20) # 10-dimensional input, 20-dimensional hidden state
            >>> rnn_state = sk.tree_state(cell)  # 20-dimensional hidden state
            >>> x = jnp.ones((10,)) # 10 features
            >>> result = cell(x, rnn_state)
            >>> result.hidden_state.shape  # 20 features
            (20,)

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
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=k1,
        )

        h2h = sk.nn.Linear(
            hidden_features,
            hidden_features,
            weight_init_func=recurrent_weight_init_func,
            bias_init_func=None,
            key=k2,
        )

        self.ih2h_weight = jnp.concatenate([i2h.weight, h2h.weight], axis=0)
        self.ih2h_bias = i2h.bias

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, state: SimpleRNNState, **k) -> SimpleRNNState:
        if not isinstance(state, SimpleRNNState):
            raise TypeError(f"Expected {state=} to be an instance of `SimpleRNNState`")

        ih = jnp.concatenate([x, state.hidden_state], axis=-1)
        h = ih @ self.ih2h_weight + self.ih2h_bias
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
        weight_init_func: the function to use to initialize the weights
        bias_init_func: the function to use to initialize the bias
        act_func: the activation function to use for the hidden state update,
            use `None` for no activation
        key: the key to use to initialize the weights

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
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        act_func: ActivationType = jax.nn.tanh,
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act_func = resolve_activation(act_func)

        self.in_to_hidden = sk.nn.Linear(
            in_features,
            hidden_features,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=key,
        )

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
        weight_init_func: the function to use to initialize the weights
        bias_init_func: the function to use to initialize the bias
        recurrent_weight_init_func: the function to use to initialize the recurrent weights
        act_func: the activation function to use for the hidden state update
        recurrent_act_func: the activation function to use for the cell state update
        key: the key to use to initialize the weights

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTMCell
        - https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable | None = "zeros",
        recurrent_weight_init_func: str | Callable = "orthogonal",
        act_func: str | Callable[[Any], Any] | None = "tanh",
        recurrent_act_func: ActivationType | None = "sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        k1, k2 = jr.split(key, 2)

        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act_func = resolve_activation(act_func)
        self.recurrent_act_func = resolve_activation(recurrent_act_func)

        i2h = sk.nn.Linear(
            in_features,
            hidden_features * 4,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=k1,
        )

        h2h = sk.nn.Linear(
            hidden_features,
            hidden_features * 4,
            weight_init_func=recurrent_weight_init_func,
            bias_init_func=None,
            key=k2,
        )

        self.ih2h_weight = jnp.concatenate([i2h.weight, h2h.weight], axis=0)
        self.ih2h_bias = i2h.bias

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, state: LSTMState, **k) -> LSTMState:
        if not isinstance(state, LSTMState):
            raise TypeError(f"Expected {state=} to be an instance of `LSTMState`")

        h, c = state.hidden_state, state.cell_state
        ih = jnp.concatenate([x, h], axis=-1)
        h = ih @ self.ih2h_weight + self.ih2h_bias
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
        weight_init_func: the function to use to initialize the weights
        bias_init_func: the function to use to initialize the bias
        recurrent_weight_init_func: the function to use to initialize the recurrent weights
        act_func: the activation function to use for the hidden state update
        recurrent_act_func: the activation function to use for the cell state update
        key: the key to use to initialize the weights

    Reference:
        - https://keras.io/api/layers/recurrent_layers/gru/
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        recurrent_weight_init_func: InitType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        k1, k2 = jr.split(key, 2)

        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act_func = resolve_activation(act_func)
        self.recurrent_act_func = resolve_activation(recurrent_act_func)

        self.in_to_hidden = sk.nn.Linear(
            in_features,
            hidden_features * 3,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=k1,
        )

        self.hidden_to_hidden = sk.nn.Linear(
            hidden_features,
            hidden_features * 3,
            weight_init_func=recurrent_weight_init_func,
            bias_init_func=None,
            key=k2,
        )

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
    """Convolution LSTM cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        input_dilation: Dilation of the input
        kernel_dilation: Dilation of the convolutional kernel
        weight_init_func: Weight initialization function
        bias_init_func: Bias initialization function
        recurrent_weight_init_func: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        kernel_size: KernelSizeType,
        *,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        kernel_dilation: DilationType = 1,
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        recurrent_weight_init_func: InitType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "hard_sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
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
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=k1,
        )

        self.hidden_to_hidden = self.convolution_layer(
            hidden_features,
            hidden_features * 4,
            kernel_size,
            strides=strides,
            padding=padding,
            kernel_dilation=kernel_dilation,
            weight_init_func=recurrent_weight_init_func,
            bias_init_func=None,
            key=k2,
        )

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
        kernel_dilation: Dilation of the convolutional kernel
        weight_init_func: Weight initialization function
        bias_init_func: Bias initialization function
        recurrent_weight_init_func: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key

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
        kernel_dilation: Dilation of the convolutional kernel
        weight_init_func: Weight initialization function
        bias_init_func: Bias initialization function
        recurrent_weight_init_func: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key

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
        kernel_dilation: Dilation of the convolutional kernel
        weight_init_func: Weight initialization function
        bias_init_func: Bias initialization function
        recurrent_weight_init_func: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key

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
        kernel_dilation: Dilation of the convolutional kernel
        weight_init_func: Weight initialization function
        bias_init_func: Bias initialization function
        recurrent_weight_init_func: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key

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
        kernel_dilation: Dilation of the convolutional kernel
        weight_init_func: Weight initialization function
        bias_init_func: Bias initialization function
        recurrent_weight_init_func: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key

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
        kernel_dilation: Dilation of the convolutional kernel
        weight_init_func: Weight initialization function
        bias_init_func: Bias initialization function
        recurrent_weight_init_func: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key

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
    """Convolution GRU cell that defines the update rule for the hidden state and cell state

    Args:
        in_features: Number of input features
        hidden_features: Number of output features
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution
        padding: Padding of the convolution
        kernel_dilation: Dilation of the convolutional kernel
        weight_init_func: Weight initialization function
        bias_init_func: Bias initialization function
        recurrent_weight_init_func: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
        spatial_ndim: Number of spatial dimensions.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        kernel_dilation: DilationType = 1,
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        recurrent_weight_init_func: InitType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
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
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=k1,
        )

        self.hidden_to_hidden = self.convolution_layer(
            hidden_features,
            hidden_features * 3,
            kernel_size,
            strides=strides,
            padding=padding,
            kernel_dilation=kernel_dilation,
            weight_init_func=recurrent_weight_init_func,
            bias_init_func=None,
            key=k2,
        )

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
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
        kernel_dilation: Dilation of the convolutional kernel
        weight_init_func: Weight initialization function
        bias_init_func: Bias initialization function
        recurrent_weight_init_func: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
        spatial_ndim: Number of spatial dimensions.
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
        kernel_dilation: Dilation of the convolutional kernel
        weight_init_func: Weight initialization function
        bias_init_func: Bias initialization function
        recurrent_weight_init_func: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
        spatial_ndim: Number of spatial dimensions.
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
        kernel_dilation: Dilation of the convolutional kernel
        weight_init_func: Weight initialization function
        bias_init_func: Bias initialization function
        recurrent_weight_init_func: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
        spatial_ndim: Number of spatial dimensions.
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
        kernel_dilation: Dilation of the convolutional kernel
        weight_init_func: Weight initialization function
        bias_init_func: Bias initialization function
        recurrent_weight_init_func: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
        spatial_ndim: Number of spatial dimensions.
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
        kernel_dilation: Dilation of the convolutional kernel
        weight_init_func: Weight initialization function
        bias_init_func: Bias initialization function
        recurrent_weight_init_func: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
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
        kernel_dilation: Dilation of the convolutional kernel
        weight_init_func: Weight initialization function
        bias_init_func: Bias initialization function
        recurrent_weight_init_func: Recurrent weight initialization function
        act_func: Activation function
        recurrent_act_func: Recurrent activation function
        key: PRNG key
    """

    @property
    def spatial_ndim(self) -> int:
        return 3

    @property
    def convolution_layer(self):
        return sk.nn.FFTConv3D


# Scanning API


class ScanRNN(sk.TreeClass):
    """Scans RNN cell over a sequence.

    Args:
        cells: the RNN cell(s) to use.
        return_sequences: whether to return the output for each timestep.
        return_state: whether to return the final state of the RNN cell(s).
        reverse: a tuple of booleans indicating whether to reverse the input
            sequence for each cell. for example, if `cells` is a tuple of
            two cells, and `reverse=(True, False)`, then the first cell will
            scan the input sequence in reverse, and the second cell will scan
            the input sequence in the original order. or a single boolean
            indicating whether to reverse the input sequence for all cells.
            default is `False`.

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

    def __init__(
        self,
        *cells: RNNCell,
        return_sequences: bool = False,
        return_state: bool = False,
        reverse: tuple[bool, ...] | bool = False,
    ):
        cell0, *_ = cells

        for cell in cells:
            if not isinstance(cell, RNNCell):
                raise TypeError(f"Expected {cell=} to be an instance of `RNNCell`.")

            if cell.in_features != cell0.in_features:
                raise ValueError(f"{cell.in_features=} != {cell0.in_features=}.")

            if cell.hidden_features != cell0.hidden_features:
                raise ValueError(
                    f"{cell.hidden_features=} != {cell0.hidden_features=}."
                )

        if isinstance(reverse, bool):
            reverse = (reverse,) * len(cells)

        if len(reverse) != len(cells):
            raise ValueError(f"{len(reverse)=} != {len(cells)=}.")

        self.cells: tuple[RNNCell, ...] | RNNCell = cells
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.reverse = reverse

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
                defined using `tree_state`.

        Returns:
            return the result and state if ``return_state`` is True. otherwise,
             return only the result.
        """

        if not isinstance(state, (RNNState, type(None))):
            raise TypeError(f"Expected state to be an instance of RNNState, {state=}")

        # non-spatial RNN : (time steps, in_features)
        # spatial RNN : (time steps, in_features, *spatial_dims)
        cell0, *_ = self.cells

        if x.ndim != cell0.spatial_ndim + 2:
            raise ValueError(
                f"Expected x to have {(cell0.spatial_ndim + 2)=} dimensions corresponds to "
                f"(timesteps, in_features, {','.join('...'*cell0.spatial_ndim)}),"
                f" got {x.ndim=}"
            )

        if cell0.in_features != x.shape[1]:
            raise ValueError(
                f"Expected x to have shape (timesteps, {cell0.in_features},"
                f"{'*'*cell0.spatial_ndim}), got {x.shape=}"
            )

        splits = len(self.cells)
        state: RNNState = tree_state(self, array=x) if state is None else state
        scan_func = _accumulate_scan if self.return_sequences else _no_accumulate_scan

        result_states: list[tuple[jax.Array, RNNState]] = [
            scan_func(x, ci, si, reverse=ri)
            for ci, ri, si in zip(self.cells, self.reverse, _split(state, splits))
        ]

        results, states = unzip2(result_states)
        result = jnp.concatenate(results, axis=int(self.return_sequences))

        if self.return_state:
            state: RNNState = _merge(states)
            return result, state
        return result


def _split(state: RNNState, splits: int) -> list[RNNState]:
    flat_arrays: list[jax.Array] = jtu.tree_leaves(state)
    return [type(state)(*x) for x in zip(*(jnp.split(x, splits) for x in flat_arrays))]


def _merge(states: list[RNNState]) -> RNNState:
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
def simple_rnn_init_state(cell: SimpleRNNCell, _) -> SimpleRNNState:
    return SimpleRNNState(jnp.zeros([cell.hidden_features]))


@tree_state.def_state(DenseCell)
def dense_init_state(cell: DenseCell, _) -> DenseState:
    return DenseState(jnp.empty([cell.hidden_features]))


@tree_state.def_state(LSTMCell)
def lstm_init_state(cell: LSTMCell, _) -> LSTMState:
    shape = [cell.hidden_features]
    return LSTMState(jnp.zeros(shape), jnp.zeros(shape))


@tree_state.def_state(GRUCell)
def gru_init_state(cell: GRUCell, _) -> GRUState:
    return GRUState(jnp.zeros([cell.hidden_features]))


def _check_rnn_cell_tree_state_input(cell: RNNCell, array):
    if not (hasattr(array, "ndim") and hasattr(array, "shape")):
        raise TypeError(
            f"Expected {array=} to have `ndim` and `shape` attributes.",
            f"To initialize the `{type(cell).__name__}` state.\n",
            "Pass a single sample array to `tree_state(..., array=)`.",
        )

    if array.ndim != cell.spatial_ndim + 1:
        raise ValueError(
            f"{array.ndim=} != {(cell.spatial_ndim + 1)=}.",
            f"Expected input to have `shape` (in_features, {'...'*cell.spatial_dim})."
            "Pass a single sample array to `tree_state",
        )

    spatial_dim = array.shape[1:]
    if len(spatial_dim) != cell.spatial_ndim:
        raise ValueError(f"{len(spatial_dim)=} != {cell.spatial_ndim=}.")

    return array


@tree_state.def_state(ConvLSTMNDCell)
def conv_lstm_init_state(cell: ConvLSTMNDCell, x: Any) -> ConvLSTMNDState:
    x = _check_rnn_cell_tree_state_input(cell, x)
    shape = (cell.hidden_features, *x.shape[1:])
    return ConvLSTMNDState(jnp.zeros(shape), jnp.zeros(shape))


@tree_state.def_state(ConvGRUNDCell)
def conv_gru_init_state(cell: ConvGRUNDCell, x: Any) -> ConvGRUNDState:
    x = _check_rnn_cell_tree_state_input(cell, x)
    shape = (cell.hidden_features, *x.shape[1:])
    return ConvGRUNDState(jnp.zeros(shape), jnp.zeros(shape))


@tree_state.def_state(ScanRNN)
def scan_rnn_init_state(rnn: ScanRNN, x: Any) -> RNNState:
    # should pass a single sample array to `tree_state`
    return _merge(tree_state(rnn.cells, array=x[0]))
