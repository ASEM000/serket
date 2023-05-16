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
import pytreeclass as pytc

import serket as sk
from serket.nn.utils import (
    ActivationType,
    DilationType,
    InitFuncType,
    KernelSizeType,
    PaddingType,
    StridesType,
    positive_int_cb,
    resolve_activation,
    validate_axis_shape,
    validate_spatial_ndim,
)

"""Defines RNN related classes."""


# Non Spatial RNN


class RNNState(pytc.TreeClass):
    hidden_state: jax.Array


class RNNCell(pytc.TreeClass, abc.ABC):
    @abc.abstractclassmethod
    def __call__(self, x: jax.Array, state: RNNState, **k) -> RNNState:
        ...

    @abc.abstractclassmethod
    def init_state(self, spatial_shape: tuple[int, ...]) -> RNNState:
        # return the initial state of the RNN for a given input
        # for non-spatial RNNs, output shape is (hidden_features,)
        # for spatial RNNs, output shape is (hidden_features, *spatial_shape)
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
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
        recurrent_weight_init_func: InitFuncType = "orthogonal",
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
            >>> cell = SimpleRNNCell(10, 20) # 10-dimensional input, 20-dimensional hidden state
            >>> rnn_state = cell.init_state()  # 20-dimensional hidden state
            >>> x = jnp.ones((10,)) # 10 features
            >>> result = cell(x, rnn_state)
            >>> result.hidden_state.shape  # 20 features
            (20,)

        Note:
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNNCell.
        """
        k1, k2 = jr.split(key, 2)

        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act_func = resolve_activation(act_func)

        in_to_hidden = sk.nn.Linear(
            in_features,
            hidden_features,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=k1,
        )

        hidden_to_hidden = sk.nn.Linear(
            hidden_features,
            hidden_features,
            weight_init_func=recurrent_weight_init_func,
            bias_init_func=None,
            key=k2,
        )

        self.in_and_hidden_to_hidden = sk.nn.MergeLinear(in_to_hidden, hidden_to_hidden)

    @property
    def spatial_ndim(self) -> int:
        return 0

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, state: SimpleRNNState, **k) -> SimpleRNNState:
        if not isinstance(state, SimpleRNNState):
            msg = "Expected state to be an instance of `SimpleRNNState`"
            msg += f", got {type(state).__name__}"
            raise TypeError(msg)

        h = self.act_func(self.in_and_hidden_to_hidden(x, state.hidden_state))
        return SimpleRNNState(h)

    def init_state(self, spatial_dim: tuple[int, ...] = ()) -> SimpleRNNState:
        del spatial_dim
        shape = (self.hidden_features,)
        return SimpleRNNState(jnp.zeros(shape))


class DenseState(RNNState):
    ...


class DenseCell(RNNCell):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
        act_func: ActivationType = jax.nn.tanh,
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
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
            >>> cell = DenseCell(10, 20) # 10-dimensional input, 20-dimensional hidden state
            >>> dummy_state = cell.init_state()  # 20-dimensional hidden state
            >>> x = jnp.ones((10,)) # 10 features
            >>> result = cell(x, dummy_state)
            >>> result.hidden_state.shape  # 20 features
            (20,)
        """

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

    @property
    def spatial_ndim(self) -> int:
        return 0

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, state: DenseState, **k) -> DenseState:
        if not isinstance(state, DenseState):
            msg = "Expected state to be an instance of `DenseState`"
            msg += f", got {type(state).__name__}"
            raise TypeError(msg)

        h = self.act_func(self.in_to_hidden(x))
        return DenseState(h)

    def init_state(self, spatial_dim: tuple[int, ...] = ()) -> DenseState:
        del spatial_dim
        shape = (self.hidden_features,)
        return DenseState(jnp.empty(shape))  # dummy state


class LSTMState(RNNState):
    cell_state: jax.Array


class LSTMCell(RNNCell):
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

        Note:
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTMCell
            https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py
        """
        k1, k2 = jr.split(key, 2)

        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act_func = resolve_activation(act_func)
        self.recurrent_act_func = resolve_activation(recurrent_act_func)

        in_to_hidden = sk.nn.Linear(
            in_features,
            hidden_features * 4,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=k1,
        )

        hidden_to_hidden = sk.nn.Linear(
            hidden_features,
            hidden_features * 4,
            weight_init_func=recurrent_weight_init_func,
            bias_init_func=None,
            key=k2,
        )

        self.in_and_hidden_to_hidden = sk.nn.MergeLinear(in_to_hidden, hidden_to_hidden)

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, state: LSTMState, **k) -> LSTMState:
        if not isinstance(state, LSTMState):
            msg = "Expected state to be an instance of `LSTMState`"
            msg += f", got {type(state).__name__}"
            raise TypeError(msg)

        h, c = state.hidden_state, state.cell_state
        h = self.in_and_hidden_to_hidden(x, h)
        i, f, g, o = jnp.split(h, 4, axis=-1)
        i = self.recurrent_act_func(i)
        f = self.recurrent_act_func(f)
        g = self.act_func(g)
        o = self.recurrent_act_func(o)
        c = f * c + i * g
        h = o * self.act_func(c)
        return LSTMState(h, c)

    def init_state(self, spatial_dim: tuple[int, ...]) -> LSTMState:
        del spatial_dim
        shape = (self.hidden_features,)
        return LSTMState(jnp.zeros(shape), jnp.zeros(shape))

    @property
    def spatial_ndim(self) -> int:
        return 0


class GRUState(RNNState):
    ...


class GRUCell(RNNCell):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
        recurrent_weight_init_func: InitFuncType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
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

        See:
            https://keras.io/api/layers/recurrent_layers/gru/
        """
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

    @property
    def spatial_ndim(self) -> int:
        return 0

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, state: GRUState, **k) -> GRUState:
        if not isinstance(state, GRUState):
            msg = "Expected state to be an instance of `GRUState`"
            msg += f", got {type(state).__name__}"
            raise TypeError(msg)

        h = state.hidden_state
        xe, xu, xo = jnp.split(self.in_to_hidden(x), 3, axis=-1)
        he, hu, ho = jnp.split(self.hidden_to_hidden(h), 3, axis=-1)
        e = self.recurrent_act_func(xe + he)
        u = self.recurrent_act_func(xu + hu)
        o = self.act_func(xo + (e * ho))
        h = (1 - u) * o + u * h
        return GRUState(hidden_state=h)

    def init_state(self, spatial_dim: tuple[int, ...]) -> GRUState:
        del spatial_dim
        shape = (self.hidden_features,)
        return GRUState(jnp.zeros(shape, dtype=jnp.float32))


# Spatial RNN


class ConvLSTMNDState(RNNState):
    cell_state: jax.Array


class ConvLSTMNDCell(RNNCell):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        kernel_size: KernelSizeType,
        *,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        input_dilation: DilationType = 1,
        kernel_dilation: DilationType = 1,
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
        recurrent_weight_init_func: InitFuncType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "hard_sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
        conv_layer: Any = None,
    ):
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

        See: https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D
        """
        k1, k2 = jr.split(key, 2)

        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act_func = resolve_activation(act_func)
        self.recurrent_act_func = resolve_activation(recurrent_act_func)

        self.in_to_hidden = conv_layer(
            in_features,
            hidden_features * 4,
            kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=k1,
        )

        self.hidden_to_hidden = conv_layer(
            hidden_features,
            hidden_features * 4,
            kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=recurrent_weight_init_func,
            bias_init_func=None,
            key=k2,
        )

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, state: ConvLSTMNDState, **k) -> ConvLSTMNDState:
        if not isinstance(state, ConvLSTMNDState):
            msg = f"Expected state to be an instance of ConvLSTMNDState, got {type(state)}"
            raise TypeError(msg)

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

    def init_state(self, spatial_dim: tuple[int, ...]) -> ConvLSTMNDState:
        msg = f"Expected spatial_dim to be a tuple of length {self.spatial_ndim}, got {spatial_dim}"
        assert len(spatial_dim) == self.spatial_ndim, msg
        shape = (self.hidden_features, *spatial_dim)
        return ConvLSTMNDState(jnp.zeros(shape), jnp.zeros(shape))


class ConvLSTM1DCell(ConvLSTMNDCell):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        kernel_size: KernelSizeType,
        *,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        input_dilation: DilationType = 1,
        kernel_dilation: DilationType = 1,
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
        recurrent_weight_init_func: InitFuncType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "hard_sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """1D Convolution LSTM cell that defines the update rule for the hidden state and cell state
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
            spatial_ndim: Number of spatial dimensions.

        Note:
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D
        """
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            recurrent_weight_init_func=recurrent_weight_init_func,
            act_func=act_func,
            recurrent_act_func=recurrent_act_func,
            key=key,
            conv_layer=sk.nn.Conv1D,
        )

    @property
    def spatial_ndim(self) -> int:
        return 1


class ConvLSTM2DCell(ConvLSTMNDCell):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        kernel_size: KernelSizeType,
        *,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        input_dilation: DilationType = 1,
        kernel_dilation: DilationType = 1,
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
        recurrent_weight_init_func: InitFuncType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "hard_sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """2D Convolution LSTM cell that defines the update rule for the hidden state and cell state
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
            spatial_ndim: Number of spatial dimensions.

        Note:
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D
        """
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            recurrent_weight_init_func=recurrent_weight_init_func,
            act_func=act_func,
            recurrent_act_func=recurrent_act_func,
            key=key,
            conv_layer=sk.nn.Conv2D,
        )

    @property
    def spatial_ndim(self) -> int:
        return 2


class ConvLSTM3DCell(ConvLSTMNDCell):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        kernel_size: KernelSizeType,
        *,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        input_dilation: DilationType = 1,
        kernel_dilation: DilationType = 1,
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
        recurrent_weight_init_func: InitFuncType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "hard_sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """3D Convolution LSTM cell that defines the update rule for the hidden state and cell state
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
            spatial_ndim: Number of spatial dimensions.

        Note:
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D
        """
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            recurrent_weight_init_func=recurrent_weight_init_func,
            act_func=act_func,
            recurrent_act_func=recurrent_act_func,
            key=key,
            conv_layer=sk.nn.Conv3D,
        )

    @property
    def spatial_ndim(self) -> int:
        return 3


class ConvGRUNDState(RNNState):
    ...


class ConvGRUNDCell(RNNCell):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        input_dilation: DilationType = 1,
        kernel_dilation: DilationType = 1,
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
        recurrent_weight_init_func: InitFuncType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
        conv_layer: Any = None,
    ):
        """Convolution GRU cell that defines the update rule for the hidden state and cell state
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
            spatial_ndim: Number of spatial dimensions.

        """
        k1, k2 = jr.split(key, 2)

        self.in_features = positive_int_cb(in_features)
        self.hidden_features = positive_int_cb(hidden_features)
        self.act_func = resolve_activation(act_func)
        self.recurrent_act_func = resolve_activation(recurrent_act_func)

        self.in_to_hidden = conv_layer(
            in_features,
            hidden_features * 3,
            kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=k1,
        )

        self.hidden_to_hidden = conv_layer(
            hidden_features,
            hidden_features * 3,
            kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=recurrent_weight_init_func,
            bias_init_func=None,
            key=k2,
        )

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, state: ConvGRUNDState, **k) -> ConvGRUNDState:
        if not isinstance(state, ConvGRUNDState):
            msg = f"Expected state to be an instance of GRUState, got {type(state)}"
            raise TypeError(msg)

        h = state.hidden_state
        xe, xu, xo = jnp.split(self.in_to_hidden(x), 3, axis=0)
        he, hu, ho = jnp.split(self.hidden_to_hidden(h), 3, axis=0)
        e = self.recurrent_act_func(xe + he)
        u = self.recurrent_act_func(xu + hu)
        o = self.act_func(xo + (e * ho))
        h = (1 - u) * o + u * h
        return ConvGRUNDState(hidden_state=h)

    def init_state(self, spatial_dim: tuple[int, ...]) -> ConvGRUNDState:
        msg = f"Expected spatial_dim to be a tuple of length {self.spatial_ndim}, got {spatial_dim}"
        assert len(spatial_dim) == self.spatial_ndim, msg
        shape = (self.hidden_features, *spatial_dim)
        return ConvGRUNDState(hidden_state=jnp.zeros(shape))


class ConvGRU1DCell(ConvGRUNDCell):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        input_dilation: DilationType = 1,
        kernel_dilation: DilationType = 1,
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
        recurrent_weight_init_func: InitFuncType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """1D Convolution GRU cell that defines the update rule for the hidden state and cell state
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
            spatial_ndim: Number of spatial dimensions.

        """
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            recurrent_weight_init_func=recurrent_weight_init_func,
            act_func=act_func,
            recurrent_act_func=recurrent_act_func,
            key=key,
            conv_layer=sk.nn.Conv1D,
        )

    @property
    def spatial_ndim(self) -> int:
        return 1


class ConvGRU2DCell(ConvGRUNDCell):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        input_dilation: DilationType = 1,
        kernel_dilation: DilationType = 1,
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
        recurrent_weight_init_func: InitFuncType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """2D Convolution GRU cell that defines the update rule for the hidden state and cell state
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
            spatial_ndim: Number of spatial dimensions.

        """
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            recurrent_weight_init_func=recurrent_weight_init_func,
            act_func=act_func,
            recurrent_act_func=recurrent_act_func,
            key=key,
            conv_layer=sk.nn.Conv2D,
        )

    @property
    def spatial_ndim(self) -> int:
        return 2


class ConvGRU3DCell(ConvGRUNDCell):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        input_dilation: DilationType = 1,
        kernel_dilation: DilationType = 1,
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
        recurrent_weight_init_func: InitFuncType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """3D Convolution GRU cell that defines the update rule for the hidden state and cell state
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
            spatial_ndim: Number of spatial dimensions.

        """
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            recurrent_weight_init_func=recurrent_weight_init_func,
            act_func=act_func,
            recurrent_act_func=recurrent_act_func,
            key=key,
            conv_layer=sk.nn.Conv3D,
            spatial_ndim=3,
        )

    @property
    def spatial_ndim(self) -> int:
        return 3


# Scanning API


class ScanRNN(pytc.TreeClass):
    def __init__(
        self,
        cell: RNNCell,
        backward_cell: RNNCell | None = None,
        *,
        return_sequences: bool = False,
    ):
        """Scans RNN cell over a sequence.

        Args:
            cell: the RNN cell to use
            backward_cell: the RNN cell to use for bidirectional scanning.
            return_sequences: whether to return the hidden state for each timestep

        Example:
            >>> cell = SimpleRNNCell(10, 20) # 10-dimensional input, 20-dimensional hidden state
            >>> rnn = ScanRNN(cell)
            >>> x = jnp.ones((5, 10)) # 5 timesteps, 10 features
            >>> result = rnn(x)  # 20 features
        """
        if not isinstance(cell, RNNCell):
            msg = f"Expected `cell` to be an instance of RNNCell got {type(cell)}"
            raise TypeError(msg)

        if not isinstance(backward_cell, (RNNCell, type(None))):
            msg = "Expected `backward_cell` to be an instance of RNNCell, "
            msg += f"got {type(backward_cell).__name__}"
            raise TypeError(msg)

        self.cell = cell
        self.backward_cell = backward_cell
        self.return_sequences = return_sequences

    def __call__(
        self,
        x: jax.Array,
        state: RNNCell | None = None,
        backward_state: RNNState | None = None,
        **k,
    ) -> jax.Array:
        if not isinstance(state, (RNNState, type(None))):
            msg = "Expected state to be an instance of RNNState, "
            msg += f"got {type(state).__name__}"
            raise TypeError(msg)

        # non-spatial RNN : (time steps, in_features)
        # spatial RNN : (time steps, in_features, *spatial_dims)

        if x.ndim != self.cell.spatial_ndim + 2:
            msg = f"Expected x to have {self.cell.spatial_ndim + 2} dimensions corresponds to "
            msg += f"(timesteps, in_features, {'*'*self.cell.spatial_ndim}),"
            msg += f" got {x.ndim}"
            raise ValueError(msg)

        if self.cell.in_features != x.shape[1]:
            msg = f"Expected x to have shape (timesteps, {self.cell.in_features}, {'*'*self.cell.spatial_ndim})"
            msg += f", got {x.shape}"
            raise ValueError(msg)

        state = state or self.cell.init_state(x.shape[2:])

        if self.backward_cell is not None and backward_state is None:
            backward_state = self.backward_cell.init_state(x.shape[2:])

        scan_func = _accumulate_scan if self.return_sequences else _no_accumulate_scan
        result = scan_func(x, self.cell, state)

        if self.backward_cell is not None:
            back_result = scan_func(x, self.backward_cell, backward_state)
            concat_axis = int(self.return_sequences)
            result = jnp.concatenate((result, back_result), axis=concat_axis)
        return result


def _accumulate_scan(
    x: jax.Array,
    cell: RNNCell,
    state: RNNState,
    reverse: bool = False,
) -> jax.Array:
    def scan_func(carry, x):
        state = cell(x, state=carry)
        return state, state

    x = jnp.flip(x, axis=0) if reverse else x  # flip over time axis
    result = jax.lax.scan(scan_func, state, x)[1].hidden_state
    return jnp.flip(result, axis=-1) if reverse else result


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
    return jax.lax.scan(scan_func, state, x)[0].hidden_state
