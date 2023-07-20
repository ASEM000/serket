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

import serket as sk
from serket.nn.activation import ActivationType, resolve_activation
from serket.nn.initialization import InitType
from serket.nn.state import tree_state
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


# Non Spatial RNN


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

        Note:
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNNCell.
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

    @property
    def spatial_ndim(self) -> int:
        return 0

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, state: SimpleRNNState, **k) -> SimpleRNNState:
        if not isinstance(state, SimpleRNNState):
            raise TypeError(f"Expected {state=} to be an instance of `SimpleRNNState`")

        ih = jnp.concatenate([x, state.hidden_state], axis=-1)
        h = ih @ self.ih2h_weight + self.ih2h_bias
        return SimpleRNNState(self.act_func(h))


@tree_state.def_state(SimpleRNNCell)
def simple_rnn_init_state(cell: SimpleRNNCell, _) -> SimpleRNNState:
    return SimpleRNNState(jnp.zeros([cell.hidden_features]))


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
        >>> cell = sk.nn.DenseCell(10, 20) # 10-dimensional input, 20-dimensional hidden state
        >>> dummy_state = sk.tree_state(cell)  # 20-dimensional hidden state
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

    @property
    def spatial_ndim(self) -> int:
        return 0

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, state: DenseState, **k) -> DenseState:
        if not isinstance(state, DenseState):
            raise TypeError(f"Expected {state=} to be an instance of `DenseState`")

        h = self.act_func(self.in_to_hidden(x))
        return DenseState(h)


@tree_state.def_state(DenseCell)
def dense_init_state(cell: DenseCell, _) -> DenseState:
    return DenseState(jnp.empty([cell.hidden_features]))


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

    Note:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTMCell
        https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py
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

    def init_state(self, spatial_dim: tuple[int, ...]) -> LSTMState:
        del spatial_dim
        shape = (self.hidden_features,)
        return LSTMState(jnp.zeros(shape), jnp.zeros(shape))

    @property
    def spatial_ndim(self) -> int:
        return 0


@tree_state.def_state(LSTMCell)
def lstm_init_state(cell: LSTMCell, _) -> LSTMState:
    shape = [cell.hidden_features]
    return LSTMState(jnp.zeros(shape), jnp.zeros(shape))


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

    Note:
        https://keras.io/api/layers/recurrent_layers/gru/
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

    @property
    def spatial_ndim(self) -> int:
        return 0

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


@tree_state.def_state(GRUCell)
def gru_init_state(cell: GRUCell, _) -> GRUState:
    return GRUState(jnp.zeros([cell.hidden_features]))


# Spatial RNN


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

    Note:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D
    """

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
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        recurrent_weight_init_func: InitType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "hard_sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
        conv_layer: Any = None,
    ):
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


@tree_state.def_state(ConvLSTMNDCell)
def conv_lstm_init_state(cell: ConvLSTMNDCell, x: jax.Array | None) -> ConvLSTMNDState:
    if not (hasattr(x, "ndim") and hasattr(x, "shape")):
        raise TypeError(
            f"Expected {x=} to have ndim and shape attributes.",
            "To initialize the `ConvLSTMNDCell` state.\n"
            "pass a single sample array to `tree_state` second argument.",
        )

    if x.ndim != cell.spatial_ndim + 1:
        raise ValueError(
            f"{x.ndim=} != {(cell.spatial_ndim + 1)=}.",
            "Expected input to have shape (channel, *spatial_dim)."
            "Pass a single sample array to `tree_state",
        )

    spatial_dim = x.shape[1:]
    if len(spatial_dim) != cell.spatial_ndim:
        raise ValueError(f"{len(spatial_dim)=} != {cell.spatial_ndim=}.")
    shape = (cell.hidden_features, *spatial_dim)
    return ConvLSTMNDState(jnp.zeros(shape), jnp.zeros(shape))


class ConvLSTM1DCell(ConvLSTMNDCell):
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

    Note:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D
    """

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
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        recurrent_weight_init_func: InitType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "hard_sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
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

    Note:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D
    """

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
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        recurrent_weight_init_func: InitType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "hard_sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
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

    Note:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D
    """

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
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        recurrent_weight_init_func: InitType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "hard_sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
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
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        recurrent_weight_init_func: InitType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
        conv_layer: Any = None,
    ):
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
            raise TypeError(f"Expected {state=} to be an instance of `GRUState`")

        h = state.hidden_state
        xe, xu, xo = jnp.split(self.in_to_hidden(x), 3, axis=0)
        he, hu, ho = jnp.split(self.hidden_to_hidden(h), 3, axis=0)
        e = self.recurrent_act_func(xe + he)
        u = self.recurrent_act_func(xu + hu)
        o = self.act_func(xo + (e * ho))
        h = (1 - u) * o + u * h
        return ConvGRUNDState(hidden_state=h)


@tree_state.def_state(ConvGRUNDCell)
def conv_gru_init_state(cell: ConvGRUNDCell, x: jax.Array | None) -> ConvGRUNDState:
    if not (hasattr(x, "ndim") and hasattr(x, "shape")):
        # maybe the input is not an array
        raise TypeError(
            f"Expected {x=} to have ndim and shape attributes.",
            "To initialize the `ConvGRUNDCell` state.\n"
            "pass a single sample array to `tree_state` second argument.",
        )

    if x.ndim != cell.spatial_ndim + 1:
        # channel, *spatial_dim
        raise ValueError(
            f"{x.ndim=} != {(cell.spatial_ndim + 1)=}.",
            "Expected input to have shape (channel, *spatial_dim)."
            "Pass a single sample array to `tree_state",
        )

    spatial_dim = x.shape[1:]
    if len(spatial_dim) != cell.spatial_ndim:
        raise ValueError(f"{len(spatial_dim)=} != {cell.spatial_ndim=}.")
    shape = (cell.hidden_features, *spatial_dim)
    return ConvGRUNDState(jnp.zeros(shape), jnp.zeros(shape))


class ConvGRU1DCell(ConvGRUNDCell):
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
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        recurrent_weight_init_func: InitType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
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
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        recurrent_weight_init_func: InitType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
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
    """

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
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        recurrent_weight_init_func: InitType = "orthogonal",
        act_func: ActivationType | None = "tanh",
        recurrent_act_func: ActivationType | None = "sigmoid",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
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


class ScanRNN(sk.TreeClass):
    """Scans RNN cell over a sequence.

    Args:
        cell: the RNN cell to use.
        backward_cell: the RNN cell to use for bidirectional scanning.
        return_sequences: whether to return the hidden state for each timestep.

    Example:
        >>> cell = sk.nn.SimpleRNNCell(10, 20) # 10-dimensional input, 20-dimensional hidden state
        >>> rnn = sk.nn.ScanRNN(cell)
        >>> x = jnp.ones((5, 10)) # 5 timesteps, 10 features
        >>> result, state = rnn(x)  # 20 features
        >>> print(result.shape)
        (20,)
        >>> cell = sk.nn.SimpleRNNCell(10, 20)
        >>> rnn = sk.nn.ScanRNN(cell, return_sequences=True)
        >>> x = jnp.ones((5, 10)) # 5 timesteps, 10 features
        >>> result, state = rnn(x)  # 5 timesteps, 20 features
        >>> print(result.shape)
        (5, 20)
    """

    # cell: RNN

    def __init__(
        self,
        cell: RNNCell,
        backward_cell: RNNCell | None = None,
        *,
        return_sequences: bool = False,
    ):
        if not isinstance(cell, RNNCell):
            raise TypeError(f"Expected {cell=} to be an instance of RNNCell.")

        if not isinstance(backward_cell, (RNNCell, type(None))):
            raise TypeError(f"Expected {backward_cell=} to be an instance of RNNCell.")

        self.cell = cell
        self.backward_cell = backward_cell
        self.return_sequences = return_sequences

    def __call__(
        self,
        x: jax.Array,
        state: RNNState | None = None,
        backward_state: RNNState | None = None,
        **k,
    ) -> tuple[jax.Array, tuple[RNNState, RNNState] | RNNState]:
        """Scans the RNN cell over a sequence.

        Args:
            x: the input sequence.
            state: the initial state. if None, a zero state is used.
            backward_state: the initial backward state. if None, a zero state is used.

        Returns:
            the output sequence and the final two states tuple if backward_cell
            is not ``None``, otherwise return the final state of the forward
            cell.
        """

        if not isinstance(state, (RNNState, type(None))):
            raise TypeError(f"Expected state to be an instance of RNNState, {state=}")

        # non-spatial RNN : (time steps, in_features)
        # spatial RNN : (time steps, in_features, *spatial_dims)

        if x.ndim != self.cell.spatial_ndim + 2:
            raise ValueError(
                f"Expected x to have {self.cell.spatial_ndim + 2} dimensions corresponds to "
                f"(timesteps, in_features, {'*'*self.cell.spatial_ndim}),"
                f" got {x.ndim=}"
            )

        if self.cell.in_features != x.shape[1]:
            raise ValueError(
                f"Expected x to have shape (timesteps, {self.cell.in_features},"
                f"{'*'*self.cell.spatial_ndim}), got {x.shape=}"
            )
        # pass a sample not the whole sequence
        state = state or tree_state(self.cell, x[0])

        if self.backward_cell is not None and backward_state is None:
            # pass a sample not the whole sequence
            backward_state = tree_state(self.backward_cell, x[0])

        scan_func = _accumulate_scan if self.return_sequences else _no_accumulate_scan
        result, state = scan_func(x, self.cell, state)

        states = state

        if self.backward_cell is not None:
            backward_result, backward_state = scan_func(
                x,
                self.backward_cell,
                backward_state,
            )
            states = (state, backward_state)
            concat_axis = int(self.return_sequences)
            result = jnp.concatenate((result, backward_result), axis=concat_axis)

        return result, states


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
