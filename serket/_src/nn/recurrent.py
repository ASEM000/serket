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

import abc
import functools as ft
from typing import Any, Callable, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr
from typing_extensions import ParamSpec

from serket import TreeClass, autoinit
from serket._src.custom_transform import tree_state
from serket._src.nn.activation import ActivationType, resolve_act
from serket._src.nn.convolution import (
    Conv1D,
    Conv2D,
    Conv3D,
    FFTConv1D,
    FFTConv2D,
    FFTConv3D,
)
from serket._src.nn.linear import Linear
from serket._src.utils.lazy import maybe_lazy_call, maybe_lazy_init
from serket._src.utils.typing import (
    DilationType,
    DType,
    InitType,
    KernelSizeType,
    PaddingType,
    StridesType,
)
from serket._src.utils.validate import (
    validate_in_features_shape,
    validate_pos_int,
    validate_spatial_ndim,
)

P = ParamSpec("P")
T = TypeVar("T")
S = TypeVar("S")

State = Any

"""Defines RNN related classes."""


def is_lazy_call(instance, *_1, **_2) -> bool:
    return instance.in_features is None


def is_lazy_init(_, in_features: int | None, *_1, **_2) -> bool:
    return in_features is None


def infer_in_features(_, input: jax.Array, *_1, **_2) -> int:
    return input.shape[0]


updates = dict(in_features=infer_in_features)


@autoinit
class RNNState(TreeClass):
    hidden_state: jax.Array


class SimpleRNNState(RNNState): ...


class SimpleRNNCell(TreeClass):
    """Vanilla RNN cell that defines the update rule for the hidden state

    Args:
        in_features: the number of input features
        hidden_features: the number of hidden features
        key: the key to use to initialize the weights
        weight_init: the function to use to initialize the weights
        bias_init: the function to use to initialize the bias
        recurrent_weight_init: the function to use to initialize the recurrent weights
        act: the activation function to use for the hidden state update
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> # 10-dimensional input, 20-dimensional hidden state
        >>> cell = sk.nn.SimpleRNNCell(10, 20, key=jr.PRNGKey(0))
        >>> # 20-dimensional hidden state
        >>> input = jnp.ones(10) # 10 features
        >>> state = sk.tree_state(cell)
        >>> output, state = cell(input, state)
        >>> state.hidden_state.shape  # 20 features
        (20,)

    Note:
        :class:`.SimpleRNNCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy = sk.nn.SimpleRNNCell(None, 20, key=jr.PRNGKey(0))
        >>> input = jnp.ones(10) # 10 features
        >>> state = sk.tree_state(lazy)
        >>> _, material = sk.value_and_tree(lambda cell: cell(input, state))(cell)
        >>> output, state = material(input, state)
        >>> state.hidden_state.shape
        (20,)

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

        self.in_features = validate_pos_int(in_features)
        self.hidden_features = validate_pos_int(hidden_features)
        self.act = resolve_act(act)

        i2h = Linear(
            in_features,
            hidden_features,
            weight_init=weight_init,
            bias_init=bias_init,
            key=k1,
            dtype=dtype,
        )

        h2h = Linear(
            hidden_features,
            hidden_features,
            weight_init=recurrent_weight_init,
            bias_init=None,
            key=k2,
            dtype=dtype,
        )

        self.in_hidden_to_hidden = Linear(
            in_features=in_features + hidden_features,
            out_features=hidden_features,
            weight_init=lambda *_: jnp.concatenate([i2h.weight, h2h.weight], axis=-1),
            bias_init=lambda *_: i2h.bias,
            dtype=dtype,
            key=k1,  # dummy key
            out_axis=0,
        )

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_ndim, argnum=0)
    @ft.partial(validate_in_features_shape, axis=0)
    def __call__(
        self,
        input: jax.Array,
        state: SimpleRNNState,
    ) -> tuple[jax.Array, SimpleRNNState]:
        if not isinstance(state, SimpleRNNState):
            raise TypeError(f"Expected {state=} to be an instance of `SimpleRNNState`")

        ih = jnp.concatenate([input, state.hidden_state], axis=-1)
        h = self.in_hidden_to_hidden(ih)
        h = self.act(h)
        return h, SimpleRNNState(h)

    spatial_ndim: int = 0


class LinearState(RNNState): ...


class LinearCell(TreeClass):
    """No hidden state cell that applies a dense(Linear+activation) layer to the input

    Args:
        in_features: the number of input features
        hidden_features: the number of hidden features
        key: the key to use to initialize the weights
        weight_init: the function to use to initialize the weights
        bias_init: the function to use to initialize the bias
        act: the activation function to use for the hidden state update,
            use `None` for no activation
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> # 10-dimensional input, 20-dimensional hidden state
        >>> cell = sk.nn.LinearCell(10, 20, key=jr.PRNGKey(0))
        >>> # 20-dimensional hidden state
        >>> input = jnp.ones(10) # 10 features
        >>> state = sk.tree_state(cell)
        >>> output, state = cell(input, state)
        >>> state.hidden_state.shape  # 20 features
        (20,)

    Note:
        :class:`.LinearCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy = sk.nn.LinearCell(None, 20, key=jr.PRNGKey(0))
        >>> input = jnp.ones(10) # 10 features
        >>> state = sk.tree_state(lazy)
        >>> _, material = sk.value_and_tree(lambda cell: cell(input, state))(cell)
        >>> output, state = material(input, state)
        >>> state.hidden_state.shape
        (20,)
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
        self.in_features = validate_pos_int(in_features)
        self.hidden_features = validate_pos_int(hidden_features)
        self.act = resolve_act(act)

        self.in_to_hidden = Linear(
            in_features,
            hidden_features,
            weight_init=weight_init,
            bias_init=bias_init,
            key=key,
            dtype=dtype,
            out_axis=0,
        )

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_ndim, argnum=0)
    @ft.partial(validate_in_features_shape, axis=0)
    def __call__(
        self,
        input: jax.Array,
        state: LinearState,
    ) -> tuple[jax.Array, LinearState]:
        if not isinstance(state, LinearState):
            raise TypeError(f"Expected {state=} to be an instance of `LinearState`")
        h = self.in_to_hidden(input)
        h = self.act(h)
        return h, LinearState(h)

    spatial_ndim: int = 0


@autoinit
class LSTMState(RNNState):
    cell_state: jax.Array


class LSTMCell(TreeClass):
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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> # 10-dimensional input, 20-dimensional hidden state
        >>> cell = sk.nn.LSTMCell(10, 20, key=jr.PRNGKey(0))
        >>> # 20-dimensional hidden state
        >>> input = jnp.ones(10) # 10 features
        >>> state = sk.tree_state(cell)
        >>> output, state = cell(input, state)
        >>> state.hidden_state.shape  # 20 features
        (20,)

    Note:
        :class:`.LSTMCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy = sk.nn.LSTMCell(None, 20, key=jr.PRNGKey(0))
        >>> input = jnp.ones(10) # 10 features
        >>> state = sk.tree_state(lazy)
        >>> _, material = sk.value_and_tree(lambda cell: cell(input, state))(cell)
        >>> output, state = material(input, state)
        >>> state.hidden_state.shape
        (20,)

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

        self.in_features = validate_pos_int(in_features)
        self.hidden_features = validate_pos_int(hidden_features)
        self.act = resolve_act(act)
        self.recurrent_act = resolve_act(recurrent_act)

        i2h = Linear(
            in_features,
            hidden_features * 4,
            weight_init=weight_init,
            bias_init=bias_init,
            key=k1,
            dtype=dtype,
        )

        h2h = Linear(
            hidden_features,
            hidden_features * 4,
            weight_init=recurrent_weight_init,
            bias_init=None,
            key=k2,
            dtype=dtype,
        )

        self.in_hidden_to_hidden = Linear(
            in_features=in_features + hidden_features,
            out_features=hidden_features,
            weight_init=lambda *_: jnp.concatenate([i2h.weight, h2h.weight], axis=-1),
            bias_init=lambda *_: i2h.bias,
            dtype=dtype,
            key=k1,  # dummy key
            out_axis=0,
        )

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_ndim, argnum=0)
    @ft.partial(validate_in_features_shape, axis=0)
    def __call__(
        self,
        input: jax.Array,
        state: LSTMState,
    ) -> tuple[jax.Array, LSTMState]:
        if not isinstance(state, LSTMState):
            raise TypeError(f"Expected {state=} to be an instance of `LSTMState`")

        h, c = state.hidden_state, state.cell_state
        ih = jnp.concatenate([input, h], axis=-1)
        h = self.in_hidden_to_hidden(ih)

        i, f, g, o = jnp.split(h, 4)
        i = self.recurrent_act(i)
        f = self.recurrent_act(f)
        g = self.act(g)
        o = self.recurrent_act(o)
        c = f * c + i * g
        h = o * self.act(c)
        return h, LSTMState(h, c)

    spatial_ndim: int = 0


class GRUState(RNNState): ...


class GRUCell(TreeClass):
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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> # 10-dimensional input, 20-dimensional hidden state
        >>> cell = sk.nn.GRUCell(10, 20, key=jr.PRNGKey(0))
        >>> # 20-dimensional hidden state
        >>> input = jnp.ones(10) # 10 features
        >>> state = sk.tree_state(cell)
        >>> output, state = cell(input, state)
        >>> state.hidden_state.shape  # 20 features
        (20,)

    Note:
        :class:`.GRUCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy = sk.nn.GRUCell(None, 20, key=jr.PRNGKey(0))
        >>> input = jnp.ones(10) # 10 features
        >>> state = sk.tree_state(lazy)
        >>> _, material = sk.value_and_tree(lambda cell: cell(input, state))(cell)
        >>> output, state = material(input, state)
        >>> state.hidden_state.shape
        (20,)

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

        self.in_features = validate_pos_int(in_features)
        self.hidden_features = validate_pos_int(hidden_features)
        self.act = resolve_act(act)
        self.recurrent_act = resolve_act(recurrent_act)

        self.in_to_hidden = Linear(
            in_features,
            hidden_features * 3,
            weight_init=weight_init,
            bias_init=bias_init,
            key=k1,
            dtype=dtype,
            out_axis=0,
        )

        self.hidden_to_hidden = Linear(
            hidden_features,
            hidden_features * 3,
            weight_init=recurrent_weight_init,
            bias_init=None,
            key=k2,
            dtype=dtype,
            out_axis=0,
        )

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_ndim, argnum=0)
    @ft.partial(validate_in_features_shape, axis=0)
    def __call__(
        self,
        input: jax.Array,
        state: GRUState,
    ) -> tuple[jax.Array, GRUState]:
        if not isinstance(state, GRUState):
            raise TypeError(f"Expected {state=} to be an instance of `GRUState`")

        h = state.hidden_state
        xe, xu, xo = jnp.split(self.in_to_hidden(input), 3)
        he, hu, ho = jnp.split(self.hidden_to_hidden(h), 3)
        e = self.recurrent_act(xe + he)
        u = self.recurrent_act(xu + hu)
        o = self.act(xo + (e * ho))
        h = (1 - u) * o + u * h
        return h, GRUState(hidden_state=h)

    spatial_ndim: int = 0


@autoinit
class ConvLSTMNDState(RNNState):
    cell_state: jax.Array


class ConvLSTMNDCell(TreeClass):
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

        self.in_features = validate_pos_int(in_features)
        self.hidden_features = validate_pos_int(hidden_features)
        self.act = resolve_act(act)
        self.recurrent_act = resolve_act(recurrent_act)

        self.in_to_hidden = self.conv_layer(
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

        self.hidden_to_hidden = self.conv_layer(
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
    @ft.partial(validate_spatial_ndim, argnum=0)
    @ft.partial(validate_in_features_shape, axis=0)
    def __call__(
        self,
        input: jax.Array,
        state: ConvLSTMNDState,
    ) -> tuple[jax.Array, ConvLSTMNDState]:
        if not isinstance(state, ConvLSTMNDState):
            raise TypeError(f"Expected {state=} to be an instance of ConvLSTMNDState.")

        h, c = state.hidden_state, state.cell_state
        h = self.in_to_hidden(input) + self.hidden_to_hidden(h)
        i, f, g, o = jnp.split(h, 4, axis=0)
        i = self.recurrent_act(i)
        f = self.recurrent_act(f)
        g = self.act(g)
        o = self.recurrent_act(o)
        c = f * c + i * g
        h = o * self.act(c)
        return h, ConvLSTMNDState(h, c)

    @property
    @abc.abstractmethod
    def conv_layer(self): ...

    spatial_ndim = property(abc.abstractmethod(lambda _: ...))


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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.ConvLSTM1DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4))  # in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> output, state = cell(input, state)
        >>> state.hidden_state.shape
        (2, 4)

    Note:
        :class:`.ConvLSTM1DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy = sk.nn.ConvLSTM1DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4))  # time, in_features, spatial dimensions
        >>> state = sk.tree_state(lazy, input=input)
        >>> _, material = sk.value_and_tree(lambda cell: cell(input, state))(cell)
        >>> output, state = material(input, state)
        >>> state.hidden_state.shape
        (2, 4)

    Reference:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D
    """

    spatial_ndim: int = 1
    conv_layer = Conv1D


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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.FFTConvLSTM1DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4))  # in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> output, state = cell(input, state)
        >>> state.hidden_state.shape
        (2, 4)

    Note:
        :class:`.FFTConvLSTM1DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy = sk.nn.FFTConvLSTM1DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4))  # in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> _, material = sk.value_and_tree(lambda cell: cell(input, state))(cell)
        >>> output, state = material(input, state)
        >>> state.hidden_state.shape
        (2, 4)

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D
    """

    spatial_ndim: int = 1
    conv_layer = FFTConv1D


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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.ConvLSTM2DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4, 4))  # in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> output, state = cell(input, state)
        >>> state.hidden_state.shape
        (2, 4, 4)

    Note:
        :class:`.ConvLSTM2DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy = sk.nn.ConvLSTM2DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4, 4))  # in_features, spatial dimensions
        >>> state = sk.tree_state(lazy, input=input)
        >>> _, material = sk.value_and_tree(lambda cell: cell(input, state))(cell)
        >>> output, state = material(input, state)
        >>> state.hidden_state.shape
        (2, 4, 4)

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM2D
    """

    spatial_ndim: int = 2
    conv_layer = Conv2D


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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.FFTConvLSTM2DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4, 4))  # in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> output, state = cell(input, state)
        >>> state.hidden_state.shape
        (2, 4, 4)

    Note:
        :class:`.FFTConvLSTM2DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy = sk.nn.FFTConvLSTM2DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4, 4))  # time, in_features, spatial dimensions
        >>> state = sk.tree_state(lazy, input=input)
        >>> _, material = sk.value_and_tree(lambda cell: cell(input, state))(cell)
        >>> output, state = material(input, state)
        >>> state.hidden_state.shape
        (2, 4, 4)

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM2D
    """

    spatial_ndim: int = 2
    conv_layer = FFTConv2D


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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.ConvLSTM3DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4, 4, 4))  # in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> output, state = cell(input, state)
        >>> state.hidden_state.shape
        (2, 4, 4, 4)

    Note:
        :class:`.ConvLSTM3DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy = sk.nn.ConvLSTM3DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4, 4, 4))  # in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> _, material = sk.value_and_tree(lambda cell: cell(input, state))(cell)
        >>> output, state = material(input, state)
        >>> state.hidden_state.shape
        (2, 4, 4, 4)

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM3D
    """

    spatial_ndim: int = 3
    conv_layer = Conv3D


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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.FFTConvLSTM3DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4, 4, 4))  # in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> output, state = cell(input, state)
        >>> state.hidden_state.shape
        (2, 4, 4, 4)

    Note:
        :class:`.FFTConvLSTM3DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy = sk.nn.FFTConvLSTM3DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4, 4, 4))  # in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> _, material = sk.value_and_tree(lambda cell: cell(input, state))(cell)
        >>> output, state = material(input, state)
        >>> state.hidden_state.shape
        (2, 4, 4, 4)

    Reference:
        - https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM3D
    """

    spatial_ndim: int = 3
    conv_layer = FFTConv3D


class ConvGRUNDState(RNNState): ...


class ConvGRUNDCell(TreeClass):
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

        self.in_features = validate_pos_int(in_features)
        self.hidden_features = validate_pos_int(hidden_features)
        self.act = resolve_act(act)
        self.recurrent_act = resolve_act(recurrent_act)

        self.in_to_hidden = self.conv_layer(
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

        self.hidden_to_hidden = self.conv_layer(
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
    @ft.partial(validate_spatial_ndim, argnum=0)
    @ft.partial(validate_in_features_shape, axis=0)
    def __call__(
        self,
        input: jax.Array,
        state: ConvGRUNDState,
    ) -> tuple[jax.Array, ConvGRUNDState]:
        if not isinstance(state, ConvGRUNDState):
            raise TypeError(f"Expected {state=} to be an instance of `GRUState`")

        h = state.hidden_state
        xe, xu, xo = jnp.split(self.in_to_hidden(input), 3)
        he, hu, ho = jnp.split(self.hidden_to_hidden(h), 3)
        e = self.recurrent_act(xe + he)
        u = self.recurrent_act(xu + hu)
        o = self.act(xo + (e * ho))
        h = (1 - u) * o + u * h
        return h, ConvGRUNDState(h)

    @property
    @abc.abstractmethod
    def conv_layer(self): ...

    spatial_ndim = property(abc.abstractmethod(lambda _: ...))


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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.ConvGRU1DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4))  # in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> output, state = cell(input, state)
        >>> state.hidden_state.shape
        (2, 4)

    Note:
        :class:`.ConvGRU1DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy = sk.nn.ConvGRU1DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4))  # in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> _, material = sk.value_and_tree(lambda cell: cell(input, state))(cell)
        >>> output, state = material(input, state)
        >>> state.hidden_state.shape
        (2, 4)
    """

    spatial_ndim: int = 1
    conv_layer = Conv1D


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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.FFTConvGRU1DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4))  # time, in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> output, state = cell(input, state)
        >>> state.hidden_state.shape
        (2, 4)

    Note:
        :class:`.FFTConvGRU1DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy = sk.nn.FFTConvGRU1DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4))  # time, in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> _, material = sk.value_and_tree(lambda cell: cell(input, state))(cell)
        >>> output, state = material(input, state)
        >>> state.hidden_state.shape
        (2, 4)
    """

    spatial_ndim: int = 1
    conv_layer = FFTConv1D


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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.ConvGRU2DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4, 4))  # in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> output, state = cell(input, state)
        >>> state.hidden_state.shape
        (2, 4, 4)

    Note:
        :class:`.ConvGRU2DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy = sk.nn.ConvGRU2DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4, 4))  # in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> _, material = sk.value_and_tree(lambda cell: cell(input, state))(cell)
        >>> output, state = material(input, state)
        >>> state.hidden_state.shape
        (2, 4, 4)
    """

    spatial_ndim: int = 2
    conv_layer = Conv2D


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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.FFTConvGRU2DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4, 4))  # in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> output, state = cell(input, state)
        >>> state.hidden_state.shape
        (2, 4, 4)

    Note:
        :class:`.FFTConvGRU2DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy = sk.nn.FFTConvGRU2DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4, 4))  # time, in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> _, material = sk.value_and_tree(lambda cell: cell(input, state))(cell)
        >>> output, state = material(input, state)
        >>> state.hidden_state.shape
        (2, 4, 4)
    """

    spatial_ndim: int = 2
    conv_layer = FFTConv2D


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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.ConvGRU3DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4, 4, 4))  # in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> output, state = cell(input, state)
        >>> state.hidden_state.shape
        (2, 4, 4, 4)

    Note:
        :class:`.ConvGRU3DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy = sk.nn.ConvGRU3DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> state = sk.tree_state(lazy, input=input)
        >>> _, material = sk.value_and_tree(lambda cell: cell(input, state))(cell)
        >>> output, state = material(input, state)
        >>> state.hidden_state.shape
        (2, 4, 4, 4)
    """

    spatial_ndim: int = 3
    conv_layer = Conv3D


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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> cell = sk.nn.FFTConvGRU3DCell(10, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4, 4, 4))  # in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> output, state = cell(input, state)
        >>> state.hidden_state.shape
        (2, 4, 4, 4)

    Note:
        :class:`.FFTConvGRU3DCell` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> lazy = sk.nn.FFTConvGRU3DCell(None, 2, 3, key=jr.PRNGKey(0))
        >>> input = jnp.ones((10, 4, 4, 4))  # time, in_features, spatial dimensions
        >>> state = sk.tree_state(cell, input=input)
        >>> _, material = sk.value_and_tree(lambda cell: cell(input, state))(cell)
        >>> output, state = material(input, state)
        >>> state.hidden_state.shape
        (2, 4, 4, 4)
    """

    spatial_ndim: int = 3
    conv_layer = FFTConv3D


def scan_cell(
    cell,
    in_axis: int = 0,
    out_axis: int = 0,
    reverse: bool = False,
) -> Callable[[jax.Array, S], tuple[jax.Array, S]]:
    """Scan am RNN cell over a sequence.

    Args:
        cell: the RNN cell to scan. The cell should have the following signature:
            `cell(input, state) -> tuple[output, state]`
        in_axis: the axis to scan over. Defaults to 0.
        out_axis: the axis to move the output to. Defaults to 0.
        reverse: whether to scan the sequence in reverse order. Defaults to ``False``.

    Example:
        Unidirectional RNN:

        >>> import serket as sk
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> key = jr.PRNGKey(0)
        >>> cell = sk.nn.SimpleRNNCell(1, 2, key=key)
        >>> state = sk.tree_state(cell)
        >>> input = jnp.ones([10, 1])
        >>> output, state = sk.nn.scan_cell(cell)(input, state)
        >>> print(output.shape)
        (10, 2)

    Example:
        Bidirectional RNN:

        >>> import serket as sk
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> k1, k2 = jr.split(jr.PRNGKey(0))
        >>> cell1 = sk.nn.SimpleRNNCell(1, 2, key=k1)
        >>> cell2 = sk.nn.SimpleRNNCell(1, 2, key=k2)
        >>> state1, state2 = sk.tree_state((cell1, cell2))
        >>> input = jnp.ones([10, 1])
        >>> output1, state1 = sk.nn.scan_cell(cell1)(input, state1)
        >>> output2, state2 = sk.nn.scan_cell(cell2, reverse=True)(input, state2)
        >>> output = jnp.concatenate((output1, output2), axis=1)
        >>> print(output.shape)
        (10, 4)

    Example:
        Combining multiple RNN cells:

        >>> import serket as sk
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import numpy.testing as npt
        >>> k1, k2 = jr.split(jr.PRNGKey(0))
        >>> cell1 = sk.nn.LSTMCell(1, 2, bias_init=None, key=k1)
        >>> cell2 = sk.nn.LSTMCell(2, 1, bias_init=None, key=k2)
        >>> def cell(input, state):
        ...     state1, state2 = state
        ...     output, state1 = cell1(input, state1)
        ...     output, state2 = cell2(output, state2)
        ...     return output, (state1, state2)
        >>> state = sk.tree_state((cell1, cell2))
        >>> input = jnp.ones([2, 1])
        >>> output1, state = sk.nn.scan_cell(cell)(input, state)
        <BLANKLINE>
        >>> # This is equivalent to:
        >>> state1, state2 = sk.tree_state((cell1, cell2))
        >>> output2 = jnp.zeros([2, 1])
        >>> # first step
        >>> output, state1 = cell1(input[0], state1)
        >>> output, state2 = cell2(output, state2)
        >>> output2 = output2.at[0].set(output)
        >>> # second step
        >>> output, state1 = cell1(input[1], state1)
        >>> output, state2 = cell2(output, state2)
        >>> output2 = output2.at[1].set(output)
        >>> npt.assert_allclose(output1, output2, atol=1e-6)
    """

    def scan_func(state: S, input: jax.Array) -> tuple[S, jax.Array]:
        output, state = cell(input, state)
        return state, output

    def wrapper(input: jax.Array, state: S) -> tuple[jax.Array, S]:
        # push the scan axis to the front
        input = jnp.moveaxis(input, in_axis, 0)
        state, output = jax.lax.scan(scan_func, state, input, reverse=reverse)
        # move the output axis to the desired location
        output = jnp.moveaxis(output, 0, out_axis)
        return output, state

    return wrapper


# register state handlers


@tree_state.def_state(SimpleRNNCell)
def _(cell: SimpleRNNCell) -> SimpleRNNState:
    return SimpleRNNState(jnp.zeros([cell.hidden_features]))


@tree_state.def_state(LinearCell)
def _(cell: LinearCell) -> LinearState:
    return LinearState(jnp.empty([cell.hidden_features]))


@tree_state.def_state(LSTMCell)
def _(cell: LSTMCell) -> LSTMState:
    shape = [cell.hidden_features]
    return LSTMState(jnp.zeros(shape), jnp.zeros(shape))


@tree_state.def_state(GRUCell)
def _(cell: GRUCell) -> GRUState:
    return GRUState(jnp.zeros([cell.hidden_features]))


def _check_rnn_cell_tree_state_input(cell, input):
    if not (hasattr(input, "ndim") and hasattr(input, "shape")):
        raise TypeError(
            f"Expected {input=} to have `ndim` and `shape` attributes."
            f"To initialize the `{type(cell).__name__}` state.\n"
            "Pass a single sample input to `tree_state(..., input=)`."
        )

    if input.ndim != cell.spatial_ndim + 1:
        raise ValueError(
            f"{input.ndim=} != {(cell.spatial_ndim+1)=}.\n"
            f"Expected input to {type(cell).__name__} to have `shape` (in_features, {'... '*cell.spatial_ndim}).\n"
            f"Pass a single sample input to `tree_state({type(cell).__name__}, input=...)`"
        )

    if len(spatial_dim := input.shape[1:]) != cell.spatial_ndim:
        raise ValueError(f"{len(spatial_dim)=} != {cell.spatial_ndim=}.")

    return input


@tree_state.def_state(ConvLSTMNDCell)
def _(cell: ConvLSTMNDCell, input) -> ConvLSTMNDState:
    input = _check_rnn_cell_tree_state_input(cell, input)
    shape = (cell.hidden_features, *input.shape[1:])
    zeros = jnp.zeros(shape).astype(input.dtype)
    return ConvLSTMNDState(zeros, zeros)


@tree_state.def_state(ConvGRUNDCell)
def _(cell: ConvGRUNDCell, *, input: Any) -> ConvGRUNDState:
    input = _check_rnn_cell_tree_state_input(cell, input)
    shape = (cell.hidden_features, *input.shape[1:])
    return ConvGRUNDState(jnp.zeros(shape).astype(input.dtype))
