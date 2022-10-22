from __future__ import annotations

# import dataclasses
# import functools as ft
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from serket.nn import Linear
from serket.nn.convolution import ConvND, SeparableConvND

# from serket.nn.utils import _TRACER_ERROR_MSG

_act_func_map = {
    "tanh": jax.nn.tanh,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "hard_sigmoid": jax.nn.hard_sigmoid,
    None: lambda x: x,
}


# --------------------------------------------------- RNN ------------------------------------------------------------ #


@pytc.treeclass
class RNNState:
    hidden_state: jnp.ndarray


@pytc.treeclass
class RNNCell:
    pass


@pytc.treeclass
class SpatialRNNCell:
    pass


@pytc.treeclass
class SimpleRNNState(RNNState):
    pass


@pytc.treeclass
class SimpleRNNCell(RNNCell):
    in_to_hidden: Linear
    hidden_to_hidden: Linear

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable | None = "zeros",
        recurrent_weight_init_func: str | Callable = "orthogonal",
        act_func: str | None = "tanh",
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """Vanilla RNN cell that defines the update rule for the hidden state
        See:
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNNCell

        Args:
            in_features: the number of input features
            hidden_features: the number of hidden features
            weight_init_func: the function to use to initialize the weights
            bias_init_func: the function to use to initialize the bias
            recurrent_weight_init_func: the function to use to initialize the recurrent weights
            act_func: the activation function to use for the hidden state update
            key: the key to use to initialize the weights

        Example:
            >>> cell = SimpleRNNCell(10, 20)
            >>> cell(jnp.ones([10]), RNNState(jnp.zeros([20]), jnp.zeros([20])))
        """
        k1, k2 = jr.split(key, 2)

        if not isinstance(in_features, int) or in_features < 1:
            raise ValueError(
                f"Expected in_features to be a positive integer, got {in_features}"
            )

        if not isinstance(hidden_features, int) or hidden_features < 1:
            raise ValueError(
                f"Expected hidden_features to be a positive integer, got {hidden_features}"
            )

        self.act_func = _act_func_map[act_func]

        self.in_features = in_features
        self.hidden_features = hidden_features

        self.in_to_hidden = Linear(
            in_features,
            hidden_features,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=k1,
        )

        self.hidden_to_hidden = Linear(
            hidden_features,
            hidden_features,
            weight_init_func=recurrent_weight_init_func,
            bias_init_func=None,
            key=k2,
        )

    def __call__(
        self, x: jnp.ndarray, state: SimpleRNNState, **kwargs
    ) -> SimpleRNNState:
        msg = f"Expected state to be an instance of SimpleRNNState, got {type(state)}"
        assert isinstance(state, SimpleRNNState), msg
        h = state.hidden_state
        h = self.act_func(self.in_to_hidden(x) + self.hidden_to_hidden(h))
        return SimpleRNNState(h)

    def init_state(self) -> SimpleRNNState:
        shape = (1, self.hidden_features)
        return SimpleRNNState(jnp.zeros(shape))


@pytc.treeclass
class LSTMState(RNNState):
    cell_state: jnp.ndarray


@pytc.treeclass
class LSTMCell(RNNCell):
    in_to_hidden: Linear
    hidden_to_hidden: Linear

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable | None = "zeros",
        recurrent_weight_init_func: str | Callable = "orthogonal",
        act_func: str | None = "tanh",
        recurrent_act_func: str | None = "sigmoid",
        key: jr.PRNGKey = jr.PRNGKey(0),
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

        See:
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTMCell
            https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py
        """
        k1, k2 = jr.split(key, 2)

        if not isinstance(in_features, int) or in_features < 1:
            raise ValueError(
                f"Expected in_features to be a positive integer, got {in_features}"
            )

        if not isinstance(hidden_features, int) or hidden_features < 1:
            raise ValueError(
                f"Expected hidden_features to be a positive integer, got {hidden_features}"
            )

        self.act_func = _act_func_map[act_func]
        self.recurrent_act_func = _act_func_map[recurrent_act_func]

        self.in_features = in_features
        self.hidden_features = hidden_features

        self.in_to_hidden = Linear(
            in_features,
            hidden_features * 4,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=k1,
        )

        self.hidden_to_hidden = Linear(
            hidden_features,
            hidden_features * 4,
            weight_init_func=recurrent_weight_init_func,
            bias_init_func=None,
            key=k2,
        )

    def __call__(self, x: jnp.ndarray, state: LSTMState, **kwargs) -> LSTMState:
        msg = f"Expected state to be an instance of LSTMState, got {type(state)}"
        assert isinstance(state, LSTMState), msg
        h, c = state.hidden_state, state.cell_state

        h = self.in_to_hidden(x) + self.hidden_to_hidden(h)
        i, f, g, o = jnp.split(h, 4, axis=-1)
        i = self.recurrent_act_func(i)
        f = self.recurrent_act_func(f)
        g = self.act_func(g)
        o = self.recurrent_act_func(o)
        c = f * c + i * g
        h = o * self.act_func(c)
        return LSTMState(h, c)

    def init_state(self) -> LSTMState:
        shape = (1, self.hidden_features)
        return LSTMState(jnp.zeros(shape), jnp.zeros(shape))


# ------------------------------------------------- Spatial RNN ------------------------------------------------------ #
@pytc.treeclass
class ConvLSTMNDState(RNNState):
    cell_state: jnp.ndarray


@pytc.treeclass
class ConvLSTMNDCell(SpatialRNNCell):
    in_to_hidden: ConvND
    hidden_to_hidden: ConvND

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        strides: int | tuple[int, ...] = 1,
        padding: str | tuple[int, ...] | tuple[tuple[int, int], ...] = "SAME",
        input_dilation: int | tuple[int, ...] = 1,
        kernel_dilation: int | tuple[int, ...] = 1,
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable | None = "zeros",
        recurrent_weight_init_func: str | Callable = "orthogonal",
        act_func: str | None = "tanh",
        recurrent_act_func: str | None = "hard_sigmoid",
        key: jr.PRNGKey = jr.PRNGKey(0),
        ndim: int = 2,
    ):
        """Convolution LSTM cell that defines the update rule for the hidden state and cell state
        Args:
            in_features: Number of input features
            out_features: Number of output features
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
            ndim: Number of spatial dimensions

        See: https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D
        """
        k1, k2 = jr.split(key, 2)

        if not isinstance(in_features, int) or in_features < 1:
            raise ValueError(
                f"Expected in_features to be a positive integer, got {in_features}"
            )

        if not isinstance(out_features, int) or out_features < 1:
            raise ValueError(
                f"Expected out_features to be a positive integer, got {out_features}"
            )

        self.act_func = _act_func_map[act_func]
        self.recurrent_act_func = _act_func_map[recurrent_act_func]
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = out_features
        self.ndim = ndim

        self.in_to_hidden = ConvND(
            in_features,
            out_features * 4,
            kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=k1,
            ndim=ndim,
        )

        self.hidden_to_hidden = ConvND(
            out_features,
            out_features * 4,
            kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=recurrent_weight_init_func,
            bias_init_func=None,
            key=k2,
            ndim=ndim,
        )

    def __call__(
        self, x: jnp.ndarray, state: ConvLSTMNDState, **kwargs
    ) -> ConvLSTMNDState:
        msg = f"Expected state to be an instance of ConvLSTMNDState, got {type(state)}"
        assert isinstance(state, ConvLSTMNDState), msg
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
        msg = f"Expected spatial_dim to be a tuple of length {self.ndim}, got {spatial_dim}"
        assert len(spatial_dim) == self.ndim, msg
        shape = (self.hidden_features, *spatial_dim)
        return ConvLSTMNDState(jnp.zeros(shape), jnp.zeros(shape))


@pytc.treeclass
class ConvLSTM1DCell(ConvLSTMNDCell):
    def __init__(
        self,
        in_features: int | None,
        out_features: int,
        kernel_size: int,
        strides: int = 1,
        padding: str = "same",
        input_dilation: int = 1,
        kernel_dilation: int = 1,
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable | None = "zeros",
        recurrent_weight_init_func: str | Callable = "orthogonal",
        act_func: str | None = "tanh",
        recurrent_act_func: str | None = "sigmoid",
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
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
            ndim=1,
        )


@pytc.treeclass
class ConvLSTM2DCell(ConvLSTMNDCell):
    def __init__(
        self,
        in_features: int | None,
        out_features: int,
        kernel_size: int | tuple[int, int],
        strides: int | tuple[int, int] = 1,
        padding: str = "same",
        input_dilation: int | tuple[int, int] = 1,
        kernel_dilation: int | tuple[int, int] = 1,
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable | None = "zeros",
        recurrent_weight_init_func: str | Callable = "orthogonal",
        act_func: str | None = "tanh",
        recurrent_act_func: str | None = "sigmoid",
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
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
            ndim=2,
        )


@pytc.treeclass
class ConvLSTM3DCell(ConvLSTMNDCell):
    def __init__(
        self,
        in_features: int | None,
        out_features: int,
        kernel_size: int | tuple[int, int, int],
        strides: int | tuple[int, int, int] = 1,
        padding: str = "same",
        input_dilation: int | tuple[int, int, int] = 1,
        kernel_dilation: int | tuple[int, int, int] = 1,
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable | None = "zeros",
        recurrent_weight_init_func: str | Callable = "orthogonal",
        act_func: str | None = "tanh",
        recurrent_act_func: str | None = "sigmoid",
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
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
            ndim=3,
        )


@pytc.treeclass
class SeparableConvLSTMNDState(RNNState):
    cell_state: jnp.ndarray


@pytc.treeclass
class SeparableConvLSTMNDCell(SpatialRNNCell):
    in_to_hidden: SeparableConvND
    hidden_to_hidden: SeparableConvND

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        strides: int | tuple[int, ...] = 1,
        padding: str | tuple[int, ...] | tuple[tuple[int, int], ...] = "SAME",
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable | None = "zeros",
        recurrent_weight_init_func: str | Callable = "orthogonal",
        act_func: str | None = "tanh",
        recurrent_act_func: str | None = "hard_sigmoid",
        key: jr.PRNGKey = jr.PRNGKey(0),
        ndim: int = 2,
    ):
        """Separable Convolution LSTM cell that defines the update rule for the hidden state and cell state
        Args:
            in_features: Number of input features
            out_features: Number of output features
            kernel_size: Size of the convolutional kernel
            strides: Stride of the convolution
            padding: Padding of the convolution
            weight_init_func: Weight initialization function
            bias_init_func: Bias initialization function
            recurrent_weight_init_func: Recurrent weight initialization function
            act_func: Activation function
            recurrent_act_func: Recurrent activation function
            key: PRNG key
            ndim: Number of spatial dimensions

        See: https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D
        """
        k1, k2 = jr.split(key, 2)

        if not isinstance(in_features, int) or in_features < 1:
            raise ValueError(
                f"Expected in_features to be a positive integer, got {in_features}"
            )

        if not isinstance(out_features, int) or out_features < 1:
            raise ValueError(
                f"Expected out_features to be a positive integer, got {out_features}"
            )

        self.act_func = _act_func_map[act_func]
        self.recurrent_act_func = _act_func_map[recurrent_act_func]
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = out_features
        self.ndim = ndim

        self.in_to_hidden = SeparableConvND(
            in_features,
            out_features * 4,
            kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=1,
            depthwise_weight_init_func=weight_init_func,
            pointwise_weight_init_func=weight_init_func,
            pointwise_bias_init_func=bias_init_func,
            key=k1,
            ndim=ndim,
        )

        self.hidden_to_hidden = SeparableConvND(
            out_features,
            out_features * 4,
            kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=1,
            depthwise_weight_init_func=recurrent_weight_init_func,
            pointwise_weight_init_func=recurrent_weight_init_func,
            pointwise_bias_init_func=None,  # no bias
            key=k2,
            ndim=ndim,
        )

    def __call__(
        self, x: jnp.ndarray, state: SeparableConvLSTMNDState, **kwargs
    ) -> SeparableConvLSTMNDState:
        msg = f"Expected input to have shape (batch_size, in_features, *spatial_dims), got {x.shape}"
        assert isinstance(state, SeparableConvLSTMNDState), msg
        h, c = state.hidden_state, state.cell_state

        h = self.in_to_hidden(x) + self.hidden_to_hidden(h)
        i, f, g, o = jnp.split(h, 4, axis=0)
        i = self.recurrent_act_func(i)
        f = self.recurrent_act_func(f)
        g = self.act_func(g)
        o = self.recurrent_act_func(o)
        c = f * c + i * g
        h = o * self.act_func(c)
        return SeparableConvLSTMNDState(hidden_state=h, cell_state=c)

    def init_state(self, spatial_dim: tuple[int, ...]) -> SeparableConvLSTMNDState:
        msg = f"Expected spatial_dim to be a tuple of length {self.ndim}, got {spatial_dim}"
        assert len(spatial_dim) == self.ndim, msg
        shape = (self.hidden_features, *spatial_dim)
        return SeparableConvLSTMNDState(jnp.zeros(shape), jnp.zeros(shape))


@pytc.treeclass
class SeparableConvLSTM1DCell(SeparableConvLSTMNDCell):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        *,
        strides: int = 1,
        padding: str = "SAME",
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable | None = "zeros",
        recurrent_weight_init_func: str | Callable = "orthogonal",
        act_func: str | None = "tanh",
        recurrent_act_func: str | None = "hard_sigmoid",
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            recurrent_weight_init_func=recurrent_weight_init_func,
            act_func=act_func,
            recurrent_act_func=recurrent_act_func,
            key=key,
            ndim=1,
        )


@pytc.treeclass
class SeparableConvLSTM2DCell(SeparableConvLSTMNDCell):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, int],
        *,
        strides: int | tuple[int, int] = 1,
        padding: str
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]] = "SAME",
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable | None = "zeros",
        recurrent_weight_init_func: str | Callable = "orthogonal",
        act_func: str | None = "tanh",
        recurrent_act_func: str | None = "hard_sigmoid",
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            recurrent_weight_init_func=recurrent_weight_init_func,
            act_func=act_func,
            recurrent_act_func=recurrent_act_func,
            key=key,
            ndim=2,
        )


@pytc.treeclass
class SeparableConvLSTM3DCell(SeparableConvLSTMNDCell):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, int, int],
        *,
        strides: int | tuple[int, int, int] = 1,
        padding: str
        | tuple[int, int, int]
        | tuple[tuple[int, int, int], tuple[int, int, int]] = "SAME",
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable | None = "zeros",
        recurrent_weight_init_func: str | Callable = "orthogonal",
        act_func: str | None = "tanh",
        recurrent_act_func: str | None = "hard_sigmoid",
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            recurrent_weight_init_func=recurrent_weight_init_func,
            act_func=act_func,
            recurrent_act_func=recurrent_act_func,
            key=key,
            ndim=3,
        )


# ------------------------------------------------- Scan layer ------------------------------------------------------ #


@pytc.treeclass
class ScanRNN:
    cell: RNNCell

    def __init__(
        self,
        cell: RNNCell,
        *,
        return_sequences: bool = False,
    ):
        """Scans a RNN cell over a sequence.

        Args:
            cell: the RNN cell to use
            reverse: whether to scan the sequence in reverse
            return_sequences: whether to return the hidden state for each timestep

        Example:
            >>> cell = SimpleRNNCell(10, 20) # 10-dimensional input, 20-dimensional hidden state
            >>> rnn = ScanRNN(cell)
            >>> x = jnp.ones((5, 10)) # 5 timesteps, 10 features
            >>> result = rnn(x)  # 1 timestep, 20 features
        """
        if not isinstance(cell, (RNNCell, SpatialRNNCell)):
            msg = f"Expected cell to be an instance of RNNCell or SpatialRNNCell, got {type(cell)}"
            raise TypeError(msg)

        self.cell = cell
        self.return_sequences = return_sequences

    def __call__(self, x: jnp.ndarray, state: RNNState = None, **kwargs) -> jnp.ndarray:
        if not isinstance(state, (type(None), RNNState)):
            msg = f"Expected state to be an instance of RNNState, got {type(state)}"
            raise TypeError(msg)

        if isinstance(self.cell, SpatialRNNCell):
            # (time steps, in_features, *spatial_dims)
            msg = f"Expected x to have {self.cell.ndim + 2}"  # account for time and in_features
            msg += f"dimensions corresponds to (timesteps, in_features, *spatial_dims), got {x.ndim}"
            assert x.ndim == (self.cell.ndim + 2), msg
            msg = f"Expected x to have shape (timesteps, {self.cell.in_features}, *spatial_dims)"
            assert self.cell.in_features == x.shape[1], msg
            state = state or self.cell.init_state(spatial_dim=x.shape[2:])

        else:
            # (time steps, in_features)
            msg = f"Expected x to have 2 dimensions corresponds to (timesteps, in_features), got {x.ndim}"
            assert x.ndim == 2, msg
            msg = f"Expected x to have shape (timesteps, {self.cell.in_features})"
            assert self.cell.in_features == x.shape[1], msg
            state = state or self.cell.init_state()

        if self.return_sequences:
            # accumulate the hidden state for each timestep
            def scan_func(carry, x):
                state = self.cell(x, state=carry)
                return state, state

            return jax.lax.scan(scan_func, state, x)[1].hidden_state

        # do not accumulate the hidden state for each timestep
        scan_func = lambda carry, x: (self.cell(x, state=carry), None)
        output, _ = jax.lax.scan(scan_func, state, x)
        return output.hidden_state
