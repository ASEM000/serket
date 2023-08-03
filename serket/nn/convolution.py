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

"""Convolutional layers."""

from __future__ import annotations

import abc
import functools as ft
import operator as op
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.lax import ConvDimensionNumbers

import serket as sk
from serket.nn.initialization import DType, InitType, resolve_init_func
from serket.nn.utils import (
    DilationType,
    KernelSizeType,
    PaddingType,
    StridesType,
    calculate_convolution_output_shape,
    calculate_transpose_padding,
    canonicalize,
    delayed_canonicalize_padding,
    maybe_lazy_call,
    maybe_lazy_init,
    positive_int_cb,
    validate_axis_shape,
    validate_spatial_ndim,
)


@ft.lru_cache(maxsize=None)
def generate_conv_dim_numbers(spatial_ndim) -> ConvDimensionNumbers:
    return ConvDimensionNumbers(*((tuple(range(spatial_ndim + 2)),) * 3))


@ft.partial(jax.jit, inline=True)
def _ungrouped_matmul(x, y) -> jax.Array:
    alpha = "".join(map(str, range(max(x.ndim, y.ndim))))
    lhs = "a" + alpha[: x.ndim - 1]
    rhs = "b" + alpha[: y.ndim - 1]
    out = "ab" + lhs[2:]
    return jnp.einsum(f"{lhs},{rhs}->{out}", x, y)


@ft.partial(jax.jit, static_argnums=(2,), inline=True)
def _grouped_matmul(x, y, groups) -> jax.Array:
    b, c, *s = x.shape  # batch, channels, spatial
    o, i, *k = y.shape  # out_channels, in_channels, kernel
    x = x.reshape(groups, b, c // groups, *s)  # groups, batch, channels, spatial
    y = y.reshape(groups, o // groups, *(i, *k))
    z = jax.vmap(_ungrouped_matmul, in_axes=(0, 0), out_axes=1)(x, y)
    return z.reshape(z.shape[0], z.shape[1] * z.shape[2], *z.shape[3:])


def grouped_matmul(x, y, groups: int = 1):
    return _ungrouped_matmul(x, y) if groups == 1 else _grouped_matmul(x, y, groups)


@ft.partial(jax.jit, static_argnums=(1, 2), inline=True)
def _intersperse_along_axis(x: jax.Array, dilation: int, axis: int) -> jax.Array:
    shape = list(x.shape)
    shape[axis] = (dilation) * shape[axis] - (dilation - 1)
    z = jnp.zeros(shape)
    z = z.at[(slice(None),) * axis + (slice(None, None, (dilation)),)].set(x)
    return z


@ft.partial(jax.jit, static_argnums=(1, 2), inline=True)
def _general_intersperse(
    x: jax.Array,
    dilation: tuple[int, ...],
    axis: tuple[int, ...],
) -> jax.Array:
    for di, ai in zip(dilation, axis):
        x = _intersperse_along_axis(x, di, ai) if di > 1 else x
    return x


@ft.partial(jax.jit, static_argnums=(1,), inline=True)
def _general_pad(x: jax.Array, pad_width: tuple[tuple[int, int], ...]) -> jax.Array:
    """Pad the input with `pad_width` on each side. Negative value will lead to cropping.
    Example:
        >>> print(_general_pad(jnp.ones([3,3]),((0,0),(-1,1))))  # DOCTEST: +NORMALIZE_WHITESPACE
        [[1. 1. 0.]
         [1. 1. 0.]
         [1. 1. 0.]]
    """

    for axis, (lhs, rhs) in enumerate(pad_width := list(pad_width)):
        if lhs < 0 and rhs < 0:
            x = jax.lax.dynamic_slice_in_dim(x, -lhs, x.shape[axis] + lhs + rhs, axis)
        elif lhs < 0:
            x = jax.lax.dynamic_slice_in_dim(x, -lhs, x.shape[axis] + lhs, axis)
        elif rhs < 0:
            x = jax.lax.dynamic_slice_in_dim(x, 0, x.shape[axis] + rhs, axis)

    return jnp.pad(x, [(max(lhs, 0), max(rhs, 0)) for (lhs, rhs) in (pad_width)])


@ft.partial(jax.jit, static_argnums=(2, 3, 4, 5), inline=True)
def fft_conv_general_dilated(
    lhs: jax.Array,
    rhs: jax.Array,
    strides: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
    groups: int,
    dilation: tuple[int, ...],
) -> jax.Array:
    spatial_ndim = lhs.ndim - 2  # spatial dimensions
    rhs = _general_intersperse(rhs, dilation=dilation, axis=range(2, 2 + spatial_ndim))
    lhs = _general_pad(lhs, ((0, 0), (0, 0), *padding))

    x_shape, w_shape = lhs.shape, rhs.shape

    if lhs.shape[-1] % 2 != 0:
        lhs = jnp.pad(lhs, tuple([(0, 0)] * (lhs.ndim - 1) + [(0, 1)]))

    kernel_padding = (
        (0, lhs.shape[i] - rhs.shape[i]) for i in range(2, spatial_ndim + 2)
    )
    rhs = _general_pad(rhs, ((0, 0), (0, 0), *kernel_padding))

    # for real-valued input
    x_fft = jnp.fft.rfftn(lhs, axes=range(2, spatial_ndim + 2))
    w_fft = jnp.conjugate(jnp.fft.rfftn(rhs, axes=range(2, spatial_ndim + 2)))
    z_fft = grouped_matmul(x_fft, w_fft, groups)

    z = jnp.fft.irfftn(z_fft, axes=range(2, spatial_ndim + 2))

    start = (0,) * (spatial_ndim + 2)
    end = [z.shape[0], z.shape[1]]
    end += [max((x_shape[i] - w_shape[i] + 1), 0) for i in range(2, spatial_ndim + 2)]

    if all(s == 1 for s in strides):
        return jax.lax.dynamic_slice(z, start, end)

    return jax.lax.slice(z, start, end, (1, 1, *strides))


def is_lazy_call(instance, *_, **__) -> bool:
    return getattr(instance, "in_features", False) is None


def is_lazy_init(_, in_features, *__, **___) -> bool:
    return in_features is None


def infer_in_features(instance, x, *_, **__) -> int:
    return x.shape[0]


updates = dict(in_features=infer_in_features)


class BaseConvND(sk.TreeClass):
    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int | None,
        out_features: int,
        kernel_size: KernelSizeType,
        *,
        strides: StridesType = 1,
        padding: PaddingType = "same",
        dilation: DilationType = 1,
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        groups: int = 1,
        key: jr.KeyArray = jr.PRNGKey(0),
        dtype: DType = jnp.float32,
    ):
        self.in_features = positive_int_cb(in_features)
        self.out_features = positive_int_cb(out_features)
        self.kernel_size = canonicalize(kernel_size, self.spatial_ndim, "kernel_size")
        self.strides = canonicalize(strides, self.spatial_ndim, "strides")
        self.padding = padding
        self.dilation = canonicalize(dilation, self.spatial_ndim, "dilation")
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.groups = positive_int_cb(groups)

        if self.out_features % self.groups != 0:
            raise ValueError(f"{(out_features % groups == 0)=}")

        weight_shape = (out_features, in_features // groups, *self.kernel_size)
        self.weight = resolve_init_func(self.weight_init)(key, weight_shape, dtype)

        bias_shape = (out_features, *(1,) * self.spatial_ndim)
        self.bias = resolve_init_func(self.bias_init)(key, bias_shape, dtype)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the convolutional layer."""
        ...


class ConvND(BaseConvND):
    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        padding = delayed_canonicalize_padding(
            in_dim=x.shape[1:],
            padding=self.padding,
            kernel_size=self.kernel_size,
            strides=self.strides,
        )

        x = jax.lax.conv_general_dilated(
            lhs=jnp.expand_dims(x, 0),
            rhs=self.weight,
            window_strides=self.strides,
            padding=padding,
            rhs_dilation=self.dilation,
            dimension_numbers=generate_conv_dim_numbers(self.spatial_ndim),
            feature_group_count=self.groups,
        )

        if self.bias is None:
            return jnp.squeeze(x, 0)
        return jnp.squeeze((x + self.bias), 0)


class Conv1D(ConvND):
    """1D Convolutional layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        strides: stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        dilation: Dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax
        >>> layer = sk.nn.Conv1D(in_features=1, out_features=2, kernel_size=3)
        >>> # single sample
        >>> x = jnp.ones((1, 5))
        >>> print(layer(x).shape)
        (2, 5)
        >>> # batch of samples
        >>> x = jnp.ones((2, 1, 5))
        >>> print(jax.vmap(layer)(x).shape)
        (2, 2, 5)

    Note:
        :class:`.Conv1D` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.Conv1D = sk.nn.Conv1D(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.Conv1D = sk.nn.Conv1D(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 12

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class Conv2D(ConvND):
    """2D Convolutional layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        dilation: dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax
        >>> layer = sk.nn.Conv2D(in_features=1, out_features=2, kernel_size=3)
        >>> # single sample
        >>> x = jnp.ones((1, 5, 5))
        >>> print(layer(x).shape)
        (2, 5, 5)
        >>> # batch of samples
        >>> x = jnp.ones((2, 1, 5, 5))
        >>> print(jax.vmap(layer)(x).shape)
        (2, 2, 5, 5)

    Note:
        :class:`.Conv2D` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.Conv2D = sk.nn.Conv2D(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.Conv2D = sk.nn.Conv2D(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 12

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class Conv3D(ConvND):
    """3D Convolutional layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        dilation: dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax
        >>> layer = sk.nn.Conv3D(in_features=1, out_features=2, kernel_size=3)
        >>> # single sample
        >>> x = jnp.ones((1, 5, 5, 5))
        >>> print(layer(x).shape)
        (2, 5, 5, 5)
        >>> # batch of samples
        >>> x = jnp.ones((2, 1, 5, 5, 5))
        >>> print(jax.vmap(layer)(x).shape)
        (2, 2, 5, 5, 5)

    Note:
        :class:`.Conv3D` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.Conv3D = sk.nn.Conv3D(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.Conv3D = sk.nn.Conv3D(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 12

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class FFTConvND(BaseConvND):
    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        padding = delayed_canonicalize_padding(
            in_dim=x.shape[1:],
            padding=self.padding,
            kernel_size=self.kernel_size,
            strides=self.strides,
        )

        x = fft_conv_general_dilated(
            lhs=jnp.expand_dims(x, 0),
            rhs=self.weight,
            strides=self.strides,
            padding=padding,
            groups=self.groups,
            dilation=self.dilation,
        )

        if self.bias is None:
            return jnp.squeeze(x, 0)
        return jnp.squeeze((x + self.bias), 0)


class FFTConv1D(FFTConvND):
    """1D Convolutional layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        dilation: Dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax
        >>> layer = sk.nn.FFTConv1D(in_features=1, out_features=2, kernel_size=3)
        >>> # single sample
        >>> x = jnp.ones((1, 5))
        >>> print(layer(x).shape)
        (2, 5)
        >>> # batch of samples
        >>> x = jnp.ones((2, 1, 5))
        >>> print(jax.vmap(layer)(x).shape)
        (2, 2, 5)

    Note:
        :class:`.FFTConv1D` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.FFTConv1D = sk.nn.FFTConv1D(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.FFTConv1D = sk.nn.FFTConv1D(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 12

    References:
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class FFTConv2D(FFTConvND):
    """2D FFT Convolutional layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        dilation: Dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax
        >>> layer = sk.nn.FFTConv2D(in_features=1, out_features=2, kernel_size=3)
        >>> # single sample
        >>> x = jnp.ones((1, 5, 5))
        >>> print(layer(x).shape)
        (2, 5, 5)
        >>> # batch of samples
        >>> x = jnp.ones((2, 1, 5, 5))
        >>> print(jax.vmap(layer)(x).shape)
        (2, 2, 5, 5)

    Note:
        :class:`.FFTConv2D` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.FFTConv2D = sk.nn.FFTConv2D(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.FFTConv2D = sk.nn.FFTConv2D(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 12

    References:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class FFTConv3D(FFTConvND):
    """3D FFT Convolutional layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        dilation: Dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax
        >>> layer = sk.nn.FFTConv3D(in_features=1, out_features=2, kernel_size=3)
        >>> # single sample
        >>> x = jnp.ones((1, 5, 5, 5))
        >>> print(layer(x).shape)
        (2, 5, 5, 5)
        >>> # batch of samples
        >>> x = jnp.ones((2, 1, 5, 5, 5))
        >>> print(jax.vmap(layer)(x).shape)
        (2, 2, 5, 5, 5)

    Note:
        :class:`.FFTConv3D` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.FFTConv3D = sk.nn.FFTConv3D(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.FFTConv3D = sk.nn.FFTConv3D(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 12

    References:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class BaseConvNDTranspose(sk.TreeClass):
    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int | None,
        out_features: int,
        kernel_size: KernelSizeType,
        *,
        strides: StridesType = 1,
        padding: PaddingType = "same",
        out_padding: int = 0,
        dilation: DilationType = 1,
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        groups: int = 1,
        key: jr.KeyArray = jr.PRNGKey(0),
        dtype: DType = jnp.float32,
    ):
        self.in_features = positive_int_cb(in_features)
        self.out_features = positive_int_cb(out_features)
        self.kernel_size = canonicalize(kernel_size, self.spatial_ndim, "kernel_size")
        self.strides = canonicalize(strides, self.spatial_ndim, "strides")
        self.padding = padding  # delayed canonicalization
        self.out_padding = canonicalize(out_padding, self.spatial_ndim, "out_padding")
        self.dilation = canonicalize(dilation, self.spatial_ndim, "dilation")
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.groups = positive_int_cb(groups)

        if self.out_features % self.groups != 0:
            raise ValueError(f"{(self.out_features % self.groups ==0)=}")

        in_features = positive_int_cb(self.in_features)
        weight_shape = (out_features, in_features // groups, *self.kernel_size)  # OIHW
        self.weight = resolve_init_func(self.weight_init)(key, weight_shape, dtype)

        bias_shape = (out_features, *(1,) * self.spatial_ndim)
        self.bias = resolve_init_func(self.bias_init)(key, bias_shape, dtype)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the convolutional layer."""
        ...


class ConvNDTranspose(BaseConvNDTranspose):
    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        padding = delayed_canonicalize_padding(
            in_dim=x.shape[1:],
            padding=self.padding,
            kernel_size=self.kernel_size,
            strides=self.strides,
        )

        transposed_padding = calculate_transpose_padding(
            padding=padding,
            extra_padding=self.out_padding,
            kernel_size=self.kernel_size,
            input_dilation=self.dilation,
        )

        x = jax.lax.conv_transpose(
            lhs=jnp.expand_dims(x, 0),
            rhs=self.weight,
            strides=self.strides,
            padding=transposed_padding,
            rhs_dilation=self.dilation,
            dimension_numbers=generate_conv_dim_numbers(self.spatial_ndim),
        )

        if self.bias is None:
            return jnp.squeeze(x, 0)
        return jnp.squeeze(x + self.bias, 0)


class Conv1DTranspose(ConvNDTranspose):
    """1D Convolution transpose layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        out_padding: padding of the output after convolution. accepts:

            - single integer for same padding in all dimensions.

        dilation: dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax
        >>> layer = sk.nn.Conv1DTranspose(1, 2, 3)
        >>> # single sample
        >>> x = jnp.ones((1, 5))
        >>> print(layer(x).shape)
        (2, 5)
        >>> # batch of samples
        >>> x = jnp.ones((2, 1, 5))
        >>> print(jax.vmap(layer)(x).shape)
        (2, 2, 5)

    Note:
        :class:`.Conv1DTranspose` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.Conv1DTranspose = sk.nn.Conv1DTranspose(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.Conv1DTranspose = sk.nn.Conv1DTranspose(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 12

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class Conv2DTranspose(ConvNDTranspose):
    """2D Convolution transpose layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        out_padding: padding of the output after convolution. accepts:

            - single integer for same padding in all dimensions.

        dilation: dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax
        >>> layer = sk.nn.Conv2DTranspose(1, 2, 3)
        >>> # single sample
        >>> x = jnp.ones((1, 5, 5))
        >>> print(layer(x).shape)
        (2, 5, 5)
        >>> # batch of samples
        >>> x = jnp.ones((2, 1, 5, 5))
        >>> print(jax.vmap(layer)(x).shape)
        (2, 2, 5, 5)

    Note:
        :class:`.Conv2DTranspose` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.Conv2DTranspose = sk.nn.Conv2DTranspose(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.Conv2DTranspose = sk.nn.Conv2DTranspose(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 12

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class Conv3DTranspose(ConvNDTranspose):
    """3D Convolution transpose layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        out_padding: padding of the output after convolution. accepts:

            - single integer for same padding in all dimensions.

        dilation: dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax
        >>> layer = sk.nn.Conv3DTranspose(1, 2, 3)
        >>> # single sample
        >>> x = jnp.ones((1, 5, 5, 5))
        >>> print(layer(x).shape)
        (2, 5, 5, 5)
        >>> # batch of samples
        >>> x = jnp.ones((2, 1, 5, 5, 5))
        >>> print(jax.vmap(layer)(x).shape)
        (2, 2, 5, 5, 5)

    Note:
        :class:`.Conv3DTranspose` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.Conv3DTranspose = sk.nn.Conv3DTranspose(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.Conv3DTranspose = sk.nn.Conv3DTranspose(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 12

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class FFTConvNDTranspose(BaseConvNDTranspose):
    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        padding = delayed_canonicalize_padding(
            in_dim=x.shape[1:],
            padding=self.padding,
            kernel_size=self.kernel_size,
            strides=self.strides,
        )

        transposed_padding = calculate_transpose_padding(
            padding=padding,
            extra_padding=self.out_padding,
            kernel_size=self.kernel_size,
            input_dilation=self.dilation,
        )

        x = fft_conv_general_dilated(
            lhs=jnp.expand_dims(x, 0),
            rhs=self.weight,
            strides=self.strides,
            padding=transposed_padding,
            dilation=self.dilation,
            groups=1,
        )
        if self.bias is None:
            return jnp.squeeze(x, 0)
        return jnp.squeeze(x + self.bias, 0)


class FFTConv1DTranspose(FFTConvNDTranspose):
    """1D FFT Convolution transpose layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        out_padding: Padding of the output after convolution. accepts:

            - single integer for same padding in all dimensions.

        dilation: Dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax
        >>> layer = sk.nn.FFTConv1DTranspose(1, 2, 3)
        >>> # single sample
        >>> x = jnp.ones((1, 5))
        >>> print(layer(x).shape)
        (2, 5)
        >>> # batch of samples
        >>> x = jnp.ones((2, 1, 5))
        >>> print(jax.vmap(layer)(x).shape)
        (2, 2, 5)

    Note:
        :class:`.FFTConv1DTranspose` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.FFTConv1DTranspose = sk.nn.FFTConv1DTranspose(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.FFTConv1DTranspose = sk.nn.FFTConv1DTranspose(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 12

    References:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class FFTConv2DTranspose(FFTConvNDTranspose):
    """2D FFT Convolution transpose layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        out_padding: Padding of the output after convolution. accepts:

            - single integer for same padding in all dimensions.

        dilation: Dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax
        >>> layer = sk.nn.FFTConv2DTranspose(1, 2, 3)
        >>> # single sample
        >>> x = jnp.ones((1, 5, 5))
        >>> print(layer(x).shape)
        (2, 5, 5)
        >>> # batch of samples
        >>> x = jnp.ones((2, 1, 5, 5))
        >>> print(jax.vmap(layer)(x).shape)
        (2, 2, 5, 5)

    Note:
        :class:`.FFTConv2DTranspose` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.FFTConv2DTranspose = sk.nn.FFTConv2DTranspose(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.FFTConv2DTranspose = sk.nn.FFTConv2DTranspose(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 12

    References:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class FFTConv3DTranspose(FFTConvNDTranspose):
    """3D FFT Convolution transpose layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        out_padding: Padding of the output after convolution. accepts:

            - single integer for same padding in all dimensions.

        dilation: Dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax
        >>> layer = sk.nn.FFTConv3DTranspose(1, 2, 3)
        >>> # single sample
        >>> x = jnp.ones((1, 5, 5, 5))
        >>> print(layer(x).shape)
        (2, 5, 5, 5)
        >>> # batch of samples
        >>> x = jnp.ones((2, 1, 5, 5, 5))
        >>> print(jax.vmap(layer)(x).shape)
        (2, 2, 5, 5, 5)

    Note:
        :class:`.FFTConv3DTranspose` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.FFTConv3DTranspose = sk.nn.FFTConv3DTranspose(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.FFTConv3DTranspose = sk.nn.FFTConv3DTranspose(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 12

    References:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class BaseDepthwiseConvND(sk.TreeClass):
    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int | None,
        kernel_size: KernelSizeType,
        *,
        depth_multiplier: int = 1,
        strides: int = 1,
        padding: PaddingType = "same",
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
        dtype: DType = jnp.float32,
    ):
        self.in_features = positive_int_cb(in_features)
        self.kernel_size = canonicalize(kernel_size, self.spatial_ndim, "kernel_size")
        self.depth_multiplier = positive_int_cb(depth_multiplier)
        self.strides = canonicalize(strides, self.spatial_ndim, "strides")
        self.padding = padding  # delayed canonicalization
        self.weight_init = weight_init
        self.bias_init = bias_init

        weight_shape = (depth_multiplier * in_features, 1, *self.kernel_size)  # OIHW
        self.weight = resolve_init_func(self.weight_init)(key, weight_shape, dtype)

        bias_shape = (depth_multiplier * in_features, *(1,) * self.spatial_ndim)
        self.bias = resolve_init_func(self.bias_init)(key, bias_shape, dtype)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the convolutional layer."""
        ...


class DepthwiseConvND(BaseDepthwiseConvND):
    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        padding = delayed_canonicalize_padding(
            in_dim=x.shape[1:],
            padding=self.padding,
            kernel_size=self.kernel_size,
            strides=self.strides,
        )

        x = jax.lax.conv_general_dilated(
            lhs=jnp.expand_dims(x, 0),
            rhs=self.weight,
            window_strides=self.strides,
            padding=padding,
            rhs_dilation=canonicalize(1, self.spatial_ndim, "dilation"),
            dimension_numbers=generate_conv_dim_numbers(self.spatial_ndim),
            feature_group_count=self.in_features,
        )

        if self.bias is None:
            return jnp.squeeze(x, 0)
        return jnp.squeeze((x + self.bias), 0)


class DepthwiseConv1D(DepthwiseConvND):
    """1D Depthwise convolution layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        depth_multiplier: multiplier for the number of output channels. for example
            if the input has 32 channels and the depth multiplier is 2 then the
            output will have 64 channels.
        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        weight_init: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.DepthwiseConv1D(3, 3, depth_multiplier=2, strides=2)
        >>> l1(jnp.ones((3, 32))).shape
        (6, 16)

    Note:
        :class:`.DepthwiseConv1D` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.DepthwiseConv1D = sk.nn.DepthwiseConv1D(None, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.DepthwiseConv1D = sk.nn.DepthwiseConv1D(None, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 5

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class DepthwiseConv2D(DepthwiseConvND):
    """2D Depthwise convolution layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        depth_multiplier: multiplier for the number of output channels. for example
            if the input has 32 channels and the depth multiplier is 2 then the
            output will have 64 channels.
        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        weight_init: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.DepthwiseConv2D(3, 3, depth_multiplier=2, strides=2)
        >>> l1(jnp.ones((3, 32, 32))).shape
        (6, 16, 16)

    Note:
        :class:`.DepthwiseConv2D` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.DepthwiseConv2D = sk.nn.DepthwiseConv2D(None, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.DepthwiseConv2D = sk.nn.DepthwiseConv2D(None, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 5

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class DepthwiseConv3D(DepthwiseConvND):
    """3D Depthwise convolution layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.

        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        depth_multiplier: multiplier for the number of output channels. for example
            if the input has 32 channels and the depth multiplier is 2 then the
            output will have 64 channels.
        strides: stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: adding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        weight_init: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.DepthwiseConv3D(3, 3, depth_multiplier=2, strides=2)
        >>> l1(jnp.ones((3, 32, 32, 32))).shape
        (6, 16, 16, 16)

    Note:
        :class:`.DepthwiseConv3D` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.DepthwiseConv3D = sk.nn.DepthwiseConv3D(None, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.DepthwiseConv3D = sk.nn.DepthwiseConv3D(None, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 5

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class DepthwiseFFTConvND(BaseDepthwiseConvND):
    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        padding = delayed_canonicalize_padding(
            in_dim=x.shape[1:],
            padding=self.padding,
            kernel_size=self.kernel_size,
            strides=self.strides,
        )

        x = fft_conv_general_dilated(
            lhs=jnp.expand_dims(x, 0),
            rhs=self.weight,
            strides=self.strides,
            padding=padding,
            dilation=canonicalize(1, self.spatial_ndim, "dilation"),
            groups=self.in_features,
        )

        if self.bias is None:
            return jnp.squeeze(x, 0)
        return jnp.squeeze((x + self.bias), 0)


class DepthwiseFFTConv1D(DepthwiseFFTConvND):
    """1D Depthwise FFT convolution layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        depth_multiplier: multiplier for the number of output channels. for example
            if the input has 32 channels and the depth multiplier is 2 then the
            output will have 64 channels.
        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        weight_init: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.DepthwiseFFTConv1D(3, 3, depth_multiplier=2, strides=2)
        >>> l1(jnp.ones((3, 32))).shape
        (6, 16)

    Note:
        :class:`.DepthwiseFFTConv1D` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.DepthwiseFFTConv1D = sk.nn.DepthwiseFFTConv1D(None, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.DepthwiseFFTConv1D = sk.nn.DepthwiseFFTConv1D(None, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 5

    References:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class DepthwiseFFTConv2D(DepthwiseFFTConvND):
    """2D Depthwise convolution layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        kernel_size: Size of the convolutional kernel. accepts:

           - single integer for same kernel size in all dimnsions.
           - sequence of integers for different kernel sizes in each dimension.

        depth_multiplier: multiplier for the number of output channels. for example
            if the input has 32 channels and the depth multiplier is 2 then the
            output will have 64 channels.
        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        weight_init: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.DepthwiseFFTConv2D(3, 3, depth_multiplier=2, strides=2)
        >>> l1(jnp.ones((3, 32, 32))).shape
        (6, 16, 16)

    Note:
        :class:`.DepthwiseFFTConv2D` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.DepthwiseFFTConv2D = sk.nn.DepthwiseFFTConv2D(None, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.DepthwiseFFTConv2D = sk.nn.DepthwiseFFTConv2D(None, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 5

    References:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class DepthwiseFFTConv3D(DepthwiseFFTConvND):
    """3D Depthwise FFT convolution layer.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        kernel_size: Size of the convolutional kernel. accepts:

           - single integer for same kernel size in all dimnsions.
           - sequence of integers for different kernel sizes in each dimension.

        depth_multiplier: multiplier for the number of output channels. for example
            if the input has 32 channels and the depth multiplier is 2 then the
            output will have 64 channels.
        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        weight_init: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.DepthwiseFFTConv3D(3, 3, depth_multiplier=2, strides=2)
        >>> l1(jnp.ones((3, 32, 32, 32))).shape
        (6, 16, 16, 16)

    Note:
        :class:`.DepthwiseFFTConv3D` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.DepthwiseFFTConv3D = sk.nn.DepthwiseFFTConv3D(None, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.DepthwiseFFTConv3D = sk.nn.DepthwiseFFTConv3D(None, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 5
    References:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class SeparableConvND(sk.TreeClass):
    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int | None,
        out_features: int,
        kernel_size: KernelSizeType,
        *,
        depth_multiplier: int = 1,
        strides: StridesType = 1,
        padding: PaddingType = "same",
        depthwise_weight_init: InitType = "glorot_uniform",
        pointwise_weight_init: InitType = "glorot_uniform",
        pointwise_bias_init: InitType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        self.depthwise_conv = self._depthwise_convolution_layer(
            in_features=in_features,
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            weight_init=depthwise_weight_init,
            bias_init=None,  # no bias for lhs
            key=key,
        )

        self.pointwise_conv = self._pointwise_convolution_layer(
            in_features=in_features * depth_multiplier,
            out_features=out_features,
            kernel_size=1,
            strides=strides,
            padding=padding,
            weight_init=pointwise_weight_init,
            bias_init=pointwise_bias_init,
            key=key,
        )

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def _pointwise_convolution_layer(self) -> type[BaseDepthwiseConvND]:
        ...

    @property
    @abc.abstractmethod
    def _depthwise_convolution_layer(self) -> type[BaseDepthwiseConvND]:
        ...


class SeparableConv1D(SeparableConvND):
    """1D Separable convolution layer.

    Separable convolution is a depthwise convolution followed by a pointwise
    convolution. The objective is to reduce the number of parameters in the
    convolutional layer. For example, for I input features and O output features,
    and a kernel size = Ki, then standard convolution has I * O * K0 ... * Kn + O
    parameters, whereas separable convolution has I * K0 ... * Kn + I * O + O
    parameters.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        depth_multiplier: multiplier for the number of output channels. for example
            if the input has 32 channels and the depth multiplier is 2 then the
            output will have 64 channels.
        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        weight_init: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.SeparableConv1D(3, 3, 3, depth_multiplier=2)
        >>> l1(jnp.ones((3, 32))).shape
        (3, 32)

    Note:
        :class:`.SeparableConv1D` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.SeparableConv1D = sk.nn.SeparableConv1D(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.SeparableConv1D = sk.nn.SeparableConv1D(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2)))

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 1

    @property
    def _pointwise_convolution_layer(self):
        return Conv1D

    @property
    def _depthwise_convolution_layer(self):
        return DepthwiseConv1D


class SeparableConv2D(SeparableConvND):
    """2D Separable convolution layer.

    Separable convolution is a depthwise convolution followed by a pointwise
    convolution. The objective is to reduce the number of parameters in the
    convolutional layer. For example, for I input features and O output features,
    and a kernel size = Ki, then standard convolution has I * O * K0 ... * Kn + O
    parameters, whereas separable convolution has I * K0 ... * Kn + I * O + O
    parameters.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        depth_multiplier: multiplier for the number of output channels. for example
            if the input has 32 channels and the depth multiplier is 2 then the
            output will have 64 channels.
        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        weight_init: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.SeparableConv2D(3, 3, 3, depth_multiplier=2)
        >>> l1(jnp.ones((3, 32, 32))).shape
        (3, 32, 32)

    Note:
        :class:`.SeparableConv2D` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.SeparableConv2D = sk.nn.SeparableConv2D(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.SeparableConv2D = sk.nn.SeparableConv2D(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2)))

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 2

    @property
    def _pointwise_convolution_layer(self):
        return Conv2D

    @property
    def _depthwise_convolution_layer(self):
        return DepthwiseConv2D


class SeparableConv3D(SeparableConvND):
    """3D Separable convolution layer.

    Separable convolution is a depthwise convolution followed by a pointwise
    convolution. The objective is to reduce the number of parameters in the
    convolutional layer. For example, for I input features and O output features,
    and a kernel size = Ki, then standard convolution has I * O * K0 ... * Kn + O
    parameters, whereas separable convolution has I * K0 ... * Kn + I * O + O
    parameters.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        depth_multiplier: multiplier for the number of output channels. for example
            if the input has 32 channels and the depth multiplier is 2 then the
            output will have 64 channels.
        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        weight_init: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.SeparableConv3D(3, 3, 3, depth_multiplier=2)
        >>> l1(jnp.ones((3, 32, 32, 32))).shape
        (3, 32, 32, 32)

    Note:
        :class:`.SeparableConv3D` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.SeparableConv3D = sk.nn.SeparableConv3D(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.SeparableConv3D = sk.nn.SeparableConv3D(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2, 2)))

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 3

    @property
    def _pointwise_convolution_layer(self):
        return Conv3D

    @property
    def _depthwise_convolution_layer(self):
        return DepthwiseConv3D


class SeparableFFTConv1D(SeparableConvND):
    """1D Separable FFT convolution layer.

    Separable convolution is a depthwise convolution followed by a pointwise
    convolution. The objective is to reduce the number of parameters in the
    convolutional layer. For example, for I input features and O output features,
    and a kernel size = Ki, then standard convolution has I * O * K0 ... * Kn + O
    parameters, whereas separable convolution has I * K0 ... * Kn + I * O + O
    parameters.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

           - single integer for same kernel size in all dimnsions.
           - sequence of integers for different kernel sizes in each dimension.

        depth_multiplier: multiplier for the number of output channels. for example
            if the input has 32 channels and the depth multiplier is 2 then the
            output will have 64 channels.
        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        weight_init: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.SeparableFFTConv1D(3, 3, 3, depth_multiplier=2)
        >>> l1(jnp.ones((3, 32))).shape
        (3, 32)

    Note:
        :class:`.SeparableFFTConv1D` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.SeparableFFTConv1D = sk.nn.SeparableFFTConv1D(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.SeparableFFTConv1D = sk.nn.SeparableFFTConv1D(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2)))

    References:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 1

    @property
    def _pointwise_convolution_layer(self):
        return FFTConv1D

    @property
    def _depthwise_convolution_layer(self):
        return DepthwiseFFTConv1D


class SeparableFFTConv2D(SeparableConvND):
    """2D Separable FFT convolution layer.

    Separable convolution is a depthwise convolution followed by a pointwise
    convolution. The objective is to reduce the number of parameters in the
    convolutional layer. For example, for I input features and O output features,
    and a kernel size = Ki, then standard convolution has I * O * K0 ... * Kn + O
    parameters, whereas separable convolution has I * K0 ... * Kn + I * O + O
    parameters.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

           - single integer for same kernel size in all dimnsions.
           - sequence of integers for different kernel sizes in each dimension.

        depth_multiplier: multiplier for the number of output channels. for example
            if the input has 32 channels and the depth multiplier is 2 then the
            output will have 64 channels.
        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        weight_init: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.SeparableFFTConv2D(3, 3, 3, depth_multiplier=2)
        >>> l1(jnp.ones((3, 32, 32))).shape
        (3, 32, 32)

    Note:
        :class:`.SeparableFFTConv2D` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.SeparableFFTConv2D = sk.nn.SeparableFFTConv2D(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.SeparableFFTConv2D = sk.nn.SeparableFFTConv2D(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2)))

    References:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 2

    @property
    def _pointwise_convolution_layer(self):
        return FFTConv2D

    @property
    def _depthwise_convolution_layer(self):
        return DepthwiseFFTConv2D


class SeparableFFTConv3D(SeparableConvND):
    """3D Separable FFT convolution layer.

    Separable convolution is a depthwise convolution followed by a pointwise
    convolution. The objective is to reduce the number of parameters in the
    convolutional layer. For example, for I input features and O output features,
    and a kernel size = Ki, then standard convolution has I * O * K0 ... * Kn + O
    parameters, whereas separable convolution has I * K0 ... * Kn + I * O + O
    parameters.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

           - single integer for same kernel size in all dimnsions.
           - sequence of integers for different kernel sizes in each dimension.

        depth_multiplier: multiplier for the number of output channels. for example
            if the input has 32 channels and the depth multiplier is 2 then the
            output will have 64 channels.
        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        weight_init: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.SeparableFFTConv3D(3, 3, 3, depth_multiplier=2)
        >>> l1(jnp.ones((3, 32, 32, 32))).shape
        (3, 32, 32, 32)

    Note:
        :class:`.SeparableFFTConv3D` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.SeparableFFTConv3D = sk.nn.SeparableFFTConv3D(None, 12, 3, key=jr.PRNGKey(1))
        ...    l2: sk.nn.SeparableFFTConv3D = sk.nn.SeparableFFTConv3D(None, 1, 3, key=jr.PRNGKey(2))
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2, 2)))

    References:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 3

    @property
    def _pointwise_convolution_layer(self):
        return FFTConv3D

    @property
    def _depthwise_convolution_layer(self):
        return DepthwiseFFTConv3D


def infer_in_size(_, x, *__, **___) -> tuple[int, ...]:
    return x.shape[1:]


updates = {**dict(in_size=infer_in_size), **updates}


class ConvNDLocal(sk.TreeClass):
    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int | None,
        out_features: int,
        kernel_size: KernelSizeType,
        *,
        in_size: Sequence[int],
        strides: StridesType = 1,
        padding: PaddingType = "same",
        dilation: DilationType = 1,
        weight_init: InitType = "glorot_uniform",
        bias_init: InitType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
        dtype: DType = jnp.float32,
    ):
        self.in_features = positive_int_cb(in_features)
        self.out_features = positive_int_cb(out_features)
        self.kernel_size = canonicalize(kernel_size, self.spatial_ndim, "kernel_size")
        self.in_size = canonicalize(in_size, self.spatial_ndim, name="in_size")
        self.strides = canonicalize(strides, self.spatial_ndim, "strides")

        self.padding = delayed_canonicalize_padding(
            self.in_size,
            padding,
            self.kernel_size,
            self.strides,
        )

        self.dilation = canonicalize(dilation, self.spatial_ndim, "dilation")
        self.weight_init = weight_init
        self.bias_init = bias_init

        out_size = calculate_convolution_output_shape(
            shape=self.in_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
            strides=self.strides,
        )

        # OIHW
        weight_shape = (
            self.out_features,
            self.in_features * ft.reduce(op.mul, self.kernel_size),
            *out_size,
        )

        self.weight = resolve_init_func(self.weight_init)(key, weight_shape, dtype)

        bias_shape = (self.out_features, *out_size)
        self.bias = resolve_init_func(self.bias_init)(key, bias_shape, dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        x = jax.lax.conv_general_dilated_local(
            lhs=jnp.expand_dims(x, 0),
            rhs=self.weight,
            window_strides=self.strides,
            padding=self.padding,
            filter_shape=self.kernel_size,
            rhs_dilation=self.dilation,
            dimension_numbers=generate_conv_dim_numbers(self.spatial_ndim),
        )

        if self.bias is None:
            return jnp.squeeze(x, 0)
        return jnp.squeeze((x + self.bias), 0)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the convolutional layer."""
        ...


class Conv1DLocal(ConvNDLocal):
    """1D Local convolutional layer.

    Local convolutional layer is a convolutional layer where the convolution
    kernel is applied to a local region of the input. The kernel weights are
    *not* shared across the spatial dimensions of the input.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        in_size: the size of the spatial dimensions of the input. e.g excluding
            the first dimension. accepts a sequence of integer(s).
        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        weight_init: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.Conv1DLocal(3, 3, 3, in_size=(32,))
        >>> l1(jnp.ones((3, 32))).shape
        (3, 32)

    Note:
        :class:`.Conv1DLocal` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> k1, k2 = jr.split(jr.PRNGKey(0))
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.Conv1DLocal = sk.nn.Conv1DLocal(None, 12, 3, in_size=None, key=k1)
        ...    l2: sk.nn.Conv1DLocal = sk.nn.Conv1DLocal(None, 5, 3, in_size=None, key=k2)
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 12

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class Conv2DLocal(ConvNDLocal):
    """2D Local convolutional layer.

    Local convolutional layer is a convolutional layer where the convolution
    kernel is applied to a local region of the input. This means that the kernel
    weights are *not* shared across the spatial dimensions of the input.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        in_size: the size of the spatial dimensions of the input. e.g excluding
            the first dimension. accepts a sequence of integer(s).
        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        weight_init: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.Conv2DLocal(3, 3, 3, in_size=(32, 32))
        >>> l1(jnp.ones((3, 32, 32))).shape
        (3, 32, 32)

    Note:
        :class:`.Conv2DLocal` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> k1, k2 = jr.split(jr.PRNGKey(0))
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.Conv2DLocal = sk.nn.Conv2DLocal(None, 12, 3, in_size=None, key=k1)
        ...    l2: sk.nn.Conv2DLocal = sk.nn.Conv2DLocal(None, 5, 3, in_size=None, key=k2)
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 12

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class Conv3DLocal(ConvNDLocal):
    """3D Local convolutional layer.

    Local convolutional layer is a convolutional layer where the convolution
    kernel is applied to a local region of the input. This means that the kernel
    weights are *not* shared across the spatial dimensions of the input.

    Args:
        in_features: Number of input feature maps, for 1D convolution this is the
            length of the input, for 2D convolution this is the number of input
            channels, for 3D convolution this is the number of input channels.
        out_features: Number of output features maps, for 1D convolution this is
            the length of the output, for 2D convolution this is the number of
            output channels, for 3D convolution this is the number of output
            channels.
        kernel_size: Size of the convolutional kernel. accepts:

            - single integer for same kernel size in all dimensions.
            - sequence of integers for different kernel sizes in each dimension.

        in_size: the size of the spatial dimensions of the input. e.g excluding
            the first dimension. accepts a sequence of integer(s).
        strides: Stride of the convolution. accepts:

            - single integer for same stride in all dimensions.
            - sequence of integers for different strides in each dimension.

        padding: Padding of the input before convolution. accepts:

            - single integer for same padding in all dimensions.
            - tuple of integers for different padding in each dimension.
            - tuple of a tuple of two integers for before and after padding in
              each dimension.
            - ``same``/``SAME`` for padding such that the output has the same shape
              as the input.
            - ``valid``/``VALID`` for no padding.

        weight_init: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``jax.random.PRNGKey(0)``.
        dtype: dtype of the weights. defaults to ``jax.numpy.float32``

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.Conv3DLocal(3, 3, 3, in_size=(32, 32, 32))
        >>> l1(jnp.ones((3, 32, 32, 32))).shape
        (3, 32, 32, 32)

    Note:
        :class:`.Conv3DLocal` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> k1, k2 = jr.split(jr.PRNGKey(0))
        >>> @sk.autoinit
        ... class CNN(sk.TreeClass):
        ...    l1: sk.nn.Conv3DLocal = sk.nn.Conv3DLocal(None, 12, 3, in_size=None, key=k1)
        ...    l2: sk.nn.Conv3DLocal = sk.nn.Conv3DLocal(None, 5, 3, in_size=None, key=k2)
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_cnn = CNN()
        >>> print(lazy_cnn.l1.in_features, lazy_cnn.l2.in_features)
        None None
        >>> # materialize the layer
        >>> _, materialized_cnn = lazy_cnn.at["__call__"](jnp.ones((5, 2, 2, 2)))
        >>> print(materialized_cnn.l1.in_features, materialized_cnn.l2.in_features)
        5 12

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 3
