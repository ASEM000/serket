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
import operator as op
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.lax import ConvDimensionNumbers

import serket as sk
from serket.nn.initialization import InitType, resolve_init_func
from serket.nn.utils import (
    DilationType,
    KernelSizeType,
    PaddingType,
    StridesType,
    calculate_convolution_output_shape,
    calculate_transpose_padding,
    canonicalize,
    delayed_canonicalize_padding,
    positive_int_cb,
    validate_axis_shape,
    validate_spatial_ndim,
)


@ft.lru_cache(maxsize=None)
def generate_conv_dim_numbers(spatial_ndim):
    return ConvDimensionNumbers(*((tuple(range(spatial_ndim + 2)),) * 3))


class ConvND(sk.TreeClass):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: KernelSizeType,
        *,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        input_dilation: DilationType = 1,
        kernel_dilation: DilationType = 1,
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        groups: int = 1,
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        self.in_features = positive_int_cb(in_features)
        self.out_features = positive_int_cb(out_features)
        self.kernel_size = canonicalize(
            kernel_size,
            self.spatial_ndim,
            name="kernel_size",
        )
        self.strides = canonicalize(strides, self.spatial_ndim, name="strides")
        self.padding = padding  # delayed canonicalization
        self.input_dilation = canonicalize(
            input_dilation,
            self.spatial_ndim,
            name="input_dilation",
        )
        self.kernel_dilation = canonicalize(
            kernel_dilation,
            self.spatial_ndim,
            name="kernel_dilation",
        )

        weight_init_func = resolve_init_func(weight_init_func)
        bias_init_func = resolve_init_func(bias_init_func)

        self.groups = positive_int_cb(groups)

        if self.out_features % self.groups != 0:
            raise ValueError(f"{(out_features % groups == 0)=}")

        weight_shape = (out_features, in_features // groups, *self.kernel_size)
        self.weight = weight_init_func(key, weight_shape)

        bias_shape = (out_features, *(1,) * self.spatial_ndim)
        self.bias = bias_init_func(key, bias_shape)

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
            lhs_dilation=self.input_dilation,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=generate_conv_dim_numbers(self.spatial_ndim),
            feature_group_count=self.groups,
        )

        if self.bias is None:
            return jnp.squeeze(x, 0)
        return jnp.squeeze((x + self.bias), 0)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the convolutional layer."""
        ...


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

        input_dilation: dilation of the input. accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        kernel_dilation: Dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init_func: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``0``.

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

        input_dilation: Dilation of the input. accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        kernel_dilation: dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init_func: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``0``.

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

        input_dilation: Dilation of the input. accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        kernel_dilation: dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init_func: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``0``.

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

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class ConvNDTranspose(sk.TreeClass):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: KernelSizeType,
        *,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        output_padding: int = 0,
        kernel_dilation: DilationType = 1,
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        groups: int = 1,
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        self.in_features = positive_int_cb(in_features)
        self.out_features = positive_int_cb(out_features)
        self.kernel_size = canonicalize(
            kernel_size, self.spatial_ndim, name="kernel_size"
        )
        self.strides = canonicalize(strides, self.spatial_ndim, name="strides")
        self.padding = padding  # delayed canonicalization
        self.output_padding = canonicalize(
            output_padding,
            self.spatial_ndim,
            name="output_padding",
        )
        self.kernel_dilation = canonicalize(
            kernel_dilation,
            self.spatial_ndim,
            name="kernel_dilation",
        )
        weight_init_func = resolve_init_func(weight_init_func)
        bias_init_func = resolve_init_func(bias_init_func)
        self.groups = positive_int_cb(groups)

        if self.out_features % self.groups != 0:
            raise ValueError(f"{(self.out_features % self.groups ==0)=}")

        weight_shape = (out_features, in_features // groups, *self.kernel_size)  # OIHW
        self.weight = weight_init_func(key, weight_shape)

        bias_shape = (out_features, *(1,) * self.spatial_ndim)
        self.bias = bias_init_func(key, bias_shape)

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
            extra_padding=self.output_padding,
            kernel_size=self.kernel_size,
            input_dilation=self.kernel_dilation,
        )

        y = jax.lax.conv_transpose(
            lhs=jnp.expand_dims(x, 0),
            rhs=self.weight,
            strides=self.strides,
            padding=transposed_padding,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=generate_conv_dim_numbers(self.spatial_ndim),
        )

        if self.bias is None:
            return jnp.squeeze(y, 0)
        return jnp.squeeze(y + self.bias, 0)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the convolutional layer."""
        ...


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

        output_padding: padding of the output after convolution. accepts:

            - single integer for same padding in all dimensions.

        kernel_dilation: dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init_func: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``0``.

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

        output_padding: padding of the output after convolution. accepts:

            - single integer for same padding in all dimensions.

        kernel_dilation: dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init_func: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``0``.

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

        output_padding: padding of the output after convolution. accepts:

            - single integer for same padding in all dimensions.

        kernel_dilation: dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init_func: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``0``.

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

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class DepthwiseConvND(sk.TreeClass):
    def __init__(
        self,
        in_features: int,
        kernel_size: KernelSizeType,
        *,
        depth_multiplier: int = 1,
        strides: int = 1,
        padding: PaddingType = "SAME",
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        self.in_features = positive_int_cb(in_features)
        self.kernel_size = canonicalize(
            kernel_size, self.spatial_ndim, name="kernel_size"
        )
        self.depth_multiplier = positive_int_cb(depth_multiplier)
        self.strides = canonicalize(strides, self.spatial_ndim, name="strides")
        self.padding = padding  # delayed canonicalization
        self.input_dilation = canonicalize(1, self.spatial_ndim, name="input_dilation")
        self.kernel_dilation = canonicalize(
            1,
            self.spatial_ndim,
            name="kernel_dilation",
        )
        weight_init_func = resolve_init_func(weight_init_func)
        bias_init_func = resolve_init_func(bias_init_func)

        weight_shape = (depth_multiplier * in_features, 1, *self.kernel_size)  # OIHW
        self.weight = weight_init_func(key, weight_shape)

        bias_shape = (depth_multiplier * in_features, *(1,) * self.spatial_ndim)
        self.bias = bias_init_func(key, bias_shape)

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        padding = delayed_canonicalize_padding(
            in_dim=x.shape[1:],
            padding=self.padding,
            kernel_size=self.kernel_size,
            strides=self.strides,
        )

        y = jax.lax.conv_general_dilated(
            lhs=jnp.expand_dims(x, axis=0),
            rhs=self.weight,
            window_strides=self.strides,
            padding=padding,
            lhs_dilation=self.input_dilation,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=generate_conv_dim_numbers(self.spatial_ndim),
            feature_group_count=self.in_features,
        )

        if self.bias is None:
            return jnp.squeeze(y, 0)
        return jnp.squeeze((y + self.bias), 0)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the convolutional layer."""
        ...


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

        weight_init_func: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``0``.


    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.DepthwiseConv1D(3, 3, depth_multiplier=2, strides=2)
        >>> l1(jnp.ones((3, 32))).shape
        (6, 16)

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

        weight_init_func: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``0``.


    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.DepthwiseConv2D(3, 3, depth_multiplier=2, strides=2)
        >>> l1(jnp.ones((3, 32, 32))).shape
        (6, 16, 16)

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

        weight_init_func: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``0``.


    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.DepthwiseConv3D(3, 3, depth_multiplier=2, strides=2)
        >>> l1(jnp.ones((3, 32, 32, 32))).shape
        (6, 16, 16, 16)

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class SeparableConvND(sk.TreeClass):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: KernelSizeType,
        *,
        depth_multiplier: int = 1,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        depthwise_weight_init_func: InitType = "glorot_uniform",
        pointwise_weight_init_func: InitType = "glorot_uniform",
        pointwise_bias_init_func: InitType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        self.depthwise_conv = self.depthwise_convolution_layer(
            in_features=in_features,
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            weight_init_func=depthwise_weight_init_func,
            bias_init_func=None,  # no bias for lhs
            key=key,
        )

        self.pointwise_conv = self.pointwise_convolution_layer(
            in_features=in_features * depth_multiplier,
            out_features=out_features,
            kernel_size=1,
            strides=strides,
            padding=padding,
            weight_init_func=pointwise_weight_init_func,
            bias_init_func=pointwise_bias_init_func,
            key=key,
        )

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
    def pointwise_convolution_layer(self):
        ...

    @property
    @abc.abstractmethod
    def depthwise_convolution_layer(self):
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

        weight_init_func: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``0``.


    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.SeparableConv1D(3, 3, 3, depth_multiplier=2)
        >>> l1(jnp.ones((3, 32))).shape
        (3, 32)

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 1

    @property
    def pointwise_convolution_layer(self):
        return Conv1D

    @property
    def depthwise_convolution_layer(self):
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

        weight_init_func: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``0``.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.SeparableConv2D(3, 3, 3, depth_multiplier=2)
        >>> l1(jnp.ones((3, 32, 32))).shape
        (3, 32, 32)

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 2

    @property
    def pointwise_convolution_layer(self):
        return Conv2D

    @property
    def depthwise_convolution_layer(self):
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

        weight_init_func: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``0``.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.SeparableConv3D(3, 3, 3, depth_multiplier=2)
        >>> l1(jnp.ones((3, 32, 32, 32))).shape
        (3, 32, 32, 32)

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 3

    @property
    def pointwise_convolution_layer(self):
        return Conv3D

    @property
    def depthwise_convolution_layer(self):
        return DepthwiseConv3D


class ConvNDLocal(sk.TreeClass):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: KernelSizeType,
        *,
        in_size: Sequence[int],
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        input_dilation: DilationType = 1,
        kernel_dilation: DilationType = 1,
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        # checked by callbacks
        self.in_features = positive_int_cb(in_features)
        self.out_features = positive_int_cb(out_features)
        self.kernel_size = canonicalize(
            kernel_size, self.spatial_ndim, name="kernel_size"
        )
        self.in_size = canonicalize(in_size, self.spatial_ndim, name="in_size")
        self.strides = canonicalize(strides, self.spatial_ndim, name="strides")
        self.padding = delayed_canonicalize_padding(
            self.in_size,
            padding,
            self.kernel_size,
            self.strides,
        )
        self.input_dilation = canonicalize(
            input_dilation,
            self.spatial_ndim,
            name="input_dilation",
        )
        self.kernel_dilation = canonicalize(
            kernel_dilation,
            self.spatial_ndim,
            name="kernel_dilation",
        )
        weight_init_func = resolve_init_func(weight_init_func)
        bias_init_func = resolve_init_func(bias_init_func)

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

        self.weight = weight_init_func(key, weight_shape)

        bias_shape = (self.out_features, *out_size)
        self.bias = bias_init_func(key, bias_shape)

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        y = jax.lax.conv_general_dilated_local(
            lhs=jnp.expand_dims(x, 0),
            rhs=self.weight,
            window_strides=self.strides,
            padding=self.padding,
            filter_shape=self.kernel_size,
            lhs_dilation=self.kernel_dilation,
            rhs_dilation=self.input_dilation,  # atrous dilation
            dimension_numbers=generate_conv_dim_numbers(self.spatial_ndim),
        )

        if self.bias is None:
            return jnp.squeeze(y, 0)
        return jnp.squeeze((y + self.bias), 0)

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

        weight_init_func: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``0``.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.Conv1DLocal(3, 3, 3, in_size=(32,))
        >>> l1(jnp.ones((3, 32))).shape
        (3, 32)

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

        weight_init_func: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``0``.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.Conv2DLocal(3, 3, 3, in_size=(32, 32))
        >>> l1(jnp.ones((3, 32, 32))).shape
        (3, 32, 32)

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

        weight_init_func: Function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: Function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``0``.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.Conv3DLocal(3, 3, 3, in_size=(32, 32, 32))
        >>> l1(jnp.ones((3, 32, 32, 32))).shape
        (3, 32, 32, 32)

    Reference:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 3
