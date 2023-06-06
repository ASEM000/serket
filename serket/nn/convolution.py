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
import pytreeclass as pytc
from jax.lax import ConvDimensionNumbers

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


class ConvND(pytc.TreeClass):
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
        """Convolutional layer.

        Args:
            in_features: number of input features
            out_features: number of output features
            kernel_size: size of the convolutional kernel
            strides: stride of the convolution
            padding: padding of the input
            input_dilation: dilation of the input
            kernel_dilation: dilation of the convolutional kernel
            weight_init_func: function to use for initializing the weights
            bias_init_func: function to use for initializing the bias
            groups: number of groups to use for grouped convolution
            key: key to use for initializing the weights

        Note:
            https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        """
        # already checked by the callbacks
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
        self.weight_init_func = resolve_init_func(weight_init_func)
        self.bias_init_func = resolve_init_func(bias_init_func)
        self.groups = positive_int_cb(groups)

        if self.out_features % self.groups != 0:
            raise ValueError(
                f"Expected out_features % groups == 0, \n"
                f"got {self.out_features % self.groups}"
            )

        weight_shape = (out_features, in_features // groups, *self.kernel_size)
        self.weight = self.weight_init_func(key, weight_shape)

        if bias_init_func is None:
            self.bias = None
        else:
            bias_shape = (out_features, *(1,) * self.spatial_ndim)
            self.bias = self.bias_init_func(key, bias_shape)

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
        """1D Convolutional layer.

        Args:
            in_features: number of input features
            out_features: number of output features
            kernel_size: size of the convolutional kernel
            strides: stride of the convolution
            padding: padding of the input
            input_dilation: dilation of the input
            kernel_dilation: dilation of the convolutional kernel
            weight_init_func: function to use for initializing the weights
            bias_init_func: function to use for initializing the bias
            groups: number of groups to use for grouped convolution
            key: key to use for initializing the weights

        Example:
            >>> import jax.numpy as jnp
            >>> import serket as sk
            >>> layer = sk.nn.Conv1D(in_features=1, out_features=2, kernel_size=3)
            >>> x = jnp.ones((1, 5))
            >>> print(layer(x).shape)
            (2, 5)

        Note:
            https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        """

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
            groups=groups,
            key=key,
        )

    @property
    def spatial_ndim(self) -> int:
        return 1


class Conv2D(ConvND):
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
        """2D Convolutional layer.

        Args:
            in_features: number of input features
            out_features: number of output features
            kernel_size: size of the convolutional kernel
            strides: stride of the convolution
            padding: padding of the input
            input_dilation: dilation of the input
            kernel_dilation: dilation of the convolutional kernel
            weight_init_func: function to use for initializing the weights
            bias_init_func: function to use for initializing the bias
            groups: number of groups to use for grouped convolution
            key: key to use for initializing the weights

        Example:
            >>> import jax.numpy as jnp
            >>> import serket as sk
            >>> layer = sk.nn.Conv2D(in_features=1, out_features=2, kernel_size=3)
            >>> x = jnp.ones((1, 5, 5))
            >>> print(layer(x).shape)
            (2, 5, 5)

        Note:
            https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        """

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
            groups=groups,
            key=key,
        )

    @property
    def spatial_ndim(self) -> int:
        return 2


class Conv3D(ConvND):
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
        """3D Convolutional layer.

        Args:
            in_features: number of input features
            out_features: number of output features
            kernel_size: size of the convolutional kernel
            strides: stride of the convolution
            padding: padding of the input
            input_dilation: dilation of the input
            kernel_dilation: dilation of the convolutional kernel
            weight_init_func: function to use for initializing the weights
            bias_init_func: function to use for initializing the bias
            groups: number of groups to use for grouped convolution
            key: key to use for initializing the weights

        Example:
            >>> import jax.numpy as jnp
            >>> import serket as sk
            >>> layer = sk.nn.Conv3D(in_features=1, out_features=2, kernel_size=3)
            >>> x = jnp.ones((1, 5, 5, 5))
            >>> print(layer(x).shape)
            (2, 5, 5, 5)


        Note:
            https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        """

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
            groups=groups,
            key=key,
        )

    @property
    def spatial_ndim(self) -> int:
        return 3


# ---------------------------------------------------------------------------- #


class ConvNDTranspose(pytc.TreeClass):
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
        """Convolutional Transpose Layer

        Args:
            in_features : Number of input channels
            out_features : Number of output channels
            kernel_size : Size of the convolutional kernel
            strides : Stride of the convolution
            padding : Padding of the input
            output_padding : Additional size added to one side of the output shape
            kernel_dilation : Dilation of the convolutional kernel
            weight_init_func : Weight initialization function
            bias_init_func : Bias initialization function
            groups : Number of groups
            key : PRNG key
        """
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
        self.weight_init_func = resolve_init_func(weight_init_func)
        self.bias_init_func = resolve_init_func(bias_init_func)
        self.groups = positive_int_cb(groups)

        if self.out_features % self.groups != 0:
            raise ValueError(
                "Expected out_features % groups == 0,"
                f"got {self.out_features % self.groups}"
            )

        weight_shape = (out_features, in_features // groups, *self.kernel_size)  # OIHW
        self.weight = self.weight_init_func(key, weight_shape)

        if bias_init_func is None:
            self.bias = None
        else:
            bias_shape = (out_features, *(1,) * self.spatial_ndim)
            self.bias = self.bias_init_func(key, bias_shape)

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
        """1D Convolutional Transpose Layer.

        Args:
            in_features : Number of input channels
            out_features : Number of output channels
            kernel_size : Size of the convolutional kernel
            strides : Stride of the convolution
            padding : Padding of the input
            output_padding : Additional size added to one side of the output shape
            kernel_dilation : Dilation of the convolutional kernel
            weight_init_func : Weight initialization function
            bias_init_func : Bias initialization function
            groups : Number of groups
            key : PRNG key
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            groups=groups,
            key=key,
        )

    @property
    def spatial_ndim(self) -> int:
        return 1


class Conv2DTranspose(ConvNDTranspose):
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
        """2D Convolutional Transpose Layer.

        Args:
            in_features : Number of input channels
            out_features : Number of output channels
            kernel_size : Size of the convolutional kernel
            strides : Stride of the convolution
            padding : Padding of the input
            output_padding : Additional size added to one side of the output shape
            kernel_dilation : Dilation of the convolutional kernel
            weight_init_func : Weight initialization function
            bias_init_func : Bias initialization function
            groups : Number of groups
            key : PRNG key
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            groups=groups,
            key=key,
        )

    @property
    def spatial_ndim(self) -> int:
        return 2


class Conv3DTranspose(ConvNDTranspose):
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
        """3D Convolutional Transpose Layer.

        Args:
            in_features : Number of input channels
            out_features : Number of output channels
            kernel_size : Size of the convolutional kernel
            strides : Stride of the convolution
            padding : Padding of the input
            output_padding : Additional size added to one side of the output shape
            kernel_dilation : Dilation of the convolutional kernel
            weight_init_func : Weight initialization function
            bias_init_func : Bias initialization function
            groups : Number of groups
            key : PRNG key
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            groups=groups,
            key=key,
        )

    @property
    def spatial_ndim(self) -> int:
        return 3


# ---------------------------------------------------------------------------- #


class DepthwiseConvND(pytc.TreeClass):
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
        """Depthwise Convolutional layer.

        Args:
            in_features: number of input features
            kernel_size: size of the convolution kernel
            depth_multiplier : number of output channels per input channel
            strides: stride of the convolution
            padding: padding of the input
            weight_init_func: function to initialize the weights
            bias_init_func: function to initialize the bias
            key: random key for weight initialization

        Note:
            https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/
            https://github.com/google/flax/blob/main/flax/linen/linear.py
        """
        self.in_features = positive_int_cb(in_features)
        self.kernel_size = canonicalize(
            kernel_size, self.spatial_ndim, name="kernel_size"
        )
        self.depth_multiplier = positive_int_cb(depth_multiplier)
        self.strides = canonicalize(strides, self.spatial_ndim, name="strides")
        self.padding = padding  # delayed canonicalization
        self.input_dilation = canonicalize(1, self.spatial_ndim, name="input_dilation")
        self.kernel_dilation = canonicalize(
            1, self.spatial_ndim, name="kernel_dilation"
        )
        self.weight_init_func = resolve_init_func(weight_init_func)
        self.bias_init_func = resolve_init_func(bias_init_func)

        weight_shape = (depth_multiplier * in_features, 1, *self.kernel_size)  # OIHW
        self.weight = self.weight_init_func(key, weight_shape)

        if bias_init_func is None:
            self.bias = None
        else:
            bias_shape = (depth_multiplier * in_features, *(1,) * self.spatial_ndim)
            self.bias = self.bias_init_func(key, bias_shape)

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
        """1D Depthwise Convolutional layer.

        Args:
            in_features: number of input features
            kernel_size: size of the convolution kernel
            depth_multiplier : number of output channels per input channel
            strides: stride of the convolution
            padding: padding of the input
            weight_init_func: function to initialize the weights
            bias_init_func: function to initialize the bias
            key: random key for weight initialization

        Example:
            >>> l1 = DepthwiseConv1D(3, 3, depth_multiplier=2, strides=2, padding="SAME")
            >>> l1(jnp.ones((3, 32))).shape
            (6, 16)

        Note:
            https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/
            https://github.com/google/flax/blob/main/flax/linen/linear.py
        """

        super().__init__(
            in_features=in_features,
            kernel_size=kernel_size,
            depth_multiplier=depth_multiplier,
            strides=strides,
            padding=padding,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=key,
        )

    @property
    def spatial_ndim(self) -> int:
        return 1


class DepthwiseConv2D(DepthwiseConvND):
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
        """2D Depthwise Convolutional layer.

        Args:
            in_features: number of input features
            kernel_size: size of the convolution kernel
            depth_multiplier : number of output channels per input channel
            strides: stride of the convolution
            padding: padding of the input
            weight_init_func: function to initialize the weights
            bias_init_func: function to initialize the bias
            key: random key for weight initialization

        Example:
            >>> l1 = DepthwiseConv2D(3, 3, depth_multiplier=2, strides=2, padding="SAME")
            >>> l1(jnp.ones((3, 32, 32))).shape
            (6, 16, 16)

        Note:
            https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/
            https://github.com/google/flax/blob/main/flax/linen/linear.py
        """

        super().__init__(
            in_features=in_features,
            kernel_size=kernel_size,
            depth_multiplier=depth_multiplier,
            strides=strides,
            padding=padding,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=key,
        )

    @property
    def spatial_ndim(self) -> int:
        return 2


class DepthwiseConv3D(DepthwiseConvND):
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
        """3D Depthwise Convolutional layer.

        Args:
            in_features: number of input features
            kernel_size: size of the convolution kernel
            depth_multiplier : number of output channels per input channel
            strides: stride of the convolution
            padding: padding of the input
            weight_init_func: function to initialize the weights
            bias_init_func: function to initialize the bias
            key: random key for weight initialization

        Example:
            >>> l1 = DepthwiseConv3D(3, 3, depth_multiplier=2, strides=2, padding="SAME")
            >>> l1(jnp.ones((3, 32, 32, 32))).shape
            (6, 16, 16, 16)

        Note:
            https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/
            https://github.com/google/flax/blob/main/flax/linen/linear.py
        """

        super().__init__(
            in_features=in_features,
            kernel_size=kernel_size,
            depth_multiplier=depth_multiplier,
            strides=strides,
            padding=padding,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=key,
        )

    @property
    def spatial_ndim(self) -> int:
        return 3


# ---------------------------------------------------------------------------- #


class SeparableConv1D(pytc.TreeClass):
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
        """1D Separable convolutional layer.

        Args:
            in_features : Number of input channels.
            out_features : Number of output channels.
            kernel_size : Size of the convolving kernel.
            depth_multiplier : Number of depthwise convolution output channels
                for each input channel.
            strides : Stride of the convolution.
            padding : Padding to apply to the input.
            depthwise_weight_init_func : Function to initialize the depthwise
                convolution weights.
            pointwise_weight_init_func : Function to initialize the pointwise
                convolution weights.
            pointwise_bias_init_func : Function to initialize the pointwise
                convolution bias.

        Note:
            https://en.wikipedia.org/wiki/Separable_filter
            https://keras.io/api/layers/convolution_layers/separable_convolution2d/
            https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/depthwise_conv.py
        """
        self.depthwise_conv = DepthwiseConv1D(
            in_features=in_features,
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            weight_init_func=depthwise_weight_init_func,
            bias_init_func=None,  # no bias for lhs
            key=key,
        )

        self.pointwise_conv = Conv1D(
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
    def spatial_ndim(self) -> int:
        return 1


class SeparableConv2D(pytc.TreeClass):
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
        """2D Separable convolutional layer.

        Args:
            in_features : Number of input channels.
            out_features : Number of output channels.
            kernel_size : Size of the convolving kernel.
            depth_multiplier : Number of depthwise convolution output channels
                for each input channel.
            strides : Stride of the convolution.
            padding : Padding to apply to the input.
            depthwise_weight_init_func : Function to initialize the depthwise
                convolution weights.
            pointwise_weight_init_func : Function to initialize the pointwise
                convolution weights.
            pointwise_bias_init_func : Function to initialize the pointwise
                convolution bias.

        Note:
            https://en.wikipedia.org/wiki/Separable_filter
            https://keras.io/api/layers/convolution_layers/separable_convolution2d/
            https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/depthwise_conv.py
        """
        self.depthwise_conv = DepthwiseConv2D(
            in_features=in_features,
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            weight_init_func=depthwise_weight_init_func,
            bias_init_func=None,  # no bias for lhs
            key=key,
        )

        self.pointwise_conv = Conv2D(
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
    def spatial_ndim(self) -> int:
        return 2


class SeparableConv3D(pytc.TreeClass):
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
        """3D Separable convolutional layer.

        Note:
            https://en.wikipedia.org/wiki/Separable_filter
            https://keras.io/api/layers/convolution_layers/separable_convolution2d/
            https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/depthwise_conv.py

        Args:
            in_features : Number of input channels.
            out_features : Number of output channels.
            kernel_size : Size of the convolving kernel.
            depth_multiplier : Number of depthwise convolution output channels
                for each input channel.
            strides : Stride of the convolution.
            padding : Padding to apply to the input.
            depthwise_weight_init_func : Function to initialize the depthwise
                convolution weights.
            pointwise_weight_init_func : Function to initialize the pointwise
                convolution weights.
            pointwise_bias_init_func : Function to initialize the pointwise
                convolution bias.
        """
        self.depthwise_conv = DepthwiseConv3D(
            in_features=in_features,
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            weight_init_func=depthwise_weight_init_func,
            bias_init_func=None,  # no bias for lhs
            key=key,
        )

        self.pointwise_conv = Conv3D(
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
    def spatial_ndim(self) -> int:
        return 3


# ---------------------------------------------------------------------------- #


class ConvNDLocal(pytc.TreeClass):
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
        """Local convolutional layer.

        Args:
            in_features: number of input features
            out_features: number of output features
            kernel_size: size of the convolution kernel
            in_size: size of the input
            strides: stride of the convolution
            padding: padding of the convolution
            input_dilation: dilation of the input
            kernel_dilation: dilation of the convolution kernel
            weight_init_func: weight initialization function
            bias_init_func: bias initialization function
            key: random number generator key
        Note:
            https://keras.io/api/layers/locally_connected_layers/
        """
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
        self.weight_init_func = resolve_init_func(weight_init_func)
        self.bias_init_func = resolve_init_func(bias_init_func)

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

        self.weight = self.weight_init_func(key, weight_shape)

        bias_shape = (self.out_features, *out_size)

        if bias_init_func is None:
            self.bias = None
        else:
            self.bias = self.bias_init_func(key, bias_shape)

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
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """1D Local convolutional layer.

        Args:
            in_features: number of input features
            out_features: number of output features
            kernel_size: size of the convolution kernel
            in_size: size of the input
            strides: stride of the convolution
            padding: padding of the convolution
            input_dilation: dilation of the input
            kernel_dilation: dilation of the convolution kernel
            weight_init_func: weight initialization function
            bias_init_func: bias initialization function
            key: random number generator key
        Note:
            https://keras.io/api/layers/locally_connected_layers/
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            in_size=in_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=key,
        )

    @property
    def spatial_ndim(self) -> int:
        return 1


class Conv2DLocal(ConvNDLocal):
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
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """2D Local convolutional layer.

        Args:
            in_features: number of input features
            out_features: number of output features
            kernel_size: size of the convolution kernel
            in_size: size of the input
            strides: stride of the convolution
            padding: padding of the convolution
            input_dilation: dilation of the input
            kernel_dilation: dilation of the convolution kernel
            weight_init_func: weight initialization function
            bias_init_func: bias initialization function
            key: random number generator key
        Note:
            https://keras.io/api/layers/locally_connected_layers/
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            in_size=in_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=key,
        )

    @property
    def spatial_ndim(self) -> int:
        return 2


class Conv3DLocal(ConvNDLocal):
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
        weight_init_func: InitType = "glorot_uniform",
        bias_init_func: InitType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """3D Local convolutional layer.

        Args:
            in_features: number of input features
            out_features: number of output features
            kernel_size: size of the convolution kernel
            in_size: size of the input
            strides: stride of the convolution
            padding: padding of the convolution
            input_dilation: dilation of the input
            kernel_dilation: dilation of the convolution kernel
            weight_init_func: weight initialization function
            bias_init_func: bias initialization function
            key: random number generator key
        Note:
            https://keras.io/api/layers/locally_connected_layers/
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            in_size=in_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=key,
        )

    @property
    def spatial_ndim(self) -> int:
        return 3
