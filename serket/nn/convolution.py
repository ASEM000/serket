# this script defines different convolutional layers
# https://arxiv.org/pdf/1603.07285.pdf
# Throughout the code, we use OIHW  as the default data format for kernels. and NCHW for data.

from __future__ import annotations

import functools as ft
import operator as op
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc
from jax.lax import ConvDimensionNumbers

from serket.nn.callbacks import (
    init_func_cb,
    positive_int_cb,
    validate_in_features,
    validate_spatial_in_shape,
)
from serket.nn.utils import (
    DilationType,
    InitFuncType,
    KernelSizeType,
    PaddingType,
    StridesType,
    calculate_convolution_output_shape,
    calculate_transpose_padding,
    canonicalize,
    delayed_canonicalize_padding,
)


@ft.lru_cache(maxsize=None)
def generate_conv_dim_numbers(spatial_ndim):
    return ConvDimensionNumbers(*((tuple(range(spatial_ndim + 2)),) * 3))


class ConvND(pytc.TreeClass):
    weight: jax.Array
    bias: jax.Array

    in_features: int = pytc.field(callbacks=[positive_int_cb])
    out_features: int = pytc.field(callbacks=[positive_int_cb])
    kernel_size: KernelSizeType
    strides: StridesType
    padding: PaddingType
    input_dilation: DilationType
    kernel_dilation: DilationType
    weight_init_func: InitFuncType = pytc.field(callbacks=[init_func_cb])
    bias_init_func: InitFuncType = pytc.field(callbacks=[init_func_cb])
    groups: int = pytc.field(callbacks=[positive_int_cb])

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        input_dilation: DilationType = 1,
        kernel_dilation: DilationType = 1,
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
        groups: int = 1,
        key: jr.KeyArray = jr.PRNGKey(0),
        spatial_ndim: int = 2,
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
            spatial_ndim: number of dimensions of the convolution
            key: key to use for initializing the weights

        See: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        """
        # already checked by the callbacks
        self.in_features = in_features
        self.out_features = out_features
        self.weight_init_func = weight_init_func
        self.bias_init_func = bias_init_func
        self.groups = groups
        self.spatial_ndim = spatial_ndim

        # needs more info to be checked
        self.kernel_size = canonicalize(kernel_size, spatial_ndim, name="kernel_size")  # fmt: skip
        self.strides = canonicalize(strides, spatial_ndim, name="strides")
        self.padding = padding  # delayed canonicalization
        self.input_dilation = canonicalize(input_dilation, spatial_ndim, name="input_dilation")  # fmt: skip
        self.kernel_dilation = canonicalize(kernel_dilation, spatial_ndim, name="kernel_dilation")  # fmt: skip

        if self.out_features % self.groups != 0:
            msg = f"Expected out_features % groups == 0, got {self.out_features % self.groups}"
            raise ValueError(msg)

        weight_shape = (out_features, in_features // groups, *self.kernel_size)
        self.weight = self.weight_init_func(key, weight_shape)

        if bias_init_func is None:
            self.bias = None
        else:
            bias_shape = (out_features, *(1,) * spatial_ndim)
            self.bias = self.bias_init_func(key, bias_shape)

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    @ft.partial(validate_in_features, attribute_name="in_features")
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
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
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

        See: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
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
            spatial_ndim=1,
        )


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
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
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

        See: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
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
            spatial_ndim=2,
        )


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
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
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


        See: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
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
            spatial_ndim=3,
        )


# ----------------------------------------------------------------------------------------------------------------------#


class ConvNDTranspose(pytc.TreeClass):
    weight: jax.Array
    bias: jax.Array

    in_features: int = pytc.field(callbacks=[positive_int_cb])
    out_features: int = pytc.field(callbacks=[positive_int_cb])
    kernel_size: KernelSizeType
    padding: PaddingType
    output_padding: DilationType
    strides: StridesType
    kernel_dilation: DilationType
    weight_init_func: InitFuncType = pytc.field(callbacks=[init_func_cb])
    bias_init_func: InitFuncType = pytc.field(callbacks=[init_func_cb])
    groups: int = pytc.field(callbacks=[positive_int_cb])

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
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
        groups: int = 1,
        key: jr.KeyArray = jr.PRNGKey(0),
        spatial_ndim: int = 2,
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
            spatial_ndim : Number of dimensions
        """
        # already checked by the callbacks
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.spatial_ndim = spatial_ndim
        self.weight_init_func = weight_init_func
        self.bias_init_func = bias_init_func

        if self.out_features % self.groups != 0:
            msg = f"Expected out_features % groups == 0, got {self.out_features % self.groups}"
            raise ValueError(msg)

        # needs more info to be checked
        self.kernel_size = canonicalize(kernel_size, spatial_ndim, "kernel_size")  # fmt: skip
        self.strides = canonicalize(strides, spatial_ndim, "strides")  # fmt: skip
        self.kernel_dilation = canonicalize(kernel_dilation, spatial_ndim, "kernel_dilation")  # fmt: skip
        self.padding = padding  # delayed canonicalization
        self.output_padding = canonicalize(output_padding, spatial_ndim, "output_padding")  # fmt: skip

        weight_shape = (out_features, in_features // groups, *self.kernel_size)  # OIHW
        self.weight = self.weight_init_func(key, weight_shape)

        if bias_init_func is None:
            self.bias = None
        else:
            bias_shape = (out_features, *(1,) * spatial_ndim)
            self.bias = self.bias_init_func(key, bias_shape)

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    @ft.partial(validate_in_features, attribute_name="in_features")
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
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
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
            spatial_ndim : Number of dimensions
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
            spatial_ndim=1,
        )


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
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
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
            spatial_ndim : Number of dimensions
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
            spatial_ndim=2,
        )


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
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
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
            spatial_ndim : Number of dimensions
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
            spatial_ndim=3,
        )


# ----------------------------------------------------------------------------------------------------------------------#


class DepthwiseConvND(pytc.TreeClass):
    weight: jax.Array
    bias: jax.Array

    in_features: int = pytc.field(callbacks=[positive_int_cb])
    kernel_size: KernelSizeType
    strides: StridesType
    padding: PaddingType
    depth_multiplier: int = pytc.field(callbacks=[positive_int_cb])

    weight_init_func: InitFuncType = pytc.field(callbacks=[init_func_cb])
    bias_init_func: InitFuncType = pytc.field(callbacks=[init_func_cb])

    def __init__(
        self,
        in_features: int,
        kernel_size: KernelSizeType,
        *,
        depth_multiplier: int = 1,
        strides: int = 1,
        padding: PaddingType = "SAME",
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
        spatial_ndim: int = 2,
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
            spatial_ndim: number of spatial dimensions
            key: random key for weight initialization

        Note:
            See :
                https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/
                https://github.com/google/flax/blob/main/flax/linen/linear.py
        """
        # already checked by the callbacks
        self.in_features = in_features
        self.depth_multiplier = depth_multiplier
        self.spatial_ndim = spatial_ndim
        self.weight_init_func = weight_init_func
        self.bias_init_func = bias_init_func

        # needs more info to be checked
        self.kernel_size = canonicalize(kernel_size, spatial_ndim, "kernel_size")  # fmt: skip
        self.strides = canonicalize(strides, spatial_ndim, "strides")  # fmt: skip
        self.input_dilation = canonicalize(1, spatial_ndim, "input_dilation")  # fmt: skip
        self.kernel_dilation = canonicalize(1, spatial_ndim, "kernel_dilation")  # fmt: skip

        self.padding = padding  # delayed canonicalization

        weight_shape = (depth_multiplier * in_features, 1, *self.kernel_size)  # OIHW
        self.weight = self.weight_init_func(key, weight_shape)

        if bias_init_func is None:
            self.bias = None
        else:
            bias_shape = (depth_multiplier * in_features, *(1,) * spatial_ndim)
            self.bias = self.bias_init_func(key, bias_shape)

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    @ft.partial(validate_in_features, attribute_name="in_features")
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


class DepthwiseConv1D(DepthwiseConvND):
    def __init__(
        self,
        in_features: int,
        kernel_size: KernelSizeType,
        *,
        depth_multiplier: int = 1,
        strides: int = 1,
        padding: PaddingType = "SAME",
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
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
            spatial_ndim: number of spatial dimensions
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
            spatial_ndim=1,
        )


class DepthwiseConv2D(DepthwiseConvND):
    def __init__(
        self,
        in_features: int,
        kernel_size: KernelSizeType,
        *,
        depth_multiplier: int = 1,
        strides: int = 1,
        padding: PaddingType = "SAME",
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
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
            spatial_ndim: number of spatial dimensions
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
            spatial_ndim=2,
        )


class DepthwiseConv3D(DepthwiseConvND):
    def __init__(
        self,
        in_features: int,
        kernel_size: KernelSizeType,
        *,
        depth_multiplier: int = 1,
        strides: int = 1,
        padding: PaddingType = "SAME",
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
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
            spatial_ndim: number of spatial dimensions
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
            spatial_ndim=3,
        )


# ----------------------------------------------------------------------------------------------------------------------#


class SeparableConvND(pytc.TreeClass):
    in_features: int = pytc.field(callbacks=[positive_int_cb])
    depthwise_conv: DepthwiseConvND
    pointwise_conv: DepthwiseConvND

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: KernelSizeType,
        *,
        depth_multiplier: int = 1,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        depthwise_weight_init_func: InitFuncType = "glorot_uniform",
        pointwise_weight_init_func: InitFuncType = "glorot_uniform",
        pointwise_bias_init_func: InitFuncType = "zeros",
        spatial_ndim: int = 2,
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """Separable convolutional layer.

        Note:
            See:
                https://en.wikipedia.org/wiki/Separable_filter
                https://keras.io/api/layers/convolution_layers/separable_convolution2d/
                https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/depthwise_conv.py

        Args:
            in_features : Number of input channels.
            out_features : Number of output channels.
            kernel_size : Size of the convolving kernel.
            depth_multiplier : Number of depthwise convolution output channels for each input channel.
            strides : Stride of the convolution.
            padding : Padding to apply to the input.
            depthwise_weight_init_func : Function to initialize the depthwise convolution weights.
            pointwise_weight_init_func : Function to initialize the pointwise convolution weights.
            pointwise_bias_init_func : Function to initialize the pointwise convolution bias.
            spatial_ndim : Number of spatial dimensions.

        """
        self.in_features = in_features
        self.depth_multiplier = canonicalize(depth_multiplier, self.in_features, "depth_multiplier")  # fmt: skip
        self.spatial_ndim = spatial_ndim

        self.depthwise_conv = DepthwiseConvND(
            in_features=in_features,
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            weight_init_func=depthwise_weight_init_func,
            bias_init_func=None,  # no bias for lhs
            key=key,
            spatial_ndim=spatial_ndim,
        )

        self.pointwise_conv = ConvND(
            in_features=in_features * depth_multiplier,
            out_features=out_features,
            kernel_size=1,
            strides=strides,
            padding=padding,
            weight_init_func=pointwise_weight_init_func,
            bias_init_func=pointwise_bias_init_func,
            key=key,
            spatial_ndim=spatial_ndim,
        )

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    @ft.partial(validate_in_features, attribute_name="in_features")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class SeparableConv1D(SeparableConvND):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: KernelSizeType,
        *,
        depth_multiplier: int = 1,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        depthwise_weight_init_func: InitFuncType = "glorot_uniform",
        pointwise_weight_init_func: InitFuncType = "glorot_uniform",
        pointwise_bias_init_func: InitFuncType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """1D Separable convolutional layer.

        Note:
            See:
                https://en.wikipedia.org/wiki/Separable_filter
                https://keras.io/api/layers/convolution_layers/separable_convolution2d/
                https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/depthwise_conv.py

        Args:
            in_features : Number of input channels.
            out_features : Number of output channels.
            kernel_size : Size of the convolving kernel.
            depth_multiplier : Number of depthwise convolution output channels for each input channel.
            strides : Stride of the convolution.
            padding : Padding to apply to the input.
            depthwise_weight_init_func : Function to initialize the depthwise convolution weights.
            pointwise_weight_init_func : Function to initialize the pointwise convolution weights.
            pointwise_bias_init_func : Function to initialize the pointwise convolution bias.
            spatial_ndim : Number of spatial dimensions.

        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            depth_multiplier=depth_multiplier,
            strides=strides,
            padding=padding,
            depthwise_weight_init_func=depthwise_weight_init_func,
            pointwise_weight_init_func=pointwise_weight_init_func,
            pointwise_bias_init_func=pointwise_bias_init_func,
            key=key,
            spatial_ndim=1,
        )


class SeparableConv2D(SeparableConvND):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: KernelSizeType,
        *,
        depth_multiplier: int = 1,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        depthwise_weight_init_func: InitFuncType = "glorot_uniform",
        pointwise_weight_init_func: InitFuncType = "glorot_uniform",
        pointwise_bias_init_func: InitFuncType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """2D Separable convolutional layer.

        Note:
            See:
                https://en.wikipedia.org/wiki/Separable_filter
                https://keras.io/api/layers/convolution_layers/separable_convolution2d/
                https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/depthwise_conv.py

        Args:
            in_features : Number of input channels.
            out_features : Number of output channels.
            kernel_size : Size of the convolving kernel.
            depth_multiplier : Number of depthwise convolution output channels for each input channel.
            strides : Stride of the convolution.
            padding : Padding to apply to the input.
            depthwise_weight_init_func : Function to initialize the depthwise convolution weights.
            pointwise_weight_init_func : Function to initialize the pointwise convolution weights.
            pointwise_bias_init_func : Function to initialize the pointwise convolution bias.
            spatial_ndim : Number of spatial dimensions.

        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            depth_multiplier=depth_multiplier,
            strides=strides,
            padding=padding,
            depthwise_weight_init_func=depthwise_weight_init_func,
            pointwise_weight_init_func=pointwise_weight_init_func,
            pointwise_bias_init_func=pointwise_bias_init_func,
            key=key,
            spatial_ndim=2,
        )


class SeparableConv3D(SeparableConvND):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: KernelSizeType,
        *,
        depth_multiplier: int = 1,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
        depthwise_weight_init_func: InitFuncType = "glorot_uniform",
        pointwise_weight_init_func: InitFuncType = "glorot_uniform",
        pointwise_bias_init_func: InitFuncType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """3D Separable convolutional layer.

        Note:
            See:
                https://en.wikipedia.org/wiki/Separable_filter
                https://keras.io/api/layers/convolution_layers/separable_convolution2d/
                https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/depthwise_conv.py

        Args:
            in_features : Number of input channels.
            out_features : Number of output channels.
            kernel_size : Size of the convolving kernel.
            depth_multiplier : Number of depthwise convolution output channels for each input channel.
            strides : Stride of the convolution.
            padding : Padding to apply to the input.
            depthwise_weight_init_func : Function to initialize the depthwise convolution weights.
            pointwise_weight_init_func : Function to initialize the pointwise convolution weights.
            pointwise_bias_init_func : Function to initialize the pointwise convolution bias.
            spatial_ndim : Number of spatial dimensions.

        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            depth_multiplier=depth_multiplier,
            strides=strides,
            padding=padding,
            depthwise_weight_init_func=depthwise_weight_init_func,
            pointwise_weight_init_func=pointwise_weight_init_func,
            pointwise_bias_init_func=pointwise_bias_init_func,
            key=key,
            spatial_ndim=3,
        )


# ----------------------------------------------------------------------------------------------------------------------#


class ConvNDLocal(pytc.TreeClass):
    weight: jax.Array
    bias: jax.Array

    in_features: int = pytc.field(callbacks=[positive_int_cb])
    out_features: int = pytc.field(callbacks=[positive_int_cb])
    kernel_size: KernelSizeType
    in_size: Sequence[int]  # size of input
    strides: StridesType
    padding: PaddingType
    input_dilation: DilationType
    kernel_dilation: DilationType
    weight_init_func: InitFuncType = pytc.field(callbacks=[init_func_cb])
    bias_init_func: InitFuncType = pytc.field(callbacks=[init_func_cb])

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
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
        spatial_ndim: int = 2,
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
            spatial_ndim: number of dimensions
            key: random number generator key
        Note:
            See : https://keras.io/api/layers/locally_connected_layers/
        """
        # checked by callbacks
        self.in_features = in_features
        self.out_features = out_features
        self.weight_init_func = weight_init_func
        self.bias_init_func = bias_init_func

        self.spatial_ndim = spatial_ndim

        # needs more info to check
        self.in_size = canonicalize(in_size, spatial_ndim, "in_size")  # fmt: skip
        self.kernel_size = canonicalize(kernel_size, spatial_ndim, "kernel_size")  # fmt: skip
        self.strides = canonicalize(strides, spatial_ndim, "strides")  # fmt: skip

        self.input_dilation = canonicalize(input_dilation, spatial_ndim, "input_dilation")  # fmt: skip
        self.kernel_dilation = canonicalize(1, spatial_ndim, "kernel_dilation")  # fmt: skip

        self.padding = delayed_canonicalize_padding(
            self.in_size,
            padding,
            self.kernel_size,
            self.strides,
        )

        self.out_size = calculate_convolution_output_shape(
            shape=self.in_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
            strides=self.strides,
        )

        # OIHW
        self.weight_shape = (
            self.out_features,
            self.in_features * ft.reduce(op.mul, self.kernel_size),
            *self.out_size,
        )

        self.weight = self.weight_init_func(key, self.weight_shape)

        bias_shape = (self.out_features, *self.out_size)

        if bias_init_func is None:
            self.bias = None
        else:
            self.bias = self.bias_init_func(key, bias_shape)

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    @ft.partial(validate_in_features, attribute_name="in_features")
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
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
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
            spatial_ndim: number of dimensions
            key: random number generator key
        Note:
            See : https://keras.io/api/layers/locally_connected_layers/
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
            spatial_ndim=1,
        )


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
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
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
            spatial_ndim: number of dimensions
            key: random number generator key
        Note:
            See : https://keras.io/api/layers/locally_connected_layers/
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
            spatial_ndim=2,
        )


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
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
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
            spatial_ndim: number of dimensions
            key: random number generator key
        Note:
            See : https://keras.io/api/layers/locally_connected_layers/
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
            spatial_ndim=3,
        )
