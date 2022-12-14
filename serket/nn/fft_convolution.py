# references
# https://github.com/fkodom/fft-conv-pytorch/blob/master/fft_conv_pytorch/fft_conv.py
# https://stackoverflow.com/questions/47272699/need-tensorflow-keras-equivalent-for-scipy-signal-fftconvolve

from __future__ import annotations

import dataclasses
import functools as ft
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from serket.nn.utils import (
    _calculate_transpose_padding,
    _check_and_return_init_func,
    _check_and_return_kernel,
    _check_and_return_kernel_dilation,
    _check_and_return_padding,
    _check_and_return_positive_int,
    _check_and_return_strides,
    _check_in_features,
    _check_non_tracer,
    _check_spatial_in_shape,
)


@jax.jit
def _ungrouped_matmul(x, y) -> jnp.ndarray:
    alpha = "abcdefghijklmnopqrstuvwx"
    lhs = "y" + alpha[: x.ndim - 1]
    rhs = "z" + alpha[: y.ndim - 1]
    out = "yz" + lhs[2:]
    return jnp.einsum(f"{lhs},{rhs}->{out}", x, y)


@ft.partial(jax.jit, static_argnums=(2,))
def _grouped_matmul(x, y, groups) -> jnp.ndarray:
    b, c, *s = x.shape  # batch, channels, spatial
    o, i, *k = y.shape  # out_channels, in_channels, kernel
    x = x.reshape(groups, b, c // groups, *s)  # groups, batch, channels, spatial
    y = y.reshape(groups, o // groups, *(i, *k))
    z = jax.vmap(_ungrouped_matmul, in_axes=(0, 0), out_axes=1)(x, y)
    return z.reshape(z.shape[0], z.shape[1] * z.shape[2], *z.shape[3:])


def grouped_matmul(x, y, groups: int = 1):
    return _ungrouped_matmul(x, y) if groups == 1 else _grouped_matmul(x, y, groups)


@ft.partial(jax.jit, static_argnums=(1, 2))
def _intersperse_along_axis(
    x: jnp.ndarray, dilation: int, axis: int, value: int | float = 0
) -> jnp.ndarray:
    if dilation == 1:
        return x

    shape = list(x.shape)
    shape[axis] = (dilation) * shape[axis] - (dilation - 1)
    z = jnp.ones(shape) * value
    z = z.at[(slice(None),) * axis + (slice(None, None, (dilation)),)].set(x)
    return z


@ft.partial(jax.jit, static_argnums=(1, 2))
def _general_intersperse(
    x: jnp.ndarray,
    dilation: tuple[int, ...],
    axis: tuple[int, ...],
    value: int | float = 0,
) -> jnp.ndarray:
    for di, ai in zip(dilation, axis):
        x = _intersperse_along_axis(x, di, ai)
    return x


@ft.partial(jax.jit, static_argnums=(1,))
def _general_pad(x: jnp.ndarray, pad_width: tuple[[int, int], ...]) -> jnp.ndarray:
    """Pad the input with `pad_width` on each side. Negative value will lead to cropping.
    Example:
        >>> _general_pad(jnp.ones([3,3]),((0,0),(-1,1)))
        [[1., 1., 0.],
        [1., 1., 0.],
        [1., 1., 0.]]
    """
    pad_width = list(pad_width)

    for i, (l, r) in enumerate(pad_width):
        if l < 0:
            x = jax.lax.dynamic_slice_in_dim(x, -l, x.shape[i] + l, i)
            pad_width[i] = (0, r)

        if r < 0:
            x = jax.lax.dynamic_slice_in_dim(x, 0, x.shape[i] + r, i)
            pad_width[i] = (l, 0)

    return jnp.pad(x, pad_width)


@ft.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def fft_conv_general_dilated(
    x: jnp.ndarray,
    w: jnp.ndarray,
    strides: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
    groups: int,
    dilation: tuple[int, ...],
) -> jnp.ndarray:
    """General dilated convolution using FFT
    Args:
        x: input array in shape (batch, in_features, *spatial_in_shape)
        w: kernel array in shape of (out_features, in_features // groups, *kernel_size)
        strides: strides in form of tuple of ints for each spatial dimension
        padding: padding in the form of ((before_1, after_1), ..., (before_N, after_N)) for each spatial dimension
        groups: number of groups
        dilation: dilation in the form of tuple of ints for each spatial dimension
    """

    spatial_ndim = x.ndim - 2  # spatial dimensions
    w = _general_intersperse(w, dilation=dilation, axis=range(2, 2 + spatial_ndim))
    x = _general_pad(x, ((0, 0), (0, 0), *padding))

    x_shape, w_shape = x.shape, w.shape

    if x.shape[-1] % 2 != 0:
        x = jnp.pad(x, tuple([(0, 0)] * (x.ndim - 1) + [(0, 1)]))

    kernel_padding = ((0, x.shape[i] - w.shape[i]) for i in range(2, spatial_ndim + 2))
    w = _general_pad(w, ((0, 0), (0, 0), *kernel_padding))

    x_fft = jnp.fft.rfftn(x, axes=range(2, spatial_ndim + 2))
    w_fft = jnp.conjugate(jnp.fft.rfftn(w, axes=range(2, spatial_ndim + 2)))
    z_fft = grouped_matmul(x_fft, w_fft, groups)

    z = jnp.fft.irfftn(z_fft, axes=range(2, spatial_ndim + 2))

    start = (0,) * (spatial_ndim + 2)
    end = (z.shape[0], z.shape[1])
    end += tuple(max((x_shape[i] - w_shape[i] + 1), 0) for i in range(2, spatial_ndim + 2))  # fmt: skip

    if all(s == 1 for s in strides):
        return jax.lax.dynamic_slice(z, start, end)

    return jax.lax.slice(z, start, end, (1, 1, *strides))


@pytc.treeclass
class FFTConvND:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int = pytc.field(nondiff=True)
    out_features: int = pytc.field(nondiff=True)
    kernel_size: int | tuple[int, ...] = pytc.field(nondiff=True)
    strides: int | tuple[int, ...] = pytc.field(nondiff=True)
    padding: str | int | tuple[int, ...] | tuple[tuple[int, int], ...] = pytc.field(nondiff=True)  # fmt: skip
    kernel_dilation: int | tuple[int, ...] = pytc.field(nondiff=True)
    weight_init_func: str | Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]
    bias_init_func: str | Callable[[jr.PRNGKey, tuple[int]], jnp.ndarray]
    groups: int = pytc.field(nondiff=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        strides: int | tuple[int, ...] = 1,
        padding: str | int | tuple[int, ...] | tuple[tuple[int, int], ...] = "SAME",
        kernel_dilation: int | tuple[int, ...] = 1,
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable = "zeros",
        groups: int = 1,
        spatial_ndim: int = 2,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """FFT Convolutional layer.

        Args:
            in_features: number of input features
            out_features: number of output features
            kernel_size: size of the convolutional kernel
            strides: stride of the convolution
            padding: padding of the input
            kernel_dilation: dilation of the kernel
            weight_init_func: function to use for initializing the weights
            bias_init_func: function to use for initializing the bias
            groups: number of groups to use for grouped convolution
            spatial_ndim: number of dimensions of the convolution
            key: key to use for initializing the weights

        See:
            https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
            The implementation is tested against https://github.com/fkodom/fft-conv-pytorch
        """
        if in_features is None:
            for field_item in dataclasses.fields(self):
                # set all fields to None to mark the class as uninitialized
                # to the user and to avoid errors
                setattr(self, field_item.name, None)

            self._init = ft.partial(
                FFTConvND.__init__,
                self=self,
                out_features=out_features,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                kernel_dilation=kernel_dilation,
                weight_init_func=weight_init_func,
                bias_init_func=bias_init_func,
                groups=groups,
                spatial_ndim=spatial_ndim,
                key=key,
            )

            return

        if hasattr(self, "_init"):
            delattr(self, "_init")

        self.in_features = _check_and_return_positive_int(in_features, "in_features")
        self.out_features = _check_and_return_positive_int(out_features, "out_features")
        self.groups = _check_and_return_positive_int(groups, "groups")
        self.spatial_ndim = _check_and_return_positive_int(spatial_ndim, "spatial_ndim")

        msg = f"Expected out_features % groups == 0, got {self.out_features % self.groups}"
        assert self.out_features % self.groups == 0, msg

        self.kernel_size = _check_and_return_kernel(kernel_size, spatial_ndim)
        self.strides = _check_and_return_strides(strides, spatial_ndim)

        self.padding = _check_and_return_padding(padding, self.kernel_size)
        self.kernel_dilation = _check_and_return_kernel_dilation(
            kernel_dilation, spatial_ndim
        )
        self.weight_init_func = _check_and_return_init_func(weight_init_func, "weight_init_func")  # fmt: skip
        self.bias_init_func = _check_and_return_init_func(bias_init_func, "bias_init_func")  # fmt: skip

        weight_shape = (out_features, in_features // groups, *self.kernel_size)
        self.weight = self.weight_init_func(key, weight_shape)

        if bias_init_func is None:
            self.bias = None
        else:
            bias_shape = (out_features, *(1,) * spatial_ndim)
            self.bias = self.bias_init_func(key, bias_shape)

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        if hasattr(self, "_init"):
            _check_non_tracer(x, name=self.__class__.__name__)
            getattr(self, "_init")(in_features=x.shape[0])
        _check_spatial_in_shape(x, self.spatial_ndim)
        _check_in_features(x, self.in_features)

        y = fft_conv_general_dilated(
            jnp.expand_dims(x, axis=0),
            self.weight,
            strides=self.strides,
            padding=self.padding,
            groups=self.groups,
            dilation=self.kernel_dilation,
        )
        y = jnp.squeeze(y, axis=0)
        if self.bias is None:
            return y
        return y + self.bias


@pytc.treeclass
class FFTConvNDTranspose:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int = pytc.field(nondiff=True)
    out_features: int = pytc.field(nondiff=True)
    kernel_size: int | tuple[int, ...] = pytc.field(nondiff=True)
    padding: str | int | tuple[int, ...] | tuple[tuple[int, int], ...] = pytc.field(nondiff=True)  # fmt: skip
    output_padding: int | tuple[int, ...] = pytc.field(nondiff=True)
    strides: int | tuple[int, ...] = pytc.field(nondiff=True)
    kernel_dilation: int | tuple[int, ...] = pytc.field(nondiff=True)
    weight_init_func: str | Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]
    bias_init_func: Callable[[jr.PRNGKey, tuple[int]], jnp.ndarray]
    groups: int = pytc.field(nondiff=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        strides: int | tuple[int, ...] = 1,
        padding: str | int | tuple[int, ...] | tuple[tuple[int, int], ...] = "SAME",
        output_padding: int = 0,
        kernel_dilation: int | tuple[int, ...] = 1,
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable = "zeros",
        groups: int = 1,
        spatial_ndim: int = 2,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """Convolutional Transpose Layer

        Args:
            in_features : Number of input channels
            out_features : Number of output channels
            kernel_size : Size of the convolutional kernel
            strides : Stride of the convolution
            padding : Padding of the input
            output_padding : Additional size added to one side of the output shape
            kernel_dilation : Dilation of the kernel
            weight_init_func : Weight initialization function
            bias_init_func : Bias initialization function
            groups : Number of groups
            spatial_ndim : Number of dimensions
            key : PRNG key
        """
        if in_features is None:
            for field_item in dataclasses.fields(self):
                # set all fields to None to mark the class as uninitialized
                # to the user and to avoid errors
                setattr(self, field_item.name, None)
            self._init = ft.partial(
                FFTConvNDTranspose.__init__,
                self=self,
                out_features=out_features,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                kernel_dilation=kernel_dilation,
                output_padding=output_padding,
                weight_init_func=weight_init_func,
                bias_init_func=bias_init_func,
                groups=groups,
                spatial_ndim=spatial_ndim,
                key=key,
            )
            return

        if hasattr(self, "_init"):
            delattr(self, "_init")

        self.in_features = _check_and_return_positive_int(in_features, "in_features")
        self.out_features = _check_and_return_positive_int(out_features, "out_features")
        self.groups = _check_and_return_positive_int(groups, "groups")
        self.spatial_ndim = _check_and_return_positive_int(spatial_ndim, "spatial_ndim")

        assert (
            self.out_features % self.groups == 0
        ), f"Expected out_features % groups == 0, got {self.out_features % self.groups}"

        self.kernel_size = _check_and_return_kernel(kernel_size, spatial_ndim)
        self.strides = _check_and_return_strides(strides, spatial_ndim)
        self.padding = _check_and_return_padding(padding, self.kernel_size)
        self.kernel_dilation = _check_and_return_kernel_dilation(
            kernel_dilation, spatial_ndim
        )
        self.output_padding = _check_and_return_strides(output_padding, spatial_ndim)
        self.weight_init_func = _check_and_return_init_func(weight_init_func, "weight_init_func")  # fmt: skip
        self.bias_init_func = _check_and_return_init_func(bias_init_func, "bias_init_func")  # fmt: skip

        weight_shape = (out_features, in_features // groups, *self.kernel_size)  # OIHW
        self.weight = self.weight_init_func(key, weight_shape)

        if bias_init_func is None:
            self.bias = None
        else:
            bias_shape = (out_features, *(1,) * spatial_ndim)
            self.bias = self.bias_init_func(key, bias_shape)

        self.transposed_padding = _calculate_transpose_padding(
            padding=self.padding,
            extra_padding=self.output_padding,
            kernel_size=self.kernel_size,
            input_dilation=self.kernel_dilation,
        )

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        if hasattr(self, "_init"):
            _check_non_tracer(x, name=self.__class__.__name__)
            getattr(self, "_init")(in_features=x.shape[0])
        _check_spatial_in_shape(x, self.spatial_ndim)
        _check_in_features(x, self.in_features)

        y = fft_conv_general_dilated(
            jnp.expand_dims(x, axis=0),
            self.weight,
            strides=self.strides,
            padding=self.transposed_padding,
            groups=self.groups,
            dilation=self.kernel_dilation,
        )
        y = jnp.squeeze(y, axis=0)
        if self.bias is None:
            return y
        return y + self.bias


@pytc.treeclass
class DepthwiseFFTConvND:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int = pytc.field(nondiff=True)  # number of input features
    kernel_size: int | tuple[int, ...] = pytc.field(nondiff=True)
    strides: int | tuple[int, ...] = pytc.field(nondiff=True)
    padding: str | int | tuple[int, ...] | tuple[tuple[int, int], ...] = pytc.field(nondiff=True)  # fmt: skip
    depth_multiplier: int = pytc.field(nondiff=True)

    weight_init_func: str | Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]
    bias_init_func: str | Callable[[jr.PRNGKey, tuple[int]], jnp.ndarray]

    def __init__(
        self,
        in_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        depth_multiplier: int = 1,
        strides: int = 1,
        padding: str | int | tuple[int, ...] | tuple[tuple[int, int], ...] = "SAME",
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable = "zeros",
        spatial_ndim: int = 2,
        key: jr.PRNGKey = jr.PRNGKey(0),
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

        Examples:
            >>> l1 = DepthwiseConvND(3, 3, depth_multiplier=2, strides=2, padding="SAME")
            >>> l1(jnp.ones((3, 32, 32))).shape
            (3, 16, 16, 6)

        Note:
            See :
                https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/
                https://github.com/google/flax/blob/main/flax/linen/linear.py
        """
        if in_features is None:
            for field_item in dataclasses.fields(self):
                setattr(self, field_item.name, None)

            self._init = ft.partial(
                DepthwiseFFTConvND.__init__,
                self=self,
                kernel_size=kernel_size,
                depth_multiplier=depth_multiplier,
                strides=strides,
                padding=padding,
                weight_init_func=weight_init_func,
                bias_init_func=bias_init_func,
                spatial_ndim=spatial_ndim,
                key=key,
            )
            return

        if hasattr(self, "_init"):
            delattr(self, "_init")

        self.in_features = _check_and_return_positive_int(in_features, "in_features")
        self.depth_multiplier = _check_and_return_positive_int(
            depth_multiplier, "in_features"
        )
        self.spatial_ndim = _check_and_return_positive_int(spatial_ndim, "spatial_ndim")

        self.kernel_size = _check_and_return_kernel(kernel_size, spatial_ndim)
        self.strides = _check_and_return_strides(strides, spatial_ndim)
        self.padding = _check_and_return_padding(padding, self.kernel_size)
        self.kernel_dilation = _check_and_return_kernel_dilation(1, spatial_ndim)
        self.weight_init_func = _check_and_return_init_func(weight_init_func, "weight_init_func")  # fmt: skip
        self.bias_init_func = _check_and_return_init_func(bias_init_func, "bias_init_func")  # fmt: skip

        weight_shape = (depth_multiplier * in_features, 1, *self.kernel_size)  # OIHW
        self.weight = self.weight_init_func(key, weight_shape)

        if bias_init_func is None:
            self.bias = None
        else:
            bias_shape = (depth_multiplier * in_features, *(1,) * spatial_ndim)
            self.bias = self.bias_init_func(key, bias_shape)

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        if hasattr(self, "_init"):
            _check_non_tracer(x, name=self.__class__.__name__)
            getattr(self, "_init")(in_features=x.shape[0])
        _check_spatial_in_shape(x, self.spatial_ndim)
        _check_in_features(x, self.in_features)

        y = fft_conv_general_dilated(
            jnp.expand_dims(x, axis=0),
            self.weight,
            strides=self.strides,
            padding=self.padding,
            groups=x.shape[0],
            dilation=self.kernel_dilation,
        )
        y = jnp.squeeze(y, axis=0)
        if self.bias is None:
            return y
        return y + self.bias


@pytc.treeclass
class SeparableFFTConvND:
    depthwise_conv: DepthwiseFFTConvND
    pointwise_conv: DepthwiseFFTConvND

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        depth_multiplier: int = 1,
        strides: int | tuple[int, ...] = 1,
        padding: str | int | tuple[int, ...] | tuple[tuple[int, int], ...] = "SAME",
        depthwise_weight_init_func: str | Callable = "glorot_uniform",
        pointwise_weight_init_func: str | Callable = "glorot_uniform",
        pointwise_bias_init_func: str | Callable = "zeros",
        spatial_ndim: int = 2,
        key: jr.PRNGKey = jr.PRNGKey(0),
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
        if in_features is None:
            for field_item in dataclasses.fields(self):
                setattr(self, field_item.name, None)
            self._init = ft.partial(
                SeparableFFTConvND.__init__,
                self=self,
                out_features=out_features,
                kernel_size=kernel_size,
                depth_multiplier=depth_multiplier,
                strides=strides,
                padding=padding,
                depthwise_weight_init_func=depthwise_weight_init_func,
                pointwise_weight_init_func=pointwise_weight_init_func,
                pointwise_bias_init_func=pointwise_bias_init_func,
                spatial_ndim=spatial_ndim,
                key=key,
            )
            return

        if hasattr(self, "_init"):
            delattr(self, "_init")

        self.in_features = _check_and_return_positive_int(in_features, "in_features")
        self.depth_multiplier = _check_and_return_positive_int(
            depth_multiplier, "in_features"
        )
        self.out_features = _check_and_return_positive_int(out_features, "out_features")
        self.spatial_ndim = _check_and_return_positive_int(spatial_ndim, "spatial_ndim")

        self.kernel_size = _check_and_return_kernel(kernel_size, spatial_ndim)
        self.strides = _check_and_return_strides(strides, spatial_ndim)
        self.padding = _check_and_return_padding(padding, self.kernel_size)
        self.depthwise_weight_init_func = _check_and_return_init_func(
            depthwise_weight_init_func, "depthwise_weight_init_func"
        )
        self.pointwise_weight_init_func = _check_and_return_init_func(
            pointwise_weight_init_func, "pointwise_weight_init_func"
        )
        self.pointwise_bias_init_func = _check_and_return_init_func(
            pointwise_bias_init_func, "pointwise_bias_init_func"
        )

        self.depthwise_conv = DepthwiseFFTConvND(
            in_features=in_features,
            depth_multiplier=depth_multiplier,
            kernel_size=self.kernel_size,
            strides=strides,
            padding=padding,
            weight_init_func=depthwise_weight_init_func,
            bias_init_func=None,  # no bias for lhs
            key=key,
            spatial_ndim=spatial_ndim,
        )

        self.pointwise_conv = FFTConvND(
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

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        if hasattr(self, "_init"):
            _check_non_tracer(x, name=self.__class__.__name__)
            getattr(self, "_init")(in_features=x.shape[0])
        _check_spatial_in_shape(x, self.spatial_ndim)
        _check_in_features(x, self.in_features)

        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


@pytc.treeclass
class FFTConv1D(FFTConvND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=1)


@pytc.treeclass
class FFTConv2D(FFTConvND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=2)


@pytc.treeclass
class FFTConv3D(FFTConvND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=3)


@pytc.treeclass
class FFTConv1DTranspose(FFTConvNDTranspose):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=1)


@pytc.treeclass
class FFTConv2DTranspose(FFTConvNDTranspose):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=2)


@pytc.treeclass
class FFTConv3DTranspose(FFTConvNDTranspose):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=3)


@pytc.treeclass
class DepthwiseFFTConv1D(DepthwiseFFTConvND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=1)


@pytc.treeclass
class DepthwiseFFTConv2D(DepthwiseFFTConvND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=2)


@pytc.treeclass
class DepthwiseFFTConv3D(DepthwiseFFTConvND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=3)


@pytc.treeclass
class SeparableFFTConv1D(SeparableFFTConvND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=1)


@pytc.treeclass
class SeparableFFTConv2D(SeparableFFTConvND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=2)


@pytc.treeclass
class SeparableFFTConv3D(SeparableFFTConvND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=3)
