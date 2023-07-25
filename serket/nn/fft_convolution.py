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

import jax
import jax.numpy as jnp
import jax.random as jr

import serket as sk
from serket.nn.initialization import InitType, resolve_init_func
from serket.nn.utils import (
    DilationType,
    KernelSizeType,
    PaddingType,
    StridesType,
    calculate_transpose_padding,
    canonicalize,
    delayed_canonicalize_padding,
    positive_int_cb,
    validate_axis_shape,
    validate_spatial_ndim,
)


@jax.jit
def _ungrouped_matmul(x, y) -> jax.Array:
    alpha = "".join(map(str, range(max(x.ndim, y.ndim))))
    lhs = "a" + alpha[: x.ndim - 1]
    rhs = "b" + alpha[: y.ndim - 1]
    out = "ab" + lhs[2:]
    return jnp.einsum(f"{lhs},{rhs}->{out}", x, y)


@ft.partial(jax.jit, static_argnums=(2,))
def _grouped_matmul(x, y, groups) -> jax.Array:
    b, c, *s = x.shape  # batch, channels, spatial
    o, i, *k = y.shape  # out_channels, in_channels, kernel
    x = x.reshape(groups, b, c // groups, *s)  # groups, batch, channels, spatial
    y = y.reshape(groups, o // groups, *(i, *k))
    z = jax.vmap(_ungrouped_matmul, in_axes=(0, 0), out_axes=1)(x, y)
    return z.reshape(z.shape[0], z.shape[1] * z.shape[2], *z.shape[3:])


def grouped_matmul(x, y, groups: int = 1):
    return _ungrouped_matmul(x, y) if groups == 1 else _grouped_matmul(x, y, groups)


@ft.partial(jax.jit, static_argnums=(1, 2))
def _intersperse_along_axis(x: jax.Array, dilation: int, axis: int) -> jax.Array:
    shape = list(x.shape)
    shape[axis] = (dilation) * shape[axis] - (dilation - 1)
    z = jnp.zeros(shape)
    z = z.at[(slice(None),) * axis + (slice(None, None, (dilation)),)].set(x)
    return z


@ft.partial(jax.jit, static_argnums=(1, 2))
def _general_intersperse(
    x: jax.Array,
    dilation: tuple[int, ...],
    axis: tuple[int, ...],
) -> jax.Array:
    for di, ai in zip(dilation, axis):
        x = _intersperse_along_axis(x, di, ai) if di > 1 else x
    return x


@ft.partial(jax.jit, static_argnums=(1,))
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


@ft.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def fft_conv_general_dilated(
    x: jax.Array,
    w: jax.Array,
    strides: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
    groups: int,
    dilation: tuple[int, ...],
) -> jax.Array:
    """General dilated convolution using FFT
    Args:
        x: input array in shape (batch, in_features, *spatial_in_shape)
        w: kernel array in shape of (out_features, in_features // groups, *kernel_size)
        strides: strides in form of tuple of ints for each spatial dimension
        padding: padding in the form of ((before_1, after_1), ..., (before_N, after_N))
            for each spatial dimension
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

    # for real-valued input
    x_fft = jnp.fft.rfftn(x, axes=range(2, spatial_ndim + 2))
    w_fft = jnp.conjugate(jnp.fft.rfftn(w, axes=range(2, spatial_ndim + 2)))
    z_fft = grouped_matmul(x_fft, w_fft, groups)

    z = jnp.fft.irfftn(z_fft, axes=range(2, spatial_ndim + 2))

    start = (0,) * (spatial_ndim + 2)
    end = [z.shape[0], z.shape[1]]
    end += [max((x_shape[i] - w_shape[i] + 1), 0) for i in range(2, spatial_ndim + 2)]

    if all(s == 1 for s in strides):
        return jax.lax.dynamic_slice(z, start, end)

    return jax.lax.slice(z, start, end, (1, 1, *strides))


class FFTConvND(sk.TreeClass):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: KernelSizeType,
        *,
        strides: StridesType = 1,
        padding: PaddingType = "SAME",
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
            ndim=self.spatial_ndim,
            name="kernel_size",
        )
        self.strides = canonicalize(strides, self.spatial_ndim, name="strides")
        self.padding = padding
        self.kernel_dilation = canonicalize(
            kernel_dilation,
            self.spatial_ndim,
            name="kernel_dilation",
        )
        weight_init_func = resolve_init_func(weight_init_func)
        bias_init_func = resolve_init_func(bias_init_func)
        self.groups = positive_int_cb(groups)

        if self.out_features % self.groups != 0:
            msg = f"Expected out_features % groups == 0, got {self.out_features % self.groups}"
            raise ValueError(msg)

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

        y = fft_conv_general_dilated(
            jnp.expand_dims(x, axis=0),
            self.weight,
            strides=self.strides,
            padding=padding,
            groups=self.groups,
            dilation=self.kernel_dilation,
        )
        y = jnp.squeeze(y, axis=0)
        if self.bias is None:
            return y
        return y + self.bias

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the convolution."""
        ...


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
            - "same"/"SAME" for padding such that the output has the same shape
              as the input.
            - "valid"/"VALID" for no padding.

        kernel_dilation: Dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init_func: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``0``.

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
            - "same"/"SAME" for padding such that the output has the same shape
              as the input.
            - "valid"/"VALID" for no padding.

        kernel_dilation: Dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init_func: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``0``.

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
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
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
            - "same"/"SAME" for padding such that the output has the same shape
              as the input.
            - "valid"/"VALID" for no padding.

        kernel_dilation: Dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init_func: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``0``.

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
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class FFTConvNDTranspose(sk.TreeClass):
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
            kernel_size,
            self.spatial_ndim,
            name="kernel_size",
        )
        self.strides = canonicalize(strides, self.spatial_ndim, name="strides")
        self.padding = padding
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

        if self.in_features % self.groups != 0:
            raise ValueError(
                f"Expected in_features % groups == 0, "
                f"got {self.in_features % self.groups}"
            )

        weight_shape = (out_features, in_features // groups, *self.kernel_size)  # OIHW
        self.weight = weight_init_func(key, weight_shape)

        if bias_init_func is None:
            self.bias = None
        else:
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

        y = fft_conv_general_dilated(
            jnp.expand_dims(x, axis=0),
            self.weight,
            strides=self.strides,
            padding=transposed_padding,
            groups=self.groups,
            dilation=self.kernel_dilation,
        )

        y = jnp.squeeze(y, axis=0)

        if self.bias is None:
            return y
        return y + self.bias

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the convolution."""
        ...


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
            - "same"/"SAME" for padding such that the output has the same shape
              as the input.
            - "valid"/"VALID" for no padding.

        output_padding: Padding of the output after convolution. accepts:

            - single integer for same padding in all dimensions.

        kernel_dilation: Dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init_func: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``0``.

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
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
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
            - "same"/"SAME" for padding such that the output has the same shape
              as the input.
            - "valid"/"VALID" for no padding.

        output_padding: Padding of the output after convolution. accepts:

            - single integer for same padding in all dimensions.

        kernel_dilation: Dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init_func: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``0``.

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
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
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
            - "same"/"SAME" for padding such that the output has the same shape
              as the input.
            - "valid"/"VALID" for no padding.

        output_padding: Padding of the output after convolution. accepts:

            - single integer for same padding in all dimensions.

        kernel_dilation: Dilation of the convolutional kernel accepts:

            - single integer for same dilation in all dimensions.
            - sequence of integers for different dilation in each dimension.

        weight_init_func: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        groups: number of groups to use for grouped convolution.
        key: key to use for initializing the weights. defaults to ``0``.

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
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class DepthwiseFFTConvND(sk.TreeClass):
    def __init__(
        self,
        in_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        depth_multiplier: int = 1,
        strides: StridesType = 1,
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
        self.padding = padding
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

        y = fft_conv_general_dilated(
            jnp.expand_dims(x, axis=0),
            self.weight,
            strides=self.strides,
            padding=padding,
            groups=x.shape[0],
            dilation=self.kernel_dilation,
        )
        y = jnp.squeeze(y, axis=0)
        if self.bias is None:
            return y
        return y + self.bias

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the convolution."""
        ...


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
            - "same"/"SAME" for padding such that the output has the same shape
              as the input.
            - "valid"/"VALID" for no padding.

        weight_init_func: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``0``.


    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.DepthwiseFFTConv1D(3, 3, depth_multiplier=2, strides=2)
        >>> l1(jnp.ones((3, 32))).shape
        (6, 16)

    Note:
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
            - "same"/"SAME" for padding such that the output has the same shape
              as the input.
            - "valid"/"VALID" for no padding.

        weight_init_func: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``0``.


    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.DepthwiseFFTConv2D(3, 3, depth_multiplier=2, strides=2)
        >>> l1(jnp.ones((3, 32, 32))).shape
        (6, 16, 16)

    Note:
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
            - "same"/"SAME" for padding such that the output has the same shape
              as the input.
            - "valid"/"VALID" for no padding.

        weight_init_func: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``0``.


    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.DepthwiseFFTConv3D(3, 3, depth_multiplier=2, strides=2)
        >>> l1(jnp.ones((3, 32, 32, 32))).shape
        (6, 16, 16, 16)

    Note:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class SeparableFFTConv1D(sk.TreeClass):
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
            - "same"/"SAME" for padding such that the output has the same shape
              as the input.
            - "valid"/"VALID" for no padding.

        weight_init_func: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``0``.


    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.SeparableFFTConv1D(3, 3, 3, depth_multiplier=2)
        >>> l1(jnp.ones((3, 32))).shape
        (3, 32)

    Note:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

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
        self.in_features = in_features
        self.depth_multiplier = canonicalize(
            depth_multiplier,
            self.in_features,
            name="depth_multiplier",
        )

        self.depthwise_conv = DepthwiseFFTConv1D(
            in_features=in_features,
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            weight_init_func=depthwise_weight_init_func,
            bias_init_func=None,  # no bias for lhs
            key=key,
        )

        self.pointwise_conv = FFTConv1D(
            in_features=in_features * depth_multiplier,
            out_features=out_features,
            kernel_size=1,
            strides=strides,
            padding=padding,
            weight_init_func=pointwise_weight_init_func,
            bias_init_func=pointwise_bias_init_func,
            key=key,
        )

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

    @property
    def spatial_ndim(self) -> int:
        return 1


class SeparableFFTConv2D(sk.TreeClass):
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
            - "same"/"SAME" for padding such that the output has the same shape
              as the input.
            - "valid"/"VALID" for no padding.

        weight_init_func: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``0``.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.SeparableFFTConv2D(3, 3, 3, depth_multiplier=2)
        >>> l1(jnp.ones((3, 32, 32))).shape
        (3, 32, 32)

    Note:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

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
        self.in_features = in_features
        self.depth_multiplier = canonicalize(
            depth_multiplier,
            self.in_features,
            name="depth_multiplier",
        )

        self.depthwise_conv = DepthwiseFFTConv2D(
            in_features=in_features,
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            weight_init_func=depthwise_weight_init_func,
            bias_init_func=None,  # no bias for lhs
            key=key,
        )

        self.pointwise_conv = FFTConv2D(
            in_features=in_features * depth_multiplier,
            out_features=out_features,
            kernel_size=1,
            strides=strides,
            padding=padding,
            weight_init_func=pointwise_weight_init_func,
            bias_init_func=pointwise_bias_init_func,
            key=key,
        )

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

    @property
    def spatial_ndim(self) -> int:
        return 2


class SeparableFFTConv3D(sk.TreeClass):
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
            - "same"/"SAME" for padding such that the output has the same shape
              as the input.
            - "valid"/"VALID" for no padding.

        weight_init_func: function to use for initializing the weights. defaults
            to ``glorot uniform``.
        bias_init_func: function to use for initializing the bias. defaults to
            ``zeros``. set to ``None`` to not use a bias.
        key: key to use for initializing the weights. defaults to ``0``.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> l1 = sk.nn.SeparableFFTConv3D(3, 3, 3, depth_multiplier=2)
        >>> l1(jnp.ones((3, 32, 32, 32))).shape
        (3, 32, 32, 32)

    Note:
        - https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        - https://github.com/google/flax/blob/main/flax/linen/linear.py
    """

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
        self.in_features = in_features
        self.depth_multiplier = canonicalize(
            depth_multiplier,
            self.in_features,
            name="depth_multiplier",
        )

        self.depthwise_conv = DepthwiseFFTConv3D(
            in_features=in_features,
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            weight_init_func=depthwise_weight_init_func,
            bias_init_func=None,  # no bias for lhs
            key=key,
        )

        self.pointwise_conv = FFTConv3D(
            in_features=in_features * depth_multiplier,
            out_features=out_features,
            kernel_size=1,
            strides=strides,
            padding=padding,
            weight_init_func=pointwise_weight_init_func,
            bias_init_func=pointwise_bias_init_func,
            key=key,
        )

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

    @property
    def spatial_ndim(self) -> int:
        return 3
