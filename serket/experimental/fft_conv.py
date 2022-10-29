from __future__ import annotations

import dataclasses
import functools as ft
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc
from jax.lax import ConvDimensionNumbers

from serket.nn.utils import (
    _check_and_return_init_func,
    _check_and_return_kernel,
    _check_and_return_padding,
    _check_and_return_positive_int,
    _check_and_return_strides,
    _check_in_features,
    _check_spatial_in_shape,
    _lazy_conv,
)


def _ungrouped_matmul(x, y) -> jnp.ndarray:
    alpha = "abcdefghijklmnopqrstuvwx"
    lhs = "y" + alpha[: x.ndim - 1]
    rhs = "z" + alpha[: y.ndim - 1]
    out = "yz" + lhs[2:]
    return jnp.einsum(f"{lhs},{rhs}->{out}", x, y)


def _grouped_matmul(x, y, groups) -> jnp.ndarray:
    b, c, *s = x.shape  # batch, channels, spatial
    o, i, *k = y.shape  # out_channels, in_channels, kernel
    x = x.reshape(groups, b, c // groups, *s)  # groups, batch, channels, spatial
    y = y.reshape(groups, o // groups, *(i, *k))
    z = jax.vmap(_ungrouped_matmul, in_axes=(0, 0), out_axes=1)(x, y)
    return z.reshape(z.shape[0], z.shape[1] * z.shape[2], *z.shape[3:])


def grouped_matmul(x, y, groups: int = 1):
    return _ungrouped_matmul(x, y) if groups == 1 else _grouped_matmul(x, y, groups)


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


def _pad_even_along_axis(x, axis):
    # pad the input with an extra zero on the right side if
    # the input size is odd along the axis
    if x.shape[axis] % 2 != 0:
        padding = [(0, 0)] * x.ndim
        padding[axis] = (0, 1)
        return jnp.pad(x, padding)
    return x


def _general_undilated_fft_conv(
    x: jnp.ndarray,
    y: jnp.ndarray,
    strides: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
    groups: int,
):
    ndim = x.ndim - 2  # spatial dimensions
    w = jnp.kron(y, jnp.ones((1, 1, *(1,) * ndim)))
    x = _general_pad(x, ((0, 0), (0, 0), *padding))
    x_shape = x.shape
    x = _pad_even_along_axis(x, -1)
    kernel_padding = tuple((0, x.shape[i] - y.shape[i]) for i in range(2, ndim + 2))
    w_pad = _general_pad(w, ((0, 0), (0, 0), *kernel_padding))
    x_fft = jnp.fft.rfftn(x, axes=range(2, ndim + 2))
    w_fft = jnp.conjugate(jnp.fft.rfftn(w_pad, axes=range(2, ndim + 2)))
    z_fft = grouped_matmul(x_fft, w_fft, groups)
    z = jnp.fft.irfftn(z_fft, axes=range(2, ndim + 2))

    z = jax.lax.slice(
        z,
        start_indices=(0,) * (ndim + 2),  # start at 0 for all dimensions
        limit_indices=(
            z.shape[0],
            z.shape[1],
            *tuple((x_shape[i] - w.shape[i] + 1) for i in range(2, ndim + 2)),
        ),
        strides=(1, 1, *strides),
    )
    return z


@pytc.treeclass
class FFTConvND:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int = pytc.nondiff_field()
    out_features: int = pytc.nondiff_field()
    kernel_size: int | tuple[int, ...] = pytc.nondiff_field()
    strides: int | tuple[int, ...] = pytc.nondiff_field()
    padding: str | int | tuple[int, ...] | tuple[tuple[int, int], ...] = pytc.nondiff_field()  # fmt: skip
    weight_init_func: str | Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]
    bias_init_func: str | Callable[[jr.PRNGKey, tuple[int]], jnp.ndarray]
    groups: int = pytc.nondiff_field()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        strides: int | tuple[int, ...] = 1,
        padding: str | int | tuple[int, ...] | tuple[tuple[int, int], ...] = "SAME",
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable = "zeros",
        groups: int = 1,
        ndim: int = 2,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """FFT Convolutional layer.

        Args:
            in_features: number of input features
            out_features: number of output features
            kernel_size: size of the convolutional kernel
            strides: stride of the convolution
            padding: padding of the input
            weight_init_func: function to use for initializing the weights
            bias_init_func: function to use for initializing the bias
            groups: number of groups to use for grouped convolution
            ndim: number of dimensions of the convolution
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

            self._partial_init = ft.partial(
                FFTConvND.__init__,
                self=self,
                out_features=out_features,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                weight_init_func=weight_init_func,
                bias_init_func=bias_init_func,
                groups=groups,
                ndim=ndim,
                key=key,
            )

            return

        if hasattr(self, "_partial_init"):
            delattr(self, "_partial_init")

        self.in_features = _check_and_return_positive_int(in_features, "in_features")
        self.out_features = _check_and_return_positive_int(out_features, "out_features")
        self.groups = _check_and_return_positive_int(groups, "groups")
        self.ndim = _check_and_return_positive_int(ndim, "ndim")

        msg = f"Expected out_features % groups == 0, got {self.out_features % self.groups}"
        assert self.out_features % self.groups == 0, msg

        self.kernel_size = _check_and_return_kernel(kernel_size, ndim)
        self.strides = _check_and_return_strides(strides, ndim)

        self.padding = _check_and_return_padding(padding, self.kernel_size)
        self.weight_init_func = _check_and_return_init_func(weight_init_func, "weight_init_func")  # fmt: skip
        self.bias_init_func = _check_and_return_init_func(bias_init_func, "bias_init_func")  # fmt: skip

        weight_shape = (out_features, in_features // groups, *self.kernel_size)
        self.weight = self.weight_init_func(key, weight_shape)

        if bias_init_func is None:
            self.bias = None
        else:
            bias_shape = (out_features, *(1,) * ndim)
            self.bias = self.bias_init_func(key, bias_shape)

    @_lazy_conv
    @_check_spatial_in_shape
    @_check_in_features
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        y = _general_undilated_fft_conv(
            jnp.expand_dims(x, axis=0),
            self.weight,
            strides=self.strides,
            padding=self.padding,
            groups=self.groups,
        )
        y = jnp.squeeze(y, axis=0)
        if self.bias is None:
            return y
        return y + self.bias


FFTConv1D = ft.partial(FFTConvND, ndim=1)
FFTConv2D = ft.partial(FFTConvND, ndim=2)
FFTConv3D = ft.partial(FFTConvND, ndim=3)
