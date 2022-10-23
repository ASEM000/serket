from __future__ import annotations

import dataclasses
import functools as ft
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from serket.nn.convolution import ConvND
from serket.nn.utils import _TRACER_ERROR_MSG

# ------------------------------ ConvNDSemiLocal Convolutional Layers ------------------------------ #


@pytc.treeclass
class ConvNDSemiLocal:
    spatial_groups: int = pytc.nondiff_field()
    convs: tuple[ConvND]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        strides: int | tuple[int, ...] = 1,
        padding: str | int | tuple[int, ...] | tuple[tuple[int, int], ...] = "SAME",
        input_dilation: int | tuple[int, ...] = 1,
        kernel_dilation: int | tuple[int, ...] = 1,
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable = "zeros",
        spatial_groups: int = 1,
        ndim: int = 2,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """Split the last dimension into `spatial_groups` and apply different kernel to each group

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
            spatial_groups: number of groups to split the spatial dimensions into
            ndim: number of dimensions of the convolution
            key: key to use for initializing the weights

        See: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        """
        for field_item in dataclasses.fields(self):
            setattr(self, field_item.name, None)
        self._partial_init = ft.partial(
            ConvNDSemiLocal.__init__,
            self=self,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            spatial_groups=spatial_groups,
            ndim=ndim,
            key=key,
        )

        if hasattr(self, "_partial_init"):
            delattr(self, "_partial_init")

        keys = jr.split(key, spatial_groups)
        self.spatial_groups = spatial_groups
        self.convs = tuple(
            [
                ConvND(
                    in_features=in_features,
                    out_features=out_features,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    input_dilation=input_dilation,
                    kernel_dilation=kernel_dilation,
                    weight_init_func=weight_init_func,
                    bias_init_func=bias_init_func,
                    ndim=ndim,
                    key=key,
                )
                for key in keys
            ]
        )

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        if hasattr(self, "_partial_init"):
            if isinstance(x, jax.core.Tracer):
                raise ValueError(_TRACER_ERROR_MSG(self.__class__.__name__))
            self._partial_init(in_features=x.shape[0])

        # split last dimensions into spatial groups
        xs = jnp.array_split(x, self.spatial_groups, axis=-1)
        # apply different conv to each group
        return jnp.concatenate([f(xi) for f, xi in zip(self.convs, xs)], axis=-1)


@pytc.treeclass
class Conv1DSemiLocal(ConvNDSemiLocal):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        strides: int | tuple[int, ...] = 1,
        padding: str | int | tuple[int, ...] | tuple[tuple[int, int], ...] = "SAME",
        input_dilation: int | tuple[int, ...] = 1,
        kernel_dilation: int | tuple[int, ...] = 1,
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable = "zeros",
        spatial_groups: int = 1,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            spatial_groups=spatial_groups,
            ndim=1,
            key=key,
        )


@pytc.treeclass
class Conv2DSemiLocal(ConvNDSemiLocal):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        strides: int | tuple[int, ...] = 1,
        padding: str | int | tuple[int, ...] | tuple[tuple[int, int], ...] = "SAME",
        input_dilation: int | tuple[int, ...] = 1,
        kernel_dilation: int | tuple[int, ...] = 1,
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable = "zeros",
        spatial_groups: int = 1,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            spatial_groups=spatial_groups,
            ndim=2,
            key=key,
        )


@pytc.treeclass
class Conv3DSemiLocal(ConvNDSemiLocal):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        strides: int | tuple[int, ...] = 1,
        padding: str | int | tuple[int, ...] | tuple[tuple[int, int], ...] = "SAME",
        input_dilation: int | tuple[int, ...] = 1,
        kernel_dilation: int | tuple[int, ...] = 1,
        weight_init_func: str | Callable = "glorot_uniform",
        bias_init_func: str | Callable = "zeros",
        spatial_groups: int = 1,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            spatial_groups=spatial_groups,
            ndim=3,
            key=key,
        )
