# this script defines spatially aware weighted scanning functions
# credits  Mahmoud Asem 2022 @KAIST

from __future__ import annotations

import dataclasses
import functools as ft
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import kernex as kex
import pytreeclass as pytc

from .utils import (
    _TRACER_ERROR_MSG,
    _check_and_return_init_func,
    _check_and_return_kernel,
    _check_and_return_padding,
    _check_and_return_strides,
)


@pytc.treeclass
class ConvScanND:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int = pytc.nondiff_field()
    out_features: int = pytc.nondiff_field()
    kernel_size: int | tuple[int, ...] = pytc.nondiff_field()
    strides: int | tuple[int, ...] = pytc.nondiff_field()
    padding: str | int | tuple[int, ...] | tuple[tuple[int, int], ...] = pytc.nondiff_field()  # fmt: skip
    weight_init_func: str | Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]
    bias_init_func: str | Callable[[jr.PRNGKey, tuple[int]], jnp.ndarray]

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        strides=1,
        padding="SAME",
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        ndim=2,
        key=jr.PRNGKey(0),
    ):
        """Spatially aware weighted scanning layer.
        This class of layers uses a kernel to scan over the input and apply a weighted sum to the input.
        The kernel is a spatially aware weight matrix that is applied to the input in a sliding window fashion.

        Args:
            in_features: number of input features
            out_features: number of output features
            kernel_size: size of the scanning kernel
            strides: stride of the scanning. Defaults to 1.
            padding: padding of the scanning. Defaults to "SAME".
            weight_init_func: function to initialize the weight matrix with . Defaults to "glorot_uniform".
            bias_init_func: bias initializer function . Defaults to zeros.
            ndim: number of spatial dimensions. Defaults to 2.
            key: Random key for weight and bias initialization. Defaults to jr.PRNGKey(0).

        Example:

            # 1D example
            >>> layer =  sk.nn.ConvScan1D(in_features=1, out_features=1, kernel_size=3, strides=1, padding=1)
            >>> layer = layer.at["weight"].set(jnp.array([[[1, 2, 3]]]))
            >>> x = jnp.array([[1, 2, 3, 4, 5, 6]])
            >>> layer(x)
            [[  8.,  21.,  39.,  62.,  90., 102.]]


            The operation goes as follows:

            *First sliding padded window
            [ 0, 1, 2 ] * [1, 2, 3] = 0*1 + 1*2 + 2*3 = 8
            Unlike a normal convolution, this result is set to the original input value
            x <- [ 8, 2, 3, 4, 5, 6]

            *Second sliding padded window
            [ 8, 2, 3 ] * [1, 2, 3] = 8*1 + 2*2 + 3*3 = 21
            x <- [ 8, 21, 3, 4, 5, 6]

            *Third sliding padded window
            [ 21, 3, 4 ] * [1, 2, 3] = 21*1 + 3*2 + 4*3 = 39
            x <- [ 8, 21, 39, 4, 5, 6]

            *Fourth sliding padded window
            [ 39, 4, 5 ] * [1, 2, 3] = 39*1 + 4*2 + 5*3 = 62
            x <- [ 8, 21, 39, 62, 5, 6]

            *Fifth sliding padded window
            [ 62, 5, 6 ] * [1, 2, 3] = 62*1 + 5*2 + 6*3 = 90
            x <- [ 8, 21, 39, 62, 90, 6]

            *Sixth sliding padded window
            [ 90, 6, 0 ] * [1, 2, 3] = 90*1 + 6*2 + 0*3 = 102
            x <- [ 8, 21, 39, 62, 90, 102]

            # 2D example
            >>> layer =  sk.nn.ConvScan2D(in_features=1, out_features=1, kernel_size=3, strides=1, padding=1)
            >>> layer = layer.at["weight"].set(jnp.array([[[[1, 2, 3],[4,5,6],[7,8,9]]]]))
            >>> x = jnp.ones([1,3,3])
            >>> layer(x)
            [[[2.80000e+01, 1.47000e+02, 6.08000e+02],
              [5.25000e+02, 4.28100e+03, 1.85070e+04],
              [1.39040e+04, 1.20235e+05, 5.22240e+05]]]


            The operation goes as follows:
                1   1   1
            x = 1   1   1
                1   1   1

            *First sliding padded window
            00 00 00     01 02 03
            00 01 01  *  04 05 06   = 28
            00 01 01     07 08 09

                 28 01 01
            x <- 01 01 01
                 01 01 01


            *Second sliding padded window
            00 00 00     01 02 03
            28 01 01  *  04 05 06   =  147
            01 01 01     07 08 09

                 28  147 01
            x <- 01  01  01
                 01  01  01

            ...



        See:
            https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
            https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
        """
        if in_features is None:
            for field_item in dataclasses.fields(self):
                setattr(self, field_item.name, None)
            self._partial_init = ft.partial(
                ConvScanND.__init__,
                self=self,
                out_features=out_features,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                weight_init_func=weight_init_func,
                bias_init_func=bias_init_func,
                ndim=ndim,
                key=key,
            )
            return

        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError(
                f"Expected `in_features` to be a positive integer, got {in_features}"
            )

        if not isinstance(out_features, int) or out_features <= 0:
            raise ValueError(
                f"Expected `out_features` to be a positive integer, got {out_features}"
            )

        self.in_features = in_features
        self.out_features = out_features

        self.kernel_size = _check_and_return_kernel(kernel_size, ndim)
        self.strides = _check_and_return_strides(strides, ndim)
        self.padding = _check_and_return_padding(padding, self.kernel_size)
        self.weight_init_func = _check_and_return_init_func(weight_init_func, "weight_init_func")  # fmt: skip
        self.bias_init_func = _check_and_return_init_func(bias_init_func, "bias_init_func")  # fmt: skip

        weight_shape = (out_features, in_features, *self.kernel_size)  # OIHW

        self.weight = self.weight_init_func(key, weight_shape)

        if bias_init_func is None:
            self.bias = None
        else:
            bias_shape = (out_features, *(1,) * ndim)
            self.bias = self.bias_init_func(key, bias_shape)

        @kex.kscan(
            kernel_size=(self.in_features, *self.kernel_size),
            strides=(1, *self.strides),
            padding=((0, 0), *self.padding),
            relative=False,
        )
        def convolved_scan(x, w):
            return jnp.sum(x * w)

        self._func = convolved_scan

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        if hasattr(self, "_partial_init"):
            if isinstance(x, jax.core.Tracer):
                raise ValueError(_TRACER_ERROR_MSG)
            self._partial_init(in_features=x.shape[0])
            object.__delattr__(self, "_partial_init")

        y = self._func(x, self.weight)

        if self.bias is None:
            return y
        return y + self.bias


@pytc.treeclass
class ConvScan1D(ConvScanND):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        strides=1,
        padding="SAME",
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            strides=strides,
            padding=padding,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            ndim=1,
            key=key,
        )


@pytc.treeclass
class ConvScan2D(ConvScanND):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        strides=1,
        padding="SAME",
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            strides=strides,
            padding=padding,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            ndim=2,
            key=key,
        )


@pytc.treeclass
class ConvScan3D(ConvScanND):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        strides=1,
        padding="SAME",
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            strides=strides,
            padding=padding,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            ndim=3,
            key=key,
        )
