# this script defines the convolutional layers
# https://arxiv.org/pdf/1603.07285.pdf

from __future__ import annotations

import functools as ft
from types import FunctionType
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytreeclass as pytc
from jax.lax import ConvDimensionNumbers


def _wrap_partial(func: Any) -> jtu.Partial | Callable:
    if isinstance(func, FunctionType):
        return jtu.Partial(func)
    return func


def _check_and_return(value, ndim, name):
    if isinstance(value, int):
        return (value,) * ndim
    elif isinstance(value, tuple):
        assert len(value) == ndim, f"{name} must be a tuple of length {ndim}"
        return tuple(value)
    else:
        raise ValueError(f"Expected int or tuple for {name}, got {value}.")


_check_and_return_kernel = ft.partial(_check_and_return, name="kernel_size")
_check_and_return_stride = ft.partial(_check_and_return, name="stride")
_check_and_return_rate = ft.partial(_check_and_return, name="rate")


def _check_and_return_padding(value, ndim):
    if isinstance(value, str):
        value = value.upper()
        assert value in ["SAME", "VALID"], f"Expected 'SAME' or 'VALID' for padding, got {value}."  # fmt: skip
    elif isinstance(value, tuple):
        assert all(
            isinstance(v, tuple) and len(v) == 2 for v in value
        ), f"Expected tuple of tuples of length 2 (begin,end) for each dimension padding, got {value}."
    elif isinstance(value, int):
        value = ((value, value),) * ndim
    else:
        raise ValueError(f"Expected int, str or tuple for padding got {value}.")
    return value


@pytc.treeclass
class ConvND:
    weight: jnp.ndarray
    bias: jnp.ndarray
    in_features: int = pytc.nondiff_field()  # number of input features
    out_features: int = pytc.nondiff_field()  # number of output features
    kernel_size: int | tuple[int, ...] = pytc.nondiff_field()
    strides: int | tuple[int, ...] = pytc.nondiff_field()  # stride of the convolution
    rate: int | tuple[int, ...] = pytc.nondiff_field()
    padding: str | tuple[tuple[int, int], ...] = pytc.nondiff_field()
    weight_init_func: Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]
    bias_init_func: Callable[[jr.PRNGKey, tuple[int]], jnp.ndarray]
    groups: int = pytc.nondiff_field(repr=False)
    dimension_numbers: jax.lax.ConvDimensionNumbers = pytc.nondiff_field(repr=False)
    kernel_dilation: int | tuple[int, ...] = pytc.nondiff_field()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        strides: int | tuple[int, ...] = 1,
        padding: str | tuple[tuple[int, int], ...] = "SAME",
        rate: int = 1,
        weight_init_func: Callable | None = jax.nn.initializers.glorot_uniform(),
        bias_init_func: Callable | None = jax.nn.initializers.zeros,
        groups: int = 1,
        ndim: int = 2,
        key: jr.PRNGKey = jr.PRNGKey(0),
        transpose: bool = False,
    ):
        """Convolutional layer.

        Args:
            in_features (int): number of input features
            out_features (int): number of output features
            kernel_size (int | tuple[int, ...]): size of the convolutional kernel
            stride (int | tuple[int, ...], optional): stride . Defaults to 1.
            padding (str | tuple[tuple[int, int], ...], optional): padding. Defaults to "SAME".
            rate (int, optional): dilation rate. Defaults to 1.
            weight_init_func (Callable | None, optional): weight initialization function.
            bias_init_func (Callable | None, optional): _description_. Defaults to jax.nn.initializers.zeros.
            groups (int, optional): number of groups. Defaults to 1.
            ndim (int, optional): spatial dimensions. Defaults to 2.
            key (jr.PRNGKey, optional): key for random number generation. Defaults to jr.PRNGKey(0).
        """
        # type assertions
        assert isinstance(
            in_features, int
        ), f"Expected int for in_features, got {in_features}."

        assert isinstance(
            out_features, int
        ), f"Expected int for out_features, got {out_features}."

        assert isinstance(groups, int), f"Expected int for groups, got {groups}."

        assert isinstance(
            weight_init_func, Callable
        ), f"Expected Callable for weight_init_func, got {weight_init_func}."

        assert isinstance(
            bias_init_func, Callable
        ), f"Expected Callable for bias_init_func, got {bias_init_func}."

        # assert proper values
        assert in_features > 0, "`in_features` must be greater than 0."
        assert out_features > 0, "`out_features` must be greater than 0."
        assert groups > 0, "`groups` must be greater than 0."
        assert out_features % groups == 0, "`out_features` not divisible by `groups`."  # fmt: skip

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = _check_and_return_kernel(kernel_size, ndim)
        self.strides = _check_and_return_stride(strides, ndim)
        self.rate = _check_and_return_rate(rate, ndim)
        self.padding = _check_and_return_padding(padding, ndim)
        self.groups = groups
        self.weight_init_func = _wrap_partial(weight_init_func)
        self.bias_init_func = _wrap_partial(bias_init_func)
        # OIHW
        self.weight = weight_init_func(
            key,
            (out_features, in_features // groups, *self.kernel_size),
        )

        self.bias = (
            bias_init_func(key, (out_features, *(1,) * ndim))
            if bias_init_func is not None
            else None
        )

        self.kernel_dilation = (1,) * ndim
        self.dimension_numbers = ConvDimensionNumbers(*((tuple(range(ndim + 2)),) * 3))

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        y = self.bias if self.bias is not None else 0.0
        y += jax.lax.conv_general_dilated(
            x[None],
            self.weight,
            self.strides,
            self.padding,
            self.kernel_dilation,
            self.rate,
            self.dimension_numbers,
            self.groups,
        )

        return y[0]


@pytc.treeclass
class Conv1D(ConvND):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        strides=1,
        padding="SAME",
        rate=1,
        weight_init_func=jax.nn.initializers.glorot_uniform(),
        bias_init_func=jax.nn.initializers.zeros,
        groups: int = 1,
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            strides=strides,
            padding=padding,
            rate=rate,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            groups=groups,
            ndim=1,
            key=key,
        )


@pytc.treeclass
class Conv2D(ConvND):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        strides=1,
        padding="SAME",
        rate=1,
        weight_init_func=jax.nn.initializers.glorot_uniform(),
        bias_init_func=jax.nn.initializers.zeros,
        groups: int = 1,
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            strides=strides,
            padding=padding,
            rate=rate,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            groups=groups,
            ndim=2,
            key=key,
        )


@pytc.treeclass
class Conv3D(ConvND):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        strides=1,
        padding="SAME",
        rate=1,
        weight_init_func=jax.nn.initializers.glorot_uniform(),
        bias_init_func=jax.nn.initializers.zeros,
        groups: int = 1,
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            strides=strides,
            padding=padding,
            rate=rate,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            groups=groups,
            ndim=3,
            key=key,
        )
