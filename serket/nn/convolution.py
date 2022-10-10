# this script defines different convolutional layers
# https://arxiv.org/pdf/1603.07285.pdf
# Throughout the code, we use OIHW as the default data format for kernels. and NCHW for data.

from __future__ import annotations

import functools as ft
import operator as op
from types import FunctionType
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytreeclass as pytc
from jax.lax import ConvDimensionNumbers


# ------------------------- Utils ------------------------- #
def _check_and_return(value, ndim, name):
    if isinstance(value, int):
        return (value,) * ndim
    elif isinstance(value, tuple):
        assert len(value) == ndim, f"{name} must be a tuple of length {ndim}"
        return tuple(value)
    raise ValueError(f"Expected int or tuple for {name}, got {value}.")


_check_and_return_kernel = ft.partial(_check_and_return, name="kernel_size")
_check_and_return_strides = ft.partial(_check_and_return, name="stride")
_check_and_return_input_dilation = ft.partial(_check_and_return, name="input_dilation")
_check_and_return_kernel_dilation = ft.partial(_check_and_return, name="kernel_dilation")  # fmt: skip
_check_and_return_input_size = ft.partial(_check_and_return, name="input_size")  # fmt: skip


def _check_and_return_padding(
    padding: tuple[int | tuple[int, int] | str, ...] | int | str,
    kernel_size: tuple[int, ...],
):
    """
    Resolve padding to a tuple of tuples of ints.

    Args:
        padding: padding to resolve
        kernel_size: kernel size to use for resolving padding

    Examples:
        >>> padding= (1, (2, 3), "same")
        >>> kernel_size = (3, 3, 3)
        >>> _check_and_return_padding(padding, kernel_size)
        ((1, 1), (2, 3), (1, 1))
    """

    def _resolve_tuple_padding(padding, kernel_size):

        assert len(padding) == len(
            kernel_size
        ), f"Expected padding to be of length {len(kernel_size)}, got {len(padding)}"

        resolved_padding = [[]] * len(kernel_size)

        for i, item in enumerate(padding):
            if isinstance(item, int):
                # ex: padding = (1, 2, 3)
                resolved_padding[i] = (item, item)

            elif isinstance(item, tuple):
                # ex: padding = ((1, 2), (3, 4), (5, 6))
                assert len(item) == 2, f"Expected tuple of length 2, got {len(item)}"
                resolved_padding[i] = item

            elif isinstance(item, str):
                # ex: padding = ("same", "valid", "same")
                if item.lower() == "same":
                    resolved_padding[i] = ((kernel_size[i] - 1) // 2), (
                        kernel_size[i] // 2
                    )

                elif item.lower() == "valid":
                    resolved_padding[i] = (0, 0)

                else:
                    raise ValueError(
                        f'string argument must be in ["same","valid"].Found {item}'
                    )
        return tuple(resolved_padding)

    def _resolve_int_padding(padding, kernel_size):
        return ((padding, padding),) * len(kernel_size)

    def _resolve_string_padding(padding, kernel_size):
        if padding.lower() == "same":
            return tuple(((wi - 1) // 2, wi // 2) for wi in kernel_size)

        elif padding.lower() == "valid":
            return ((0, 0),) * len(kernel_size)

        raise ValueError(f'string argument must be in ["same","valid"].Found {padding}')

    if isinstance(padding, int):
        return _resolve_int_padding(padding, kernel_size)

    elif isinstance(padding, str):
        return _resolve_string_padding(padding, kernel_size)

    elif isinstance(padding, tuple):
        return _resolve_tuple_padding(padding, kernel_size)

    raise ValueError(
        f"Expected padding to be of type int, str or tuple, got {type(padding)}"
    )


_init_func_dict = {
    "he_normal": jax.nn.initializers.he_normal(),
    "he_uniform": jax.nn.initializers.he_uniform(),
    "glorot_normal": jax.nn.initializers.glorot_normal(),
    "glorot_uniform": jax.nn.initializers.glorot_uniform(),
    "lecun_normal": jax.nn.initializers.lecun_normal(),
    "lecun_uniform": jax.nn.initializers.lecun_uniform(),
    "normal": jax.nn.initializers.normal(),
    "uniform": jax.nn.initializers.uniform(),
    "ones": jax.nn.initializers.ones,
    "zeros": jax.nn.initializers.zeros,
    "xavier_normal": jax.nn.initializers.xavier_normal(),
    "xavier_uniform": jax.nn.initializers.xavier_uniform(),
}


def _transpose_padding(padding, kernel_size, input_dilation, extra_padding):
    """
    Transpose padding to get the padding for the transpose convolution.

    Args:
        padding: padding to transpose
        kernel_size: kernel size to use for transposing padding
        input_dilation: input dilation to use for transposing padding
        extra_padding: extra padding to use for transposing padding
    """
    return tuple(
        ((ki - 1) * di - pl, (ki - 1) * di - pr + ep)
        for (pl, pr), ki, ep, di in zip(
            padding, kernel_size, extra_padding, input_dilation
        )
    )


def _check_and_return_init_func(
    init_func: str | Callable, name: str
) -> Callable | None:
    if isinstance(init_func, FunctionType):
        return jtu.Partial(init_func)

    elif isinstance(init_func, str):
        if init_func in _init_func_dict:
            return jtu.Partial(_init_func_dict[init_func])

        raise ValueError(f"{name} must be one of {list(_init_func_dict.keys())}")

    elif init_func is None:
        return None

    raise ValueError(f"`{name}` must be a string or a function.")


def _output_shape(shape, kernel_size, padding, strides):
    """Compute the shape of the output of a convolutional layer."""
    return tuple(
        (xi + (li + ri) - ki) // si + 1
        for xi, ki, si, (li, ri) in zip(shape, kernel_size, strides, padding)
    )


# ------------------------------ Convolutional Layers ------------------------------ #


@pytc.treeclass
class ConvND:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int = pytc.nondiff_field()
    out_features: int = pytc.nondiff_field()
    kernel_size: int | tuple[int, ...] = pytc.nondiff_field()
    strides: int | tuple[int, ...] = pytc.nondiff_field()
    padding: str | int | tuple[int, ...] | tuple[tuple[int, int], ...] = pytc.nondiff_field()  # fmt: skip
    input_dilation: int | tuple[int, ...] = pytc.nondiff_field()
    kernel_dilation: int | tuple[int, ...] = pytc.nondiff_field()
    weight_init_func: str | Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]
    bias_init_func: str | Callable[[jr.PRNGKey, tuple[int]], jnp.ndarray]
    groups: int = pytc.nondiff_field()

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        strides=1,
        padding="SAME",
        input_dilation=1,
        kernel_dilation=1,
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        groups=1,
        ndim=2,
        key=jr.PRNGKey(0),
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
            ndim: number of dimensions of the convolution
            key: key to use for initializing the weights

        See: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html
        """
        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError(
                f"Expected `in_features` to be a positive integer, got {in_features}"
            )

        if not isinstance(out_features, int) or out_features <= 0:
            raise ValueError(
                f"Expected `out_features` to be a positive integer, got {out_features}"
            )

        if not isinstance(groups, int) or groups <= 0:
            raise ValueError(f"Expected groups to be a positive integer, got {groups}")

        assert (
            out_features % groups == 0
        ), f"Expected out_features % groups == 0, got {out_features % groups}"

        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups

        self.kernel_size = _check_and_return_kernel(kernel_size, ndim)
        self.strides = _check_and_return_strides(strides, ndim)
        self.input_dilation = _check_and_return_input_dilation(input_dilation, ndim)
        self.kernel_dilation = _check_and_return_kernel_dilation(kernel_dilation, ndim)
        self.padding = _check_and_return_padding(padding, self.kernel_size)
        self.weight_init_func = _check_and_return_init_func(weight_init_func, "weight_init_func")  # fmt: skip
        self.bias_init_func = _check_and_return_init_func(bias_init_func, "bias_init_func")  # fmt: skip

        weight_shape = (out_features, in_features // groups, *self.kernel_size)  # OIHW
        self.weight = self.weight_init_func(key, weight_shape)

        bias_shape = (out_features, *(1,) * ndim)

        if bias_init_func is None:
            self.bias = None
        else:
            self.bias = self.bias_init_func(key, bias_shape)

        self.dimension_numbers = ConvDimensionNumbers(*((tuple(range(ndim + 2)),) * 3))

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        y = jax.lax.conv_general_dilated(
            lhs=x[None],
            rhs=self.weight,
            window_strides=self.strides,
            padding=self.padding,
            lhs_dilation=self.input_dilation,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=self.dimension_numbers,
            feature_group_count=self.groups,
        )

        if self.bias is None:
            return y[0]
        return (y + self.bias)[0]


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
        input_dilation=1,
        kernel_dilation=1,
        weight_init_func=jax.nn.initializers.glorot_uniform(),
        bias_init_func=jax.nn.initializers.zeros,
        groups: int = 1,
        key=jr.PRNGKey(0),
    ):
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
        input_dilation=1,
        kernel_dilation=1,
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        groups: int = 1,
        key=jr.PRNGKey(0),
    ):
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
        input_dilation=1,
        kernel_dilation=1,
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        groups: int = 1,
        key=jr.PRNGKey(0),
    ):
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
            ndim=3,
            key=key,
        )


# ------------------------------ Transposed Convolutional Layers ------------------------------ #


@pytc.treeclass
class ConvNDTranspose:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int = pytc.nondiff_field()
    out_features: int = pytc.nondiff_field()
    kernel_size: int | tuple[int, ...] = pytc.nondiff_field()
    padding: str | int | tuple[int, ...] | tuple[tuple[int, int], ...] = pytc.nondiff_field()  # fmt: skip
    output_padding: int | tuple[int, ...] = pytc.nondiff_field()
    strides: int | tuple[int, ...] = pytc.nondiff_field()
    kernel_dilation: int | tuple[int, ...] = pytc.nondiff_field()
    weight_init_func: str | Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]
    bias_init_func: Callable[[jr.PRNGKey, tuple[int]], jnp.ndarray]
    groups: int = pytc.nondiff_field()

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        strides=1,
        padding="SAME",
        output_padding=0,
        kernel_dilation=1,
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        groups=1,
        ndim=2,
        key=jr.PRNGKey(0),
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
            ndim : Number of dimensions
            key : PRNG key
        """
        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError(
                f"Expected in_features to be a positive integer, got {in_features}"
            )

        if not isinstance(out_features, int) or out_features <= 0:
            raise ValueError(
                f"Expected out_features to be a positive integer, got {out_features}"
            )

        if not isinstance(groups, int) or groups <= 0:
            raise ValueError(f"Expected groups to be a positive integer, got {groups}")

        assert (
            out_features % groups == 0
        ), f"Expected out_features % groups == 0, got {out_features % groups}"

        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups

        self.kernel_size = _check_and_return_kernel(kernel_size, ndim)
        self.strides = _check_and_return_strides(strides, ndim)
        self.kernel_dilation = _check_and_return_kernel_dilation(kernel_dilation, ndim)
        self.padding = _check_and_return_padding(padding, self.kernel_size)
        self.output_padding = _check_and_return_strides(output_padding, ndim)
        self.weight_init_func = _check_and_return_init_func(weight_init_func, "weight_init_func")  # fmt: skip
        self.bias_init_func = _check_and_return_init_func(bias_init_func, "bias_init_func")  # fmt: skip

        weight_shape = (out_features, in_features // groups, *self.kernel_size)  # OIHW
        self.weight = self.weight_init_func(key, weight_shape)

        bias_shape = (out_features, *(1,) * ndim)

        if bias_init_func is None:
            self.bias = None
        else:
            self.bias = self.bias_init_func(key, bias_shape)

        self.dimension_numbers = ConvDimensionNumbers(*((tuple(range(ndim + 2)),) * 3))

        self.transposed_padding = _transpose_padding(
            padding=self.padding,
            extra_padding=self.output_padding,
            kernel_size=self.kernel_size,
            input_dilation=self.kernel_dilation,
        )

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        y = jax.lax.conv_transpose(
            lhs=x[None],
            rhs=self.weight,
            strides=self.strides,
            padding=self.transposed_padding,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=self.dimension_numbers,
        )

        if self.bias is None:
            return y[0]
        return (y + self.bias)[0]


@pytc.treeclass
class Conv1DTranspose(ConvNDTranspose):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        strides=1,
        padding="SAME",
        output_padding=0,
        kernel_dilation=1,
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        groups=1,
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            groups=groups,
            ndim=1,
            key=key,
        )


@pytc.treeclass
class Conv2DTranspose(ConvNDTranspose):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        strides=1,
        padding="SAME",
        output_padding=0,
        kernel_dilation=1,
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        groups=1,
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            groups=groups,
            ndim=2,
            key=key,
        )


@pytc.treeclass
class Conv3DTranspose(ConvNDTranspose):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        strides=1,
        padding="SAME",
        output_padding=0,
        kernel_dilation=1,
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        groups=1,
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            groups=groups,
            ndim=3,
            key=key,
        )


# ------------------------------ Depthwise Convolutional Layers ------------------------------ #


@pytc.treeclass
class DepthwiseConvND:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int = pytc.nondiff_field()  # number of input features
    kernel_size: int | tuple[int, ...] = pytc.nondiff_field()
    strides: int | tuple[int, ...] = pytc.nondiff_field()  # stride of the convolution
    padding: str | int | tuple[tuple[int, int], ...] = pytc.nondiff_field()

    weight_init_func: str | Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]
    bias_init_func: str | Callable[[jr.PRNGKey, tuple[int]], jnp.ndarray]
    kernel_dilation: int | tuple[int, ...] = pytc.nondiff_field()

    def __init__(
        self,
        in_features,
        kernel_size,
        *,
        depth_multiplier=1,
        strides=1,
        padding="SAME",
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        ndim: int = 2,
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
            ndim: number of spatial dimensions
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
        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError(
                f"Expected in_features to be a positive integer, got {in_features}"
            )

        if not isinstance(depth_multiplier, int) or depth_multiplier <= 0:
            raise ValueError(
                f"Expected depth_multiplier to be a positive integer, got {depth_multiplier}"
            )

        self.in_features = in_features
        self.depth_multiplier = depth_multiplier

        self.kernel_size = _check_and_return_kernel(kernel_size, ndim)
        self.strides = _check_and_return_strides(strides, ndim)
        self.input_dilation = _check_and_return_input_dilation(1, ndim)
        self.kernel_dilation = _check_and_return_kernel_dilation(1, ndim)
        self.padding = _check_and_return_padding(padding, self.kernel_size)
        self.weight_init_func = _check_and_return_init_func(weight_init_func, "weight_init_func")  # fmt: skip
        self.bias_init_func = _check_and_return_init_func(bias_init_func, "bias_init_func")  # fmt: skip

        weight_shape = (depth_multiplier * in_features, 1, *self.kernel_size)  # OIHW
        self.weight = self.weight_init_func(key, weight_shape)

        bias_shape = (depth_multiplier * in_features, *(1,) * ndim)

        if bias_init_func is None:
            self.bias = None
        else:
            self.bias = self.bias_init_func(key, bias_shape)

        self.dimension_numbers = ConvDimensionNumbers(*((tuple(range(ndim + 2)),) * 3))

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        y = jax.lax.conv_general_dilated(
            lhs=x[None],
            rhs=self.weight,
            window_strides=self.strides,
            padding=self.padding,
            lhs_dilation=self.input_dilation,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=self.dimension_numbers,
            feature_group_count=self.in_features,
        )

        if self.bias is None:
            return y[0]
        return (y + self.bias)[0]


@pytc.treeclass
class DepthwiseConv1D(DepthwiseConvND):
    def __init__(
        self,
        in_features,
        kernel_size,
        *,
        depth_multiplier=1,
        strides=1,
        padding="SAME",
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            kernel_size,
            depth_multiplier=depth_multiplier,
            strides=strides,
            padding=padding,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            ndim=1,
            key=key,
        )


@pytc.treeclass
class DepthwiseConv2D(DepthwiseConvND):
    def __init__(
        self,
        in_features,
        kernel_size,
        *,
        depth_multiplier=1,
        strides=1,
        padding="SAME",
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            kernel_size,
            depth_multiplier=depth_multiplier,
            strides=strides,
            padding=padding,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            ndim=2,
            key=key,
        )


@pytc.treeclass
class DepthwiseConv3D(DepthwiseConvND):
    def __init__(
        self,
        in_features,
        kernel_size,
        *,
        depth_multiplier=1,
        strides=1,
        padding="SAME",
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            kernel_size,
            depth_multiplier=depth_multiplier,
            strides=strides,
            padding=padding,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            ndim=3,
            key=key,
        )


# ------------------------------ SeparableConvND Depthwise Convolutional Layers ------------------------------ #


@pytc.treeclass
class SeparableConvND:
    depthwise_conv: DepthwiseConvND
    pointwise_conv: DepthwiseConvND

    in_features: int = pytc.nondiff_field()
    out_features: int = pytc.nondiff_field()
    depth_multiplier: int = pytc.nondiff_field()
    kernel_size: int | tuple[int, ...] = pytc.nondiff_field()
    strides: int | tuple[int, ...] = pytc.nondiff_field()
    padding: str | int | tuple[tuple[int, int], ...] = pytc.nondiff_field()

    depthwise_weight_init_func: str | Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]  # fmt: skip
    pointwise_weight_init_func: str | Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]  # fmt: skip
    pointwise_bias_init_func: str | Callable[[jr.PRNGKey, tuple[int]], jnp.ndarray]

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        depth_multiplier=1,
        strides=1,
        padding="SAME",
        depthwise_weight_init_func="glorot_uniform",
        pointwise_weight_init_func="glorot_uniform",
        pointwise_bias_init_func="zeros",
        ndim=2,
        key=jr.PRNGKey(0),
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
            ndim : Number of spatial dimensions.

        """

        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError(
                f"Expected in_features to be a positive integer, got {in_features}"
            )

        if not isinstance(out_features, int) or out_features <= 0:
            raise ValueError(
                f"Expected out_features to be a positive integer, got {out_features}"
            )

        if not isinstance(depth_multiplier, int) or depth_multiplier <= 0:
            raise ValueError(
                f"Expected depth_multiplier to be a positive integer, got {depth_multiplier}"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.depth_multiplier = depth_multiplier

        self.kernel_size = _check_and_return_kernel(kernel_size, ndim)
        self.strides = _check_and_return_strides(strides, ndim)
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

        self.ndim = ndim

        self.depthwise_conv = DepthwiseConvND(
            in_features=in_features,
            depth_multiplier=depth_multiplier,
            kernel_size=self.kernel_size,
            strides=strides,
            padding=padding,
            weight_init_func=depthwise_weight_init_func,
            bias_init_func=None,  # no bias for lhs
            key=key,
            ndim=ndim,
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
            ndim=ndim,
        )

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


@pytc.treeclass
class SeparableConv1D(SeparableConvND):
    """1D separable convolutional layer."""

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        depth_multiplier=1,
        strides=1,
        padding="SAME",
        depthwise_weight_init_func="glorot_uniform",
        pointwise_weight_init_func="glorot_uniform",
        pointwise_bias_init_func="zeros",
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            depth_multiplier=depth_multiplier,
            strides=strides,
            padding=padding,
            depthwise_weight_init_func=depthwise_weight_init_func,
            pointwise_weight_init_func=pointwise_weight_init_func,
            pointwise_bias_init_func=pointwise_bias_init_func,
            ndim=1,
            key=key,
        )


@pytc.treeclass
class SeparableConv2D(SeparableConvND):
    """2D separable convolutional layer."""

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        depth_multiplier=1,
        strides=1,
        padding="SAME",
        depthwise_weight_init_func="glorot_uniform",
        pointwise_weight_init_func="glorot_uniform",
        pointwise_bias_init_func="zeros",
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            depth_multiplier=depth_multiplier,
            strides=strides,
            padding=padding,
            depthwise_weight_init_func=depthwise_weight_init_func,
            pointwise_weight_init_func=pointwise_weight_init_func,
            pointwise_bias_init_func=pointwise_bias_init_func,
            ndim=2,
            key=key,
        )


@pytc.treeclass
class SeparableConv3D(SeparableConvND):
    """3D separable convolutional layer."""

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        depth_multiplier=1,
        strides=1,
        padding="SAME",
        depthwise_weight_init_func="glorot_uniform",
        pointwise_weight_init_func="glorot_uniform",
        pointwise_bias_init_func="zeros",
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            depth_multiplier=depth_multiplier,
            strides=strides,
            padding=padding,
            depthwise_weight_init_func=depthwise_weight_init_func,
            pointwise_weight_init_func=pointwise_weight_init_func,
            pointwise_bias_init_func=pointwise_bias_init_func,
            ndim=3,
            key=key,
        )


# ------------------------------ ConvNDLocal Convolutional Layers ------------------------------ #


@pytc.treeclass
class ConvNDLocal:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int = pytc.nondiff_field()  # number of input features
    out_features: int = pytc.nondiff_field()  # number of output features
    kernel_size: int | tuple[int, ...] = pytc.nondiff_field()
    in_size: tuple[int, ...] = pytc.nondiff_field()  # size of input
    strides: int | tuple[int, ...] = pytc.nondiff_field()  # stride of the convolution
    padding: str | int | tuple[tuple[int, int], ...] = pytc.nondiff_field()
    input_dilation: int | tuple[int, ...] = pytc.nondiff_field()
    kernel_dilation: int | tuple[int, ...] = pytc.nondiff_field()
    weight_init_func: Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]
    bias_init_func: Callable[[jr.PRNGKey, tuple[int]], jnp.ndarray]

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        in_size,
        strides=1,
        padding="SAME",
        input_dilation=1,
        kernel_dilation=1,
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        ndim=2,
        key=jr.PRNGKey(0),
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
            ndim: number of dimensions
            key: random number generator key
        Note:
            See : https://keras.io/api/layers/locally_connected_layers/
        """

        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError(
                f"Expected in_features to be a positive integer, got {in_features}"
            )

        if not isinstance(out_features, int) or out_features <= 0:
            raise ValueError(
                f"Expected out_features to be a positive integer, got {out_features}"
            )

        self.in_features = in_features
        self.out_features = out_features

        self.in_size = _check_and_return_input_size(in_size, ndim)
        self.kernel_size = _check_and_return_kernel(kernel_size, ndim)
        self.strides = _check_and_return_strides(strides, ndim)
        self.input_dilation = _check_and_return_input_dilation(input_dilation, ndim)
        self.kernel_dilation = _check_and_return_kernel_dilation(1, ndim)
        self.padding = _check_and_return_padding(padding, self.kernel_size)
        self.weight_init_func = _check_and_return_init_func(
            weight_init_func, "weight_init_func"
        )
        self.bias_init_func = _check_and_return_init_func(
            bias_init_func, "bias_init_func"
        )
        self.dimension_numbers = ConvDimensionNumbers(*((tuple(range(ndim + 2)),) * 3))
        self.out_size = _output_shape(
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

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        y = jax.lax.conv_general_dilated_local(
            lhs=x[None],
            rhs=self.weight,
            window_strides=self.strides,
            padding=self.padding,
            filter_shape=self.kernel_size,
            lhs_dilation=self.kernel_dilation,
            rhs_dilation=self.input_dilation,  # atrous dilation
            dimension_numbers=self.dimension_numbers,
        )

        if self.bias is None:
            return y[0]
        return (y + self.bias)[0]


@pytc.treeclass
class Conv1DLocal(ConvNDLocal):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        in_size,
        strides=1,
        padding="SAME",
        input_dilation=1,
        kernel_dilation=1,
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            in_size=in_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            ndim=1,
            key=key,
        )


@pytc.treeclass
class Conv2DLocal(ConvNDLocal):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        in_size,
        strides=1,
        padding="SAME",
        input_dilation=1,
        kernel_dilation=1,
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            in_size=in_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            ndim=2,
            key=key,
        )


@pytc.treeclass
class Conv3DLocal(ConvNDLocal):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        in_size,
        strides=1,
        padding="SAME",
        input_dilation=1,
        kernel_dilation=1,
        weight_init_func="glorot_uniform",
        bias_init_func="zeros",
        key=jr.PRNGKey(0),
    ):
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            in_size=in_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            ndim=3,
            key=key,
        )
