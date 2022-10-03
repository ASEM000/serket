# this script defines different convolutional layers
# https://arxiv.org/pdf/1603.07285.pdf
# Throughout the code, we use OIHW as the default data format for kernels. and NCHW for data.

from __future__ import annotations

import functools as ft
import operator as op
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
_check_and_return_strides = ft.partial(_check_and_return, name="stride")
_check_and_return_rate = ft.partial(_check_and_return, name="rate")


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


def _output_shape(shape, kernel_size, padding, strides):
    """Compute the shape of the output of a convolutional layer."""
    return tuple(
        (xi + (li + ri) - ki) // si + 1
        for xi, ki, si, (li, ri) in zip(shape, kernel_size, strides, padding)
    )


@pytc.treeclass
class ConvND:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int = pytc.nondiff_field()  # number of input features
    out_features: int = pytc.nondiff_field()  # number of output features
    kernel_size: int | tuple[int, ...] = pytc.nondiff_field()
    strides: int | tuple[int, ...] = pytc.nondiff_field()  # stride of the convolution
    rate: int | tuple[int, ...] = pytc.nondiff_field()
    padding: str | int | tuple[tuple[int, int], ...] = pytc.nondiff_field()

    weight_init_func: Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]
    bias_init_func: Callable[[jr.PRNGKey, tuple[int]], jnp.ndarray]
    kernel_dilation: int | tuple[int, ...] = pytc.nondiff_field()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        strides: int | tuple[int, ...] = 1,
        padding: str | int | tuple[tuple[int, int], ...] = "SAME",
        rate: int = 1,
        weight_init_func: Callable | None = jax.nn.initializers.glorot_uniform(),
        bias_init_func: Callable | None = jax.nn.initializers.zeros,
        groups: int = 1,
        ndim: int = 2,
        key: jr.PRNGKey = jr.PRNGKey(0),
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
        ), f"Expected int for `in_features`, got {in_features}."

        assert isinstance(
            out_features, int
        ), f"Expected int for `out_features`, got {out_features}."

        assert isinstance(
            weight_init_func, Callable
        ), f"Expected Callable for `weight_init_func`. Found {weight_init_func}."

        assert isinstance(
            bias_init_func, (Callable, type(None))
        ), f"Expected Callable or None for `bias_init_func`. Found {bias_init_func}."

        # assert proper values
        assert in_features > 0, "`in_features` must be greater than 0."
        assert out_features > 0, "`out_features` must be greater than 0."
        assert groups > 0, "`groups` must be greater than 0."
        assert out_features % groups == 0, "`out_features` not divisible by `groups`."  # fmt: skip

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = _check_and_return_kernel(kernel_size, ndim)
        self.strides = _check_and_return_strides(strides, ndim)
        self.rate = _check_and_return_rate(rate, ndim)
        self.padding = _check_and_return_padding(padding, self.kernel_size)
        self.groups = groups
        self.weight_init_func = _wrap_partial(weight_init_func)
        self.bias_init_func = _wrap_partial(bias_init_func)

        weight_shape = (out_features, in_features // groups, *self.kernel_size)  # OIHW
        self.weight = weight_init_func(key, weight_shape)

        bias_shape = (out_features, *(1,) * ndim)
        self.bias = bias_init_func(key, bias_shape) if bias_init_func is not None else 0

        self.kernel_dilation = (1,) * ndim
        self.dimension_numbers = ConvDimensionNumbers(*((tuple(range(ndim + 2)),) * 3))

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        y = self.bias
        y += jax.lax.conv_general_dilated(
            lhs=x[None],
            rhs=self.weight,
            window_strides=self.strides,
            padding=self.padding,
            lhs_dilation=self.kernel_dilation,  # image dilation
            rhs_dilation=self.rate,  # kernel dilation
            dimension_numbers=self.dimension_numbers,
            feature_group_count=self.groups,
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


@pytc.treeclass
class ConvNDTranspose:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int = pytc.nondiff_field()  # number of input features
    out_features: int = pytc.nondiff_field()  # number of output features
    kernel_size: int | tuple[int, ...] = pytc.nondiff_field()
    strides: int | tuple[int, ...] = pytc.nondiff_field()  # stride of the convolution
    rate: int | tuple[int, ...] = pytc.nondiff_field()

    padding: str | int | tuple[tuple[int, int], ...] = pytc.nondiff_field()
    weight_init_func: Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]
    bias_init_func: Callable[[jr.PRNGKey, tuple[int]], jnp.ndarray]

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
        ndim: int = 2,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """Convolutional transpose layer.

        Args:
            in_features (int): number of input features
            out_features (int): number of output features
            kernel_size (int | tuple[int, ...]): size of the convolutional kernel
            stride (int | tuple[int, ...], optional): stride . Defaults to 1.
            padding (str | tuple[tuple[int, int], ...], optional): padding. Defaults to "SAME".
            rate (int, optional): dilation rate. Defaults to 1.
            weight_init_func (Callable | None, optional): weight initialization function.
            bias_init_func (Callable | None, optional): _description_. Defaults to jax.nn.initializers.zeros.
            ndim (int, optional): spatial dimensions. Defaults to 2.
            key (jr.PRNGKey, optional): key for random number generation. Defaults to jr.PRNGKey(0).


        Note:
            padding follows the same convention as in jax.lax.conv_general_dilated
        """
        # type assertions
        assert isinstance(
            in_features, int
        ), f"Expected int for `in_features`. Found {in_features}."

        assert isinstance(
            out_features, int
        ), f"Expected int for `out_features`. Found {out_features}."

        assert isinstance(
            weight_init_func, Callable
        ), f"Expected Callable for `weight_init_func`. Found  {weight_init_func}."

        assert isinstance(
            bias_init_func, (Callable, type(None))
        ), f"Expected Callable or None for `bias_init_func`. Found {bias_init_func}."

        # assert proper values
        assert in_features > 0, "`in_features` must be greater than 0."
        assert out_features > 0, "`out_features` must be greater than 0."

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = _check_and_return_kernel(kernel_size, ndim)
        self.strides = _check_and_return_strides(strides, ndim)
        self.rate = _check_and_return_rate(rate, ndim)
        self.padding = _check_and_return_padding(padding, self.kernel_size)
        self.weight_init_func = _wrap_partial(weight_init_func)
        self.bias_init_func = _wrap_partial(bias_init_func)

        weight_shape = (out_features, in_features, *self.kernel_size)  # IOHW
        self.weight = weight_init_func(key, weight_shape)

        bias_shape = (out_features, *(1,) * ndim)
        self.bias = bias_init_func(key, bias_shape) if bias_init_func is not None else 0

        self.dimension_numbers = ConvDimensionNumbers(*((tuple(range(ndim + 2)),) * 3))

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        y = self.bias
        y += jax.lax.conv_transpose(
            lhs=x[None],
            rhs=self.weight,
            strides=self.strides,
            padding=self.padding,
            rhs_dilation=self.rate,
            dimension_numbers=self.dimension_numbers,
        )

        return y[0]


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
        rate=1,
        weight_init_func=jax.nn.initializers.glorot_uniform(),
        bias_init_func=jax.nn.initializers.zeros,
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
        rate=1,
        weight_init_func=jax.nn.initializers.glorot_uniform(),
        bias_init_func=jax.nn.initializers.zeros,
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
        rate=1,
        weight_init_func=jax.nn.initializers.glorot_uniform(),
        bias_init_func=jax.nn.initializers.zeros,
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
            ndim=3,
            key=key,
        )


@pytc.treeclass
class DepthwiseConvND:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int = pytc.nondiff_field()  # number of input features
    kernel_size: int | tuple[int, ...] = pytc.nondiff_field()
    strides: int | tuple[int, ...] = pytc.nondiff_field()  # stride of the convolution
    padding: str | int | tuple[tuple[int, int], ...] = pytc.nondiff_field()

    weight_init_func: Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]
    bias_init_func: Callable[[jr.PRNGKey, tuple[int]], jnp.ndarray]
    kernel_dilation: int | tuple[int, ...] = pytc.nondiff_field()

    def __init__(
        self,
        in_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        depth_multiplier: int = 1,
        strides: int | tuple[int, ...] = 1,
        padding: str | int | tuple[tuple[int, int], ...] = "SAME",
        weight_init_func: Callable | None = jax.nn.initializers.glorot_uniform(),
        bias_init_func: Callable | None = jax.nn.initializers.zeros,
        ndim: int = 2,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """Convolutional layer.

        Args:
            in_features (int): Number of input features (i.e. channels).
            kernel_size (int | tuple[int, ...]): Size of the convolution kernel.
            depth_multiplier (int, optional): Number of convolutional filters to apply per input channel. Defaults to 1.
            strides (int | tuple[int, ...], optional): Stride of the convolution. Defaults to 1.
            padding (str | tuple[tuple[int, int], ...], optional): Padding of the convolution. Defaults to "SAME".
            weight_init_func (Callable, optional): Function to initialize the weights.
            bias_init_func (Callable, optional): Function to initialize the bias.
            ndim (int, optional): Number of spatial dimensions. Defaults to 2.
            key (jr.PRNGKey, optional): PRNG key to use for initialization. Defaults to jr.PRNGKey(0).

        Examples:
            >>> l1 = DepthwiseConvND(3, 3, depth_multiplier=2, strides=2, padding="SAME")
            >>> l1(jnp.ones((3, 32, 32))).shape
            (3, 16, 16, 6)

        Note:
            See :
                https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/
                https://github.com/google/flax/blob/main/flax/linen/linear.py
        """
        # type assertions
        assert isinstance(
            in_features, int
        ), f"Expected int for `in_features`. Found {in_features}."

        assert isinstance(
            depth_multiplier, int
        ), f"Expected int for `depth_multiplier`. Found {depth_multiplier}."

        assert isinstance(
            weight_init_func, Callable
        ), f"Expected Callable for `weight_init_func`. Found {weight_init_func}."

        assert isinstance(
            bias_init_func, (Callable, type(None))
        ), f"Expected Callable or None for `bias_init_func`. Found {bias_init_func}."

        # assert proper values
        assert in_features > 0, "`in_features` must be greater than 0."
        assert depth_multiplier > 0, "`depth_multiplier` must be greater than 0."

        self.in_features = in_features
        self.kernel_size = _check_and_return_kernel(kernel_size, ndim)
        self.strides = _check_and_return_strides(strides, ndim)
        self.depth_multiplier = depth_multiplier
        self.rate = _check_and_return_rate(1, ndim)
        self.padding = _check_and_return_padding(padding, self.kernel_size)
        self.weight_init_func = _wrap_partial(weight_init_func)
        self.bias_init_func = _wrap_partial(bias_init_func)

        weight_shape = (depth_multiplier * in_features, 1, *self.kernel_size)  # OIHW
        self.weight = weight_init_func(key, weight_shape)

        bias_shape = (depth_multiplier * in_features, *(1,) * ndim)
        self.bias = bias_init_func(key, bias_shape) if bias_init_func is not None else 0

        self.kernel_dilation = (1,) * ndim
        self.dimension_numbers = ConvDimensionNumbers(*((tuple(range(ndim + 2)),) * 3))

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        y = self.bias
        y += jax.lax.conv_general_dilated(
            lhs=x[None],
            rhs=self.weight,
            window_strides=self.strides,
            padding=self.padding,
            lhs_dilation=self.kernel_dilation,  # image dilation
            rhs_dilation=self.rate,  # kernel dilation
            dimension_numbers=self.dimension_numbers,
            feature_group_count=self.in_features,
        )

        return y[0]


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
        weight_init_func=jax.nn.initializers.glorot_uniform(),
        bias_init_func=jax.nn.initializers.zeros,
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
        depth_multiplier: int = 1,
        strides=1,
        padding="SAME",
        weight_init_func=jax.nn.initializers.glorot_uniform(),
        bias_init_func=jax.nn.initializers.zeros,
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
        weight_init_func=jax.nn.initializers.glorot_uniform(),
        bias_init_func=jax.nn.initializers.zeros,
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


@pytc.treeclass
class SeparableConvND:
    depthwise_conv: DepthwiseConvND
    pointwise_conv: DepthwiseConvND

    in_features: int = pytc.nondiff_field()
    out_features: int = pytc.nondiff_field()
    depth_multiplier: int = pytc.nondiff_field()
    kernel_size: int | tuple[int, ...] = pytc.nondiff_field()
    strides: int | tuple[int, ...] = pytc.nondiff_field()
    rate: int | tuple[int, ...] = pytc.nondiff_field()
    padding: str | int | tuple[tuple[int, int], ...] = pytc.nondiff_field()

    depthwise_weight_init_func: Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]
    pointwise_weight_init_func: Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]
    pointwise_bias_init_func: Callable[[jr.PRNGKey, tuple[int]], jnp.ndarray]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        depth_multiplier: int = 1,
        strides: int | tuple[int, ...] = 1,
        padding: str | int | tuple[tuple[int, int], ...] = "SAME",
        rate: int = 1,
        depthwise_weight_init_func: Callable = jax.nn.initializers.glorot_uniform(),
        pointwise_weight_init_func: Callable = jax.nn.initializers.glorot_uniform(),
        pointwise_bias_init_func: Callable | None = jax.nn.initializers.zeros,
        ndim: int = 2,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """Separable convolutional layer.

        Note:
            See:
                https://en.wikipedia.org/wiki/Separable_filter
                https://keras.io/api/layers/convolution_layers/separable_convolution2d/
                https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/depthwise_conv.py

        Args:
            in_features: number of input features
            out_features: number of output features
            kernel_size: size of the convolutional kernel
            strides: stride of the convolution
            padding: padding of the input
            rate: dilation rate to use for dilated convolution
            depthwise_weight_init_func: function to initialize the depthwise convolution weights
            pointwise_weight_init_func: function to initialize the pointwise convolution weights
            pointwise_bias_init_func: function to initialize the pointwise convolution bias
            ndim: number of spatial dimensions.
            key: random key to use for weight initialization.

        """

        # type assertions
        assert isinstance(
            in_features, int
        ), f"Expected int for `in_features`. Found {in_features}."

        assert isinstance(
            out_features, int
        ), f"Expected int for `out_features`. Found {out_features}."

        assert isinstance(
            depthwise_weight_init_func, Callable
        ), f"Expected Callable for `weight_init_func`. Found {depthwise_weight_init_func}."

        assert isinstance(
            pointwise_weight_init_func, Callable
        ), f"Expected Callable for `bias_init_func`. Found {pointwise_weight_init_func}."

        assert isinstance(
            pointwise_bias_init_func, (Callable, type(None))
        ), f"Expected Callable for `bias_init_func`. Found {pointwise_bias_init_func}."

        # assert proper values
        assert in_features > 0, "`in_features` must be greater than 0."
        assert out_features > 0, "`out_features` must be greater than 0."

        self.in_features = in_features
        self.out_features = out_features
        self.depth_multiplier = depth_multiplier
        self.kernel_size = _check_and_return_kernel(kernel_size, ndim)
        self.strides = _check_and_return_strides(strides, ndim)
        self.rate = _check_and_return_rate(rate, ndim)
        self.padding = _check_and_return_padding(padding, self.kernel_size)
        self.depthwise_weight_init_func = _wrap_partial(depthwise_weight_init_func)
        self.pointwise_weight_init_func = _wrap_partial(pointwise_weight_init_func)
        self.pointwise_bias_init_func = _wrap_partial(pointwise_bias_init_func)

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
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        depth_multiplier=1,
        strides=1,
        padding="SAME",
        rate=1,
        depthwise_weight_init_func=jax.nn.initializers.glorot_uniform(),
        pointwise_weight_init_func=jax.nn.initializers.glorot_uniform(),
        pointwise_bias_init_func=jax.nn.initializers.zeros,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            depth_multiplier=depth_multiplier,
            strides=strides,
            padding=padding,
            rate=rate,
            depthwise_weight_init_func=depthwise_weight_init_func,
            pointwise_weight_init_func=pointwise_weight_init_func,
            pointwise_bias_init_func=pointwise_bias_init_func,
            ndim=1,
            key=key,
        )


@pytc.treeclass
class SeparableConv2D(SeparableConvND):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        depth_multiplier=1,
        strides=1,
        padding="SAME",
        rate=1,
        depthwise_weight_init_func=jax.nn.initializers.glorot_uniform(),
        pointwise_weight_init_func=jax.nn.initializers.glorot_uniform(),
        pointwise_bias_init_func=jax.nn.initializers.zeros,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            depth_multiplier=depth_multiplier,
            strides=strides,
            padding=padding,
            rate=rate,
            depthwise_weight_init_func=depthwise_weight_init_func,
            pointwise_weight_init_func=pointwise_weight_init_func,
            pointwise_bias_init_func=pointwise_bias_init_func,
            ndim=2,
            key=key,
        )


@pytc.treeclass
class SeparableConv3D(SeparableConvND):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        depth_multiplier=1,
        strides=1,
        padding="SAME",
        rate=1,
        depthwise_weight_init_func=jax.nn.initializers.glorot_uniform(),
        pointwise_weight_init_func=jax.nn.initializers.glorot_uniform(),
        pointwise_bias_init_func=jax.nn.initializers.zeros,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            depth_multiplier=depth_multiplier,
            strides=strides,
            padding=padding,
            rate=rate,
            depthwise_weight_init_func=depthwise_weight_init_func,
            pointwise_weight_init_func=pointwise_weight_init_func,
            pointwise_bias_init_func=pointwise_bias_init_func,
            ndim=3,
            key=key,
        )


@pytc.treeclass
class ConvNDLocal:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int = pytc.nondiff_field()  # number of input features
    out_features: int = pytc.nondiff_field()  # number of output features
    kernel_size: int | tuple[int, ...] = pytc.nondiff_field()
    in_size: tuple[int, ...] = pytc.nondiff_field()  # size of input
    strides: int | tuple[int, ...] = pytc.nondiff_field()  # stride of the convolution
    rate: int | tuple[int, ...] = pytc.nondiff_field()
    padding: str | int | tuple[tuple[int, int], ...] = pytc.nondiff_field()

    weight_init_func: Callable[[jr.PRNGKey, tuple[int, ...]], jnp.ndarray]
    bias_init_func: Callable[[jr.PRNGKey, tuple[int]], jnp.ndarray]
    kernel_dilation: int | tuple[int, ...] = pytc.nondiff_field()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, ...],
        *,
        in_size: tuple[int, ...],
        strides: int | tuple[int, ...] = 1,
        padding: str | int | tuple[tuple[int, int], ...] = "SAME",
        rate: int = 1,
        weight_init_func: Callable | None = jax.nn.initializers.glorot_uniform(),
        bias_init_func: Callable | None = jax.nn.initializers.zeros,
        ndim: int = 2,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """Local convolutional layer.

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

        Note:
            See : https://keras.io/api/layers/locally_connected_layers/
        """
        # type assertions
        assert isinstance(
            in_features, int
        ), f"Expected int for `in_features`, got {in_features}."

        assert isinstance(
            out_features, int
        ), f"Expected int for `out_features`, got {out_features}."

        assert isinstance(
            in_size, (tuple)
        ), f"Expected tuple for `in_size`, got {in_size}."

        assert all(
            isinstance(i, int) and i > 0 for i in in_size
        ), f"Expected tuple of positive ints for `in_size`, got {in_size}."

        assert isinstance(
            weight_init_func, Callable
        ), f"Expected Callable for `weight_init_func`. Found {weight_init_func}."

        assert isinstance(
            bias_init_func, (Callable, type(None))
        ), f"Expected Callable or None for `bias_init_func`. Found {bias_init_func}."

        # assert proper values
        assert in_features > 0, "`in_features` must be greater than 0."
        assert out_features > 0, "`out_features` must be greater than 0."

        self.in_features = in_features
        self.out_features = out_features
        self.in_size = _check_and_return_strides(in_size, ndim)
        self.kernel_size = _check_and_return_kernel(kernel_size, ndim)
        self.strides = _check_and_return_strides(strides, ndim)
        self.rate = _check_and_return_rate(rate, ndim)
        self.padding = _check_and_return_padding(padding, self.kernel_size)
        self.weight_init_func = _wrap_partial(weight_init_func)
        self.bias_init_func = _wrap_partial(bias_init_func)

        self.out_size = _output_shape(
            self.in_size, self.kernel_size, self.padding, self.strides
        )

        # OIHW
        self.weight_shape = (
            self.out_features,
            self.in_features * ft.reduce(op.mul, self.kernel_size),
            *self.out_size,
        )

        self.weight = weight_init_func(key, self.weight_shape)

        bias_shape = (self.out_features, *self.out_size)
        self.bias = bias_init_func(key, bias_shape) if bias_init_func is not None else 0

        self.kernel_dilation = (1,) * ndim
        self.dimension_numbers = ConvDimensionNumbers(*((tuple(range(ndim + 2)),) * 3))

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        y = self.bias
        y += jax.lax.conv_general_dilated_local(
            lhs=x[None],
            rhs=self.weight,
            window_strides=self.strides,
            padding=self.padding,
            filter_shape=self.kernel_size,
            lhs_dilation=self.kernel_dilation,  # image dilation
            rhs_dilation=self.rate,  # kernel dilation
            dimension_numbers=self.dimension_numbers,
        )

        return y[0]


@pytc.treeclass
class Conv1DLocal(ConvNDLocal):
    """1D local convolutional layer."""

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        in_size,
        strides=1,
        padding="SAME",
        rate=1,
        weight_init_func=jax.nn.initializers.glorot_uniform(),
        bias_init_func=jax.nn.initializers.zeros,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """1D convolutional layer.

        Args:
            in_features (int): number of input features
            out_features (int): number of output features
            kernel_size (int): size of the convolutional kernel
            stride (int, optional): stride . Defaults to 1.
            padding (str | tuple[tuple[int, int], ...], optional): padding. Defaults to "SAME".
            rate (int, optional): dilation rate. Defaults to 1.
            weight_init_func (Callable | None, optional): weight initialization function.
            bias_init_func (Callable | None, optional): _description_. Defaults to jax.nn.initializers.zeros.
            key (jr.PRNGKey, optional): key for random number generation. Defaults to jr.PRNGKey(0).
        """
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            in_size=in_size,
            strides=strides,
            padding=padding,
            rate=rate,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            ndim=1,
            key=key,
        )


@pytc.treeclass
class Conv2DLocal(ConvNDLocal):
    """2D local convolutional layer."""

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        in_size,
        strides=1,
        padding="SAME",
        rate=1,
        weight_init_func=jax.nn.initializers.glorot_uniform(),
        bias_init_func=jax.nn.initializers.zeros,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """2D local convolutional layer.

        Args:
            in_features (int): number of input features
            out_features (int): number of output features
            kernel_size (int | tuple[int, ...]): size of the convolutional kernel
            stride (int | tuple[int, ...], optional): stride . Defaults to 1.
            padding (str | tuple[tuple[int, int], ...], optional): padding. Defaults to "SAME".
            rate (int, optional): dilation rate. Defaults to 1.
            weight_init_func (Callable | None, optional): weight initialization function.
            bias_init_func (Callable | None, optional): _description_. Defaults to jax.nn.initializers.zeros.
            key (jr.PRNGKey, optional): key for random number generation. Defaults to jr.PRNGKey(0).
        """
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            in_size=in_size,
            strides=strides,
            padding=padding,
            rate=rate,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            ndim=2,
            key=key,
        )


@pytc.treeclass
class Conv3DLocal(ConvNDLocal):
    """3D local convolutional layer."""

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        in_size,
        strides=1,
        padding="SAME",
        rate=1,
        weight_init_func=jax.nn.initializers.glorot_uniform(),
        bias_init_func=jax.nn.initializers.zeros,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """3D local convolutional layer.

        Args:
            in_features (int): number of input features
            out_features (int): number of output features
            kernel_size (int | tuple[int, ...]): size of the convolutional kernel
            stride (int | tuple[int, ...], optional): stride . Defaults to 1.
            padding (str | tuple[tuple[int, int], ...], optional): padding. Defaults to "SAME".
            rate (int, optional): dilation rate. Defaults to 1.
            weight_init_func (Callable | None, optional): weight initialization function.
            bias_init_func (Callable | None, optional): _description_. Defaults to jax.nn.initializers.zeros.
            key (jr.PRNGKey, optional): key for random number generation. Defaults to jr.PRNGKey(0).
        """
        super().__init__(
            in_features,
            out_features,
            kernel_size,
            in_size=in_size,
            strides=strides,
            padding=padding,
            rate=rate,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            ndim=3,
            key=key,
        )
