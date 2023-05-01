from __future__ import annotations

import copy
import functools as ft
from types import FunctionType
from typing import Any, Callable, Literal, Sequence, Tuple, Union

import jax
import jax.nn.initializers as ji
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np


@ft.lru_cache(maxsize=128)
def calculate_transpose_padding(padding, kernel_size, input_dilation, extra_padding):
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


ActivationLiteral = Literal["tanh", "relu", "sigmoid", "hard_sigmoid"]
ActivationFunctionType = Callable[[jax.typing.ArrayLike], jax.Array]
ActivationType = Union[ActivationLiteral, ActivationFunctionType]


def resolve_activation(act_func: ActivationType) -> ActivationFunctionType:
    entries = {
        "tanh": jtu.Partial(jax.nn.tanh),
        "relu": jtu.Partial(jax.nn.relu),
        "sigmoid": jtu.Partial(jax.nn.sigmoid),
        "hard_sigmoid": jtu.Partial(jax.nn.hard_sigmoid),
        None: jtu.Partial(lambda x: x),
    }

    # in case the user passes a trainable activation function
    # we need to make a copy of it to unpredictable side effects
    return entries.get(act_func, copy.copy(act_func))


def calculate_convolution_output_shape(shape, kernel_size, padding, strides):
    """Compute the shape of the output of a convolutional layer."""
    return tuple(
        (xi + (li + ri) - ki) // si + 1
        for xi, ki, si, (li, ri) in zip(shape, kernel_size, strides, padding)
    )


@ft.lru_cache(maxsize=128)
def delayed_canonicalize_padding(
    in_dim: tuple[int, ...],
    padding: tuple[int | tuple[int, int] | str, ...] | int | str,
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
):
    # in case of `str` padding, we need to know the input dimension
    # to calculate the padding thus we need to delay the canonicalization
    # until the call
    def same_padding_along_dim(di: int, ki: int, si: int):
        # https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
        # di: input dimension
        # ki: kernel size
        # si: stride
        if di % si == 0:
            pad = max(ki - si, 0)
        else:
            pad = max(ki - (di % si), 0)

        return (pad // 2, pad - pad // 2)

    def resolve_tuple_padding(padding, kernel_size):
        if len(padding) != len(kernel_size):
            msg = f"Expected padding to be of length {len(kernel_size)}, got {len(padding)}"
            raise ValueError(msg)

        resolved_padding = [[]] * len(kernel_size)

        for i, item in enumerate(padding):
            if isinstance(item, int):
                # ex: padding = (1, 2, 3)
                resolved_padding[i] = (item, item)

            elif isinstance(item, tuple):
                # ex: padding = ((1, 2), (3, 4), (5, 6))
                if len(item) != 2:
                    msg = f"Expected tuple of length 2, got {len(item)}"
                    raise ValueError(msg)
                resolved_padding[i] = item

            elif isinstance(item, str):
                # ex: padding = ("same", "valid", "same")
                if item.lower() == "same":
                    di, ki, si = in_dim[i], kernel_size[i], strides[i]
                    resolved_padding[i] = same_padding_along_dim(di, ki, si)

                elif item.lower() == "valid":
                    resolved_padding[i] = (0, 0)

                else:
                    msg = f'string argument must be in ["same","valid"].Found {item}'
                    raise ValueError(msg)

        return tuple(resolved_padding)

    def resolve_int_padding(padding, kernel_size):
        return ((padding, padding),) * len(kernel_size)

    def resolve_string_padding(padding, kernel_size):
        if padding.lower() == "same":
            return tuple(
                same_padding_along_dim(di, ki, si)
                for di, ki, si in zip(in_dim, kernel_size, strides)
            )

        elif padding.lower() == "valid":
            return ((0, 0),) * len(kernel_size)

        raise ValueError(f'string argument must be in ["same","valid"].Found {padding}')

    if isinstance(padding, int):
        return resolve_int_padding(padding, kernel_size)

    if isinstance(padding, str):
        return resolve_string_padding(padding, kernel_size)

    if isinstance(padding, tuple):
        return resolve_tuple_padding(padding, kernel_size)

    msg = f"Expected padding to be of type int, str or tuple, got {type(padding)}"
    raise ValueError(msg)


Shape = Any
Dtype = Any
KernelSizeType = Union[int, Sequence[int]]
StridesType = Union[int, Sequence[int]]
PaddingType = Union[str, int, Sequence[int], Sequence[Tuple[int, int]]]
DilationType = Union[int, Sequence[int]]


InitLiteral = Literal[
    "he_normal",
    "he_uniform",
    "glorot_normal",
    "glorot_uniform",
    "lecun_normal",
    "lecun_uniform",
    "normal",
    "uniform",
    "ones",
    "zeros",
    "xavier_normal",
    "xavier_uniform",
    "orthogonal",
]


InitFuncType = Union[InitLiteral, Callable[[jr.PRNGKey, Shape, Dtype], jax.Array]]


def canonicalize(value, ndim, name: str | None = None):
    if isinstance(value, int):
        return (value,) * ndim
    if isinstance(value, jax.Array):
        return jnp.repeat(value, ndim)
    if isinstance(value, tuple):
        if len(value) != ndim:
            msg = f"Expected tuple of length {ndim}, got {len(value)}: {value}"
            msg += f" for {name}" if name is not None else ""
            raise ValueError(msg)
        return tuple(value)
    msg = f"Expected int or tuple for , got {value}."
    msg += f" for {name}" if name is not None else ""
    raise ValueError(msg)


def range_cb_factory(min_val: float = -float("inf"), max_val: float = float("inf")):
    """Return a function that checks if the input is in the range [min_val, max_val]."""

    def range_check(value: float):
        if min_val <= value <= max_val:
            return value
        raise ValueError(f"Expected value between {min_val} and {max_val}, got {value}")

    return range_check


def isinstance_factory(expected_type: type | Sequence[type]):
    """Return a function that checks if the input is an instance of expected_type."""

    def instance_check(value: Any):
        if isinstance(value, expected_type):
            return value

        msg = f"Expected value of type {expected_type}, "
        msg += f"got {type(value).__name__}"
        raise TypeError(msg)

    return instance_check


def scalar_like_cb(value: Any):
    """Return a function that checks if the input is a trainable scalar."""
    if isinstance(value, (float, complex)):
        return value
    if (
        isinstance(value, (jax.Array, np.ndarray))
        and np.issubdtype(value.dtype, np.inexact)
        and value.shape == ()
    ):
        return value

    msg = f"Expected value to be a float, complex, or array-like object, got {type(value)}"
    raise ValueError(msg)


def canonicalize_cb(value, ndim, name: str | None = None):
    # in essence this is a type check that allows for int, tuple, and jax.Array
    # canonicalization is done by converting to a tuple of length ndim
    if isinstance(value, int):
        return (value,) * ndim
    if isinstance(value, jax.Array):
        return jnp.repeat(value, ndim)
    if isinstance(value, tuple):
        if len(value) != ndim:
            msg = f"Expected tuple of length {ndim}, got {len(value)}: {value}"
            msg += f" for {name}" if name is not None else ""
            raise ValueError(msg)
        return tuple(value)
    msg = f"Expected int or tuple for , got {value}."
    msg += f" for {name}" if name is not None else ""
    raise ValueError(msg)


def positive_int_cb(value):
    """Return if value is a positive integer, otherwise raise an error."""
    if not isinstance(value, int):
        raise ValueError(f"value must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"value must be positive, got {value!r}")
    return value


def _rename_func(func: Callable, name: str) -> Callable:
    """Rename a function."""
    func.__name__ = name
    return func


_INIT_FUNC_MAP = {
    "he_normal": _rename_func(ji.he_normal(), "he_normal_init"),
    "he_uniform": _rename_func(ji.he_uniform(), "he_uniform_init"),
    "glorot_normal": _rename_func(ji.glorot_normal(), "glorot_normal_init"),
    "glorot_uniform": _rename_func(ji.glorot_uniform(), "glorot_uniform_init"),
    "lecun_normal": _rename_func(ji.lecun_normal(), "lecun_normal_init"),
    "lecun_uniform": _rename_func(ji.lecun_uniform(), "lecun_uniform_init"),
    "normal": _rename_func(ji.normal(), "normal_init"),
    "uniform": _rename_func(ji.uniform(), "uniform_init"),
    "ones": _rename_func(ji.ones, "ones_init"),
    "zeros": _rename_func(ji.zeros, "zeros_init"),
    "xavier_normal": _rename_func(ji.xavier_normal(), "xavier_normal_init"),
    "xavier_uniform": _rename_func(ji.xavier_uniform(), "xavier_uniform_init"),
    "orthogonal": _rename_func(ji.orthogonal(), "orthogonal_init"),
}


def init_func_cb(init_func: str | Callable) -> Callable:
    if isinstance(init_func, FunctionType):
        return jtu.Partial(init_func)

    if isinstance(init_func, str):
        if init_func in _INIT_FUNC_MAP:
            return jtu.Partial(_INIT_FUNC_MAP[init_func])
        raise ValueError(f"value must be one of {list(_INIT_FUNC_MAP.keys())}")

    if init_func is None:
        return None

    raise ValueError("Value must be a string or a function.")


def validate_spatial_in_shape(call_wrapper, *, attribute_name: str):
    """Decorator to validate spatial input shape."""

    def check_spatial_in_shape(x, spatial_ndim: int) -> None:
        spatial_tuple = ("rows", "cols", "depths")
        if x.ndim != spatial_ndim + 1:
            # the extra dimension is for the input features (channels)
            msg = f"Input must be a {spatial_ndim+1}D tensor in shape of "
            msg += f"(in_features, {', '.join(spatial_tuple[:spatial_ndim])}), "
            msg += f"but got {x.shape}.\n"

            if x.ndim == spatial_ndim + 2:
                # maybe the user adding batch dimension by mistake
                msg += "To apply on batched input, use `jax.vmap(layer)(input)`."
            raise ValueError(msg)
        return x

    @ft.wraps(call_wrapper)
    def wrapper(self, array, *a, **k):
        array = check_spatial_in_shape(array, getattr(self, attribute_name))
        return call_wrapper(self, array, *a, **k)

    return wrapper


def validate_in_features(call_wrapper, *, attribute_name: str, axis: int = 0):
    """Decorator to validate input features."""

    def check_in_features(x, in_features: int, axis: int) -> None:
        if x.shape[axis] != in_features:
            msg = f"Specified {in_features=} ,"
            msg += f"but got input with input_features={x.shape[axis]}."
            raise ValueError(msg)
        return x

    @ft.wraps(call_wrapper)
    def wrapper(self, array, *a, **k):
        check_in_features(array, getattr(self, attribute_name), axis)
        return call_wrapper(self, array, *a, **k)

    return wrapper


non_negative_scalar_cbs = [range_cb_factory(0), scalar_like_cb]
