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
import pytreeclass as pytc

ActivationLiteral = Literal["tanh", "relu", "sigmoid", "hard_sigmoid"]
ActivationFunctionType = Callable[[jax.typing.ArrayLike], jax.Array]
ActivationType = Union[ActivationLiteral, ActivationFunctionType]

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


InitFuncType = Union[InitLiteral, Callable[[jr.KeyArray, Shape, Dtype], jax.Array]]


@ft.lru_cache(maxsize=128)
def calculate_transpose_padding(
    padding,
    kernel_size,
    input_dilation,
    extra_padding,
):
    """Transpose padding to get the padding for the transpose convolution.

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


def calculate_convolution_output_shape(
    shape: tuple[int, ...],
    kernel_size: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
    strides: tuple[int, ...],
):
    """Compute the shape of the output of a convolutional layer."""
    return tuple(
        (xi + (li + ri) - ki) // si + 1
        for xi, ki, si, (li, ri) in zip(shape, kernel_size, strides, padding)
    )


def resolve_activation(act_func: ActivationType) -> ActivationFunctionType:
    entries = {
        "tanh": jtu.Partial(jax.nn.tanh),
        "relu": jtu.Partial(jax.nn.relu),
        "sigmoid": jtu.Partial(jax.nn.sigmoid),
        "hard_sigmoid": jtu.Partial(jax.nn.hard_sigmoid),
        None: jtu.Partial(lambda x: x),
    }

    # in case the user passes a trainable activation function
    # we need to make a copy of it to avoid unpredictable side effects
    return entries.get(act_func, copy.copy(act_func))


def same_padding_along_dim(in_dim: int, kernel_size: int, stride: int):
    # https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    # di: input dimension
    # ki: kernel size
    # si: stride
    if in_dim % stride == 0:
        pad = max(kernel_size - stride, 0)
    else:
        pad = max(kernel_size - (in_dim % stride), 0)

    return (pad // 2, pad - pad // 2)


def resolve_tuple_padding(
    in_dim: tuple[int, ...],
    padding: PaddingType,
    kernel_size: KernelSizeType,
    strides: StridesType,
):
    if len(padding) != len(kernel_size):
        raise ValueError(f"Length mismatch {len(kernel_size)=}!={len(padding)=}.")

    resolved_padding = [[]] * len(kernel_size)

    for i, item in enumerate(padding):
        if isinstance(item, int):
            resolved_padding[i] = (item, item)  # ex: padding = (1, 2, 3)

        elif isinstance(item, tuple):
            if len(item) != 2:
                # ex: padding = ((1, 2), (3, 4), (5, 6))
                raise ValueError(f"Expected tuple of length 2, got {len(item)=}")
            resolved_padding[i] = item

        elif isinstance(item, str):
            if item.lower() == "same":
                di, ki, si = in_dim[i], kernel_size[i], strides[i]
                resolved_padding[i] = same_padding_along_dim(di, ki, si)

            elif item.lower() == "valid":
                resolved_padding[i] = (0, 0)

            else:
                raise ValueError("Invalid padding, must be in [`same`, `valid`].")

    return tuple(resolved_padding)


def resolve_int_padding(
    in_dim: tuple[int, ...],
    padding: PaddingType,
    kernel_size: KernelSizeType,
    strides: StridesType,
):
    del in_dim, strides
    return ((padding, padding),) * len(kernel_size)


def resolve_string_padding(in_dim, padding, kernel_size, strides):
    if padding.lower() == "same":
        return tuple(
            same_padding_along_dim(di, ki, si)
            for di, ki, si in zip(in_dim, kernel_size, strides)
        )

    if padding.lower() == "valid":
        return ((0, 0),) * len(kernel_size)

    raise ValueError(f'string argument must be in ["same","valid"].Found {padding}')


@ft.lru_cache(maxsize=128)
def delayed_canonicalize_padding(
    in_dim: tuple[int, ...],
    padding: PaddingType,
    kernel_size: KernelSizeType,
    strides: StridesType,
):
    # in case of `str` padding, we need to know the input dimension
    # to calculate the padding thus we need to delay the canonicalization
    # until the call

    if isinstance(padding, int):
        return resolve_int_padding(in_dim, padding, kernel_size, strides)

    if isinstance(padding, str):
        return resolve_string_padding(in_dim, padding, kernel_size, strides)

    if isinstance(padding, tuple):
        return resolve_tuple_padding(in_dim, padding, kernel_size, strides)

    raise ValueError(
        "Expected padding to be of:\n"
        "* int, for same padding along all dimensions\n"
        "* str, for `same` or `valid` padding along all dimensions\n"
        "* tuple of int, for individual padding along each dimension\n"
        "* tuple of tuple of int, for padding before and after each dimension\n"
        f"Got {padding=}."
    )


def canonicalize(value, ndim, *, name: str | None = None):
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


class Range(pytc.TreeClass):
    """Check if the input is in the range [min_val, max_val]."""

    min_val: float = -float("inf")
    max_val: float = float("inf")

    def __call__(self, value: Any):
        if self.min_val <= value <= self.max_val:
            return value
        raise ValueError(
            f"Expected value between {self.min_val} and {self.max_val}, "
            f"got {value} of type {type(value).__name__}."
        )


class IsInstance(pytc.TreeClass):
    """Check if the input is an instance of expected_type."""

    predicted_type: type | Sequence[type]

    def __call__(self, value: Any):
        if isinstance(value, self.predicted_type):
            return value

        raise TypeError(f"Expected {self.predicted_type}, got {type(value).__name__}")


class ScalarLike(pytc.TreeClass):
    """Check if the input is a scalar"""

    def __call__(self, value: Any):
        if isinstance(value, (float, complex)):
            return value
        if (
            isinstance(value, (jax.Array, np.ndarray))
            and np.issubdtype(value.dtype, np.inexact)
            and value.shape == ()
        ):
            return value
        raise ValueError(
            f"Expected value to be a float, complex, or array-like object"
            f", got {type(value)}"
        )


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


def init_func_cb(init_func: str | Callable) -> Callable:
    if isinstance(init_func, FunctionType):
        return jtu.Partial(init_func)

    if isinstance(init_func, str):
        init_map = {
            "he_normal": ji.he_normal(),
            "he_uniform": ji.he_uniform(),
            "glorot_normal": ji.glorot_normal(),
            "glorot_uniform": ji.glorot_uniform(),
            "lecun_normal": ji.lecun_normal(),
            "lecun_uniform": ji.lecun_uniform(),
            "normal": ji.normal(),
            "uniform": ji.uniform(),
            "ones": ji.ones,
            "zeros": ji.zeros,
            "xavier_normal": ji.xavier_normal(),
            "xavier_uniform": ji.xavier_uniform(),
            "orthogonal": ji.orthogonal(),
        }

        if init_func in init_map:
            func = init_map[init_func]
            func.__name__ = init_func + "_init"
            return jtu.Partial(init_map[init_func])
        raise ValueError(f"value must be one of ({', '.join(init_map.keys())})")

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
            raise ValueError(
                f"Specified {in_features=} ,"
                f"but got input with input_features={x.shape[axis]}."
            )
        return x

    @ft.wraps(call_wrapper)
    def wrapper(self, array, *a, **k):
        check_in_features(array, getattr(self, attribute_name), axis)
        return call_wrapper(self, array, *a, **k)

    return wrapper
