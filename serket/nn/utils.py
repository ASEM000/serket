from __future__ import annotations

import functools as ft
from typing import Any, Callable, Literal, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr


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
ActivationType = Union[ActivationLiteral, Callable[[Any], Any]]

_ACT_FUNC_MAP = {
    "tanh": jax.nn.tanh,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "hard_sigmoid": jax.nn.hard_sigmoid,
    None: lambda x: x,
}


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
InitFuncType = Union[str, Callable[[jr.PRNGKey, Shape, Dtype], jax.Array]]


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
