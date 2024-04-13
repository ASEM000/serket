# Copyright 2024 serket authors
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

import functools as ft
from typing import Sequence

import jax
import jax.numpy as jnp

from serket._src.utils.typing import KernelSizeType, PaddingType, StridesType, T


def canonicalize(value, ndim, name: str | None = None):
    if isinstance(value, (int, float)):
        return (value,) * ndim
    if isinstance(value, jax.Array):
        return jnp.repeat(value, ndim)
    if isinstance(value, Sequence):
        if len(value) != ndim:
            raise ValueError(f"{len(value)=} != {ndim=} for {name=} and {value=}.")
        return value
    raise TypeError(f"Expected int or tuple for {name}, got {value=}.")


def tuplify(value: T) -> T | tuple[T]:
    return tuple(value) if isinstance(value, Sequence) else (value,)


def same_padding_along_dim(
    in_dim: int,
    kernel_size: int,
    stride: int,
) -> tuple[int, int]:
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
) -> tuple[tuple[int, int], ...]:
    del in_dim, strides
    if len(padding) != len(kernel_size):
        raise ValueError(f"Length mismatch {len(kernel_size)=}!={len(padding)=}.")

    resolved_padding = [[]] * len(kernel_size)

    for i, item in enumerate(padding):
        if isinstance(item, int):
            resolved_padding[i] = (item, item)  # ex: padding = (1, 2, 3)

        elif isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError(f"Expected tuple of length 2, got {len(item)=}")
            resolved_padding[i] = item

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
