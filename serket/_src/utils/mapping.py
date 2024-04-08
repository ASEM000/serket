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

import functools as ft
from typing import Callable

import jax
import jax.numpy as jnp
from jax.util import safe_zip

from serket._src.utils.typing import PaddingMode


def kernel_map(
    func: dict,
    shape: tuple[int, ...],
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
    padding_mode: PaddingMode = "constant",
) -> Callable:
    """Minimal implementation of kmap from kernex"""
    # Copyright 2023 Kernex authors
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

    # copied here to avoid requiring kernex as a dependency
    # does not support most of the kernex features
    if isinstance(padding_mode, (int, float)):
        pad_kwargs = dict(mode="constant", constant_values=padding_mode)
    elif isinstance(padding_mode, (str, Callable)):
        pad_kwargs = dict(mode=padding_mode)

    gather_kwargs = dict(
        mode="promise_in_bounds",
        indices_are_sorted=True,
        unique_indices=True,
    )

    def calculate_kernel_map_output_shape(
        shape: tuple[int, ...],
        kernel_size: tuple[int, ...],
        strides: tuple[int, ...],
        border: tuple[tuple[int, int], ...],
    ) -> tuple[int, ...]:
        return tuple(
            (xi + (li + ri) - ki) // si + 1
            for xi, ki, si, (li, ri) in safe_zip(shape, kernel_size, strides, border)
        )

    @ft.partial(jax.profiler.annotate_function, name="general_arange")
    def general_arange(di: int, ki: int, si: int, x0: int, xf: int) -> jax.Array:
        # this function is used to calculate the windows indices for a given dimension
        start, end = -x0 + ((ki - 1) // 2), di + xf - (ki // 2)
        size = end - start

        res = (
            jax.lax.broadcasted_iota(dtype=jnp.int32, shape=(size, ki), dimension=0)
            + jax.lax.broadcasted_iota(dtype=jnp.int32, shape=(ki, size), dimension=0).T
            + (start)
            - ((ki - 1) // 2)
        )

        # res[::si] is slightly slower.
        return (res) if si == 1 else (res)[::si]

    @ft.lru_cache(maxsize=128)
    def recursive_vmap(*, ndim: int):
        def nvmap(n):
            in_axes = [None] * ndim
            in_axes[-n] = 0
            return (
                jax.vmap(lambda *x: x, in_axes=in_axes)
                if n == 1
                else jax.vmap(nvmap(n - 1), in_axes=in_axes)
            )

        return nvmap(ndim)

    @ft.partial(jax.profiler.annotate_function, name="general_product")
    def general_product(*args: jax.Array):
        return recursive_vmap(ndim=len(args))(*args)

    def generate_views(
        shape: tuple[int, ...],
        kernel_size: tuple[int, ...],
        strides: tuple[int, ...],
        border: tuple[tuple[int, int], ...],
    ) -> tuple[jax.Array, ...]:
        dim_range = tuple(
            general_arange(di, ki, si, x0, xf)
            for (di, ki, si, (x0, xf)) in zip(shape, kernel_size, strides, border)
        )
        matrix = general_product(*dim_range)
        return tuple(map(lambda xi, wi: xi.reshape(-1, wi), matrix, kernel_size))

    def absolute_wrapper(*a, **k):
        def map_func(view: tuple[jax.Array, ...], array: jax.Array):
            patch = array.at[ix_(*view)].get(**gather_kwargs)
            return func(patch, *a, **k)

        return map_func

    def ix_(*args):
        """modified version of jnp.ix_"""
        n = len(args)
        output = []
        for i, a in enumerate(args):
            shape = [1] * n
            shape[i] = a.shape[0]
            output.append(jax.lax.broadcast_in_dim(a, shape, (i,)))
        return tuple(output)

    pad_width = tuple([0, max(0, pi[0]) + max(0, pi[1])] for pi in padding)
    args = (shape, kernel_size, strides, padding)
    views = generate_views(*args)
    output_shape = calculate_kernel_map_output_shape(*args)

    def single_call_wrapper(array: jax.Array, *a, **k):
        padded_array = jnp.pad(array, pad_width, **pad_kwargs)
        reduced_func = absolute_wrapper(*a, **k)

        def map_func(view):
            return reduced_func(view, padded_array)

        result = jax.vmap(map_func)(views)
        return result.reshape(*output_shape, *result.shape[1:])

    return single_call_wrapper
