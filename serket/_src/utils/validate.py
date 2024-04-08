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
import operator as op
from typing import Any, Callable, Sequence, TypeVar

import jax
import numpy as np
from typing_extensions import ParamSpec

from serket import TreeClass, autoinit

P = ParamSpec("P")
T = TypeVar("T")


@autoinit
class Range(TreeClass):
    min_val: float = -float("inf")
    max_val: float = float("inf")
    min_inclusive: bool = True
    max_inclusive: bool = True

    def __call__(self, value: Any):
        lop, ls = (op.ge, "[") if self.min_inclusive else (op.gt, "(")
        rop, rs = (op.le, "]") if self.max_inclusive else (op.lt, ")")

        if lop(value, self.min_val) and rop(value, self.max_val):
            return value

        raise ValueError(f"Not in {ls}{self.min_val}, {self.max_val}{rs} got {value=}.")


@autoinit
class IsInstance(TreeClass):
    klass: type | Sequence[type]

    def __call__(self, value: Any):
        if isinstance(value, self.klass):
            return value
        raise TypeError(f"Expected {self.klass}, got {type(value).__name__}")


class ScalarLike(TreeClass):
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
        raise ValueError(f"Expected inexact type got {value=}")


def validate_pos_int(value):
    """Return if value is a positive integer, otherwise raise an error."""
    if not isinstance(value, int):
        raise ValueError(f"value must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{value=} must be positive.")
    return value


def validate_spatial_ndim(func: Callable[P, T], argnum: int = 0) -> Callable[P, T]:
    """Decorator to validate spatial input shape."""

    @ft.wraps(func)
    def wrapper(self, *args, **kwargs):
        input = args[argnum]
        spatial_ndim = self.spatial_ndim

        if input.ndim != spatial_ndim + 1:
            spatial = ", ".join(("rows", "cols", "depths")[:spatial_ndim])
            name = type(self).__name__
            raise ValueError(
                f"Dimesion mismatch error in inputs of {name}\n"
                f"Input should satisfy:\n"
                f"  - {(spatial_ndim + 1) = } dimension, but got {input.ndim = }.\n"
                f"  - shape of (in_features, {spatial}), but got {input.shape = }.\n"
                + (
                    # maybe the user apply the layer on a batched input
                    "The input should be unbatched (no batch dimension).\n"
                    "To apply on batched input, use `jax.vmap(...)(input)`."
                    if input.ndim == spatial_ndim + 2
                    else ""
                )
            )
        return func(self, *args, **kwargs)

    return wrapper


def validate_in_features_shape(func: Callable[P, T], axis: int) -> Callable[P, T]:
    """Decorator to validate input features."""

    def check_axis_shape(input, in_features: int, axis: int) -> None:
        if input.shape[axis] != in_features:
            raise ValueError(f"Specified {in_features=}, got {input.shape[axis]=}.")
        return input

    @ft.wraps(func)
    def wrapper(self, array, *a, **k):
        check_axis_shape(array, self.in_features, axis)
        return func(self, array, *a, **k)

    return wrapper
