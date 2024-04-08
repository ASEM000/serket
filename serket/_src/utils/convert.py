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

from typing import Sequence, TypeVar

import jax
import jax.numpy as jnp

T = TypeVar("T")


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
