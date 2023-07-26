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

from types import FunctionType
from typing import Callable, Literal, TypeVar, Union, get_args

import jax
import jax.nn.initializers as ji
import jax.random as jr
import jax.tree_util as jtu

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

Shape = TypeVar("Shape")
Dtype = TypeVar("Dtype")
InitFuncType = Callable[[jr.KeyArray, Shape, Dtype], jax.Array]
InitType = Union[InitLiteral, InitFuncType]


inits = [
    ji.he_normal(),
    ji.he_uniform(),
    ji.glorot_normal(),
    ji.glorot_uniform(),
    ji.lecun_normal(),
    ji.lecun_uniform(),
    ji.normal(),
    ji.uniform(),
    ji.ones,
    ji.zeros,
    ji.xavier_normal(),
    ji.xavier_uniform(),
    ji.orthogonal(),
]

init_map: dict[str, InitType] = dict(zip(get_args(InitLiteral), inits))


def resolve_init_func(init_func: str | InitFuncType) -> Callable:
    if isinstance(init_func, FunctionType):
        return jtu.Partial(init_func)

    if isinstance(init_func, str):
        if init_func in init_map:
            func = init_map[init_func]
            func = jtu.Partial(func)
            return func
        raise ValueError(f"value must be one of ({', '.join(init_map.keys())})")

    if init_func is None:
        return jtu.Partial(lambda key, shape, dtype=None: None)

    raise ValueError("Value must be a string or a function.")
