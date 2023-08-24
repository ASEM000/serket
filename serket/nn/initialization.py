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

import functools as ft
from collections.abc import Callable
from typing import Any, Literal, Tuple, Union, get_args

import jax
import jax.nn.initializers as ji
import jax.random as jr
import jax.tree_util as jtu
import numpy as np

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

Shape = Tuple[int, ...]
DType = Union[np.dtype, str, Any]
InitFuncType = Callable[[jr.KeyArray, Shape, DType], jax.Array]
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


@ft.singledispatch
def resolve_init(init):
    raise TypeError(f"Unknown type {type(init)}")


@resolve_init.register(str)
def _(init: str):
    try:
        return jtu.Partial(jax.tree_map(lambda x: x, init_map[init]))
    except KeyError:
        raise ValueError(f"Unknown {init=}, available init: {list(init_map)}")


@resolve_init.register(type(None))
def _(init: None):
    return jtu.Partial(lambda key, shape, dtype=None: None)


@resolve_init.register(Callable)
def _(init: Callable):
    return jtu.Partial(init)


def def_init_entry(key: str, init_func: InitFuncType) -> None:
    """Register a custom initialization function key for use in ``serket`` layers.

    Args:
        key: The key to register the function under.
        init_func: The function to register. must take three arguments: a key,
            a shape, and a dtype, and return an array of the given shape and dtype.

    Note:
        By design initialization function can be passed directly to ``serket`` layers
        without registration. This function is useful if you want to
        represent initialization functions as a string in a configuration file.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import math
        >>> def my_init_func(key, shape, dtype=jnp.float32):
        ...     return jnp.arange(math.prod(shape), dtype=dtype).reshape(shape)
        >>> sk.def_init_entry("my_init", my_init_func)
        >>> sk.nn.Linear(1, 5, weight_init="my_init").weight
        Array([[0., 1., 2., 3., 4.]], dtype=float32)
    """
    import inspect

    signature = inspect.signature(init_func)

    if key in init_map:
        raise ValueError(f"`init_key` {key=} already registered")

    if len(signature.parameters) != 3:
        # verify its a three argument function
        raise ValueError(f"`init_func` {len(signature.parameters)=} != 3")

    init_map[key] = init_func
