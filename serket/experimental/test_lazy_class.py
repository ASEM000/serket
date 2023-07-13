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

import functools as ft

import jax
import jax.numpy as jnp
import pytest

import serket as sk
from serket.experimental import lazy_class


def test_lazy_class():
    @ft.partial(
        lazy_class,
        lazy_keywords=["in_features"],  # -> `in_features` is lazy evaluated
        infer_func=lambda self, x: (x.shape[-1],),
        infer_method_name="__call__",  # -> `infer_func` is applied to `__call__` method
        lazy_marker=None,  # -> `None` is used to indicate a lazy argument
    )
    class LazyLinear(sk.TreeClass):
        weight: jax.Array
        bias: jax.Array

        def __init__(self, in_features: int, out_features: int):
            self.in_features = in_features
            self.out_features = out_features
            self.weight = jax.random.normal(
                jax.random.PRNGKey(0), (in_features, out_features)
            )
            self.bias = jax.random.normal(jax.random.PRNGKey(0), (out_features,))

        def __call__(self, x):
            return x @ self.weight + self.bias

    layer = LazyLinear(None, 20)
    x = jnp.ones([10, 1])

    assert layer(x).shape == (10, 20)

    layer = LazyLinear(None, 20)

    with pytest.raises(ValueError):
        jax.vmap(layer)(jnp.ones([10, 1, 1]))
