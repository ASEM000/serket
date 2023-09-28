# Copyright 2023 serket authors
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
from typing import Any

import jax
import jax.random as jr
import jax.tree_util as jtu
import numpy.testing as npt
import pytest

import serket as sk
from serket._src.nn.initialization import resolve_init
from serket._src.utils import (
    IsInstance,
    ScalarLike,
    canonicalize,
    delayed_canonicalize_padding,
    positive_int_cb,
    resolve_string_padding,
    validate_axis_shape,
    validate_spatial_nd,
)


@pytest.mark.parametrize(
    "init_name",
    [
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
    ],
)
def test_canonicalize_init_string(init_name):
    k = jr.PRNGKey(0)
    assert resolve_init(init_name)(k, (2, 2)).shape == (2, 2)


def test_canonicalize_init_func():
    assert isinstance(resolve_init(jax.nn.initializers.he_normal()), jtu.Partial)
    assert isinstance(resolve_init(None), jtu.Partial)

    with pytest.raises(ValueError):
        resolve_init("invalid")

    with pytest.raises(TypeError):
        resolve_init(1)


def test_canonicalize():
    assert canonicalize(3, 2) == (3, 3)
    assert canonicalize((3, 3), 2) == (3, 3)
    assert canonicalize((3, 3, 3), 3) == (3, 3, 3)

    with pytest.raises(ValueError):
        canonicalize((3, 3), 3)

    with pytest.raises(ValueError):
        canonicalize((3, 3, 3), 2)

    with pytest.raises(ValueError):
        canonicalize((3, 3, 3), 1)

    assert canonicalize(3, 2) == (3, 3)
    assert canonicalize((3, 3), 2) == (3, 3)
    assert canonicalize((3, 3, 3), 3) == (3, 3, 3)

    assert canonicalize(3, 2) == (3, 3)
    assert canonicalize((3, 3), 2) == (3, 3)
    assert canonicalize((3, 3, 3), 3) == (3, 3, 3)
    npt.assert_allclose(canonicalize(jax.numpy.array([1]), 2), jax.numpy.array([1, 1]))


def test_resolve_string_padding():
    with pytest.raises(ValueError):
        resolve_string_padding(1, "invalid", 3, 4)


def test_delayed_padding():
    with pytest.raises(ValueError):
        delayed_canonicalize_padding(1, "invalid", 3, 4)


def test_is_instance_error():
    with pytest.raises(TypeError):
        IsInstance(int)(1.0)


def test_scalar_like_error():
    with pytest.raises(ValueError):
        ScalarLike()(1)


def test_positive_int_cb_error():
    with pytest.raises(ValueError):
        positive_int_cb(1.0)


def test_validate_spatial_nd_error():
    with pytest.raises(ValueError):

        class T:
            @ft.partial(validate_spatial_nd, attribute_name="ndim")
            def __call__(self, x) -> Any:
                return x

            @property
            def ndim(self):
                return 1

        T()(jax.numpy.ones([5]))


def test_validate_axis_shape_error():
    with pytest.raises(ValueError):

        class T:
            @ft.partial(validate_axis_shape, attribute_name="in_dim")
            def __call__(self, x) -> Any:
                return x

            @property
            def in_dim(self):
                return 1

        T()(jax.numpy.ones([5, 5]))


def test_lazy_call():
    layer = sk.nn.Linear(None, 1, key=jax.random.PRNGKey(0))

    with pytest.raises(RuntimeError):
        # calling a lazy layer
        layer(jax.numpy.ones([5, 5]))
