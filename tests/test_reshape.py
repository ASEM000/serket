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

import math

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

import serket as sk


def test_random_crop_1d():
    x = jnp.arange(10)[None, :]
    assert sk.nn.RandomCrop1D(size=5)(x, key=jax.random.key(0)).shape == (1, 5)


def test_random_crop_2d():
    x = jnp.arange(25).reshape(1, 5, 5)
    assert sk.nn.RandomCrop2D(size=(3, 3))(x, key=jax.random.key(0)).shape == (
        1,
        3,
        3,
    )


def test_random_crop_3d():
    x = jnp.arange(125).reshape(1, 5, 5, 5)
    assert sk.nn.RandomCrop3D(size=(3, 3, 3))(x, key=jax.random.key(0)).shape == (
        1,
        3,
        3,
        3,
    )


def test_upsample1d():
    assert sk.nn.Upsample1D(2)(jnp.ones([1, 2])).shape == (1, 4)


def test_upsample2d():
    assert sk.nn.Upsample2D(2)(jnp.ones([1, 2, 2])).shape == (1, 4, 4)
    assert sk.nn.Upsample2D((2, 3))(jnp.ones([1, 2, 2])).shape == (1, 4, 6)


def test_upsample3d():
    assert sk.nn.Upsample3D(2)(jnp.ones([1, 2, 2, 2])).shape == (1, 4, 4, 4)
    assert sk.nn.Upsample3D((2, 3, 4))(jnp.ones([1, 2, 2, 2])).shape == (1, 4, 6, 8)


@pytest.mark.parametrize(
    "layer,shape,crop_shape",
    [
        [sk.nn.CenterCrop1D, (2, 5), (2, 3)],
        [sk.nn.CenterCrop2D, (2, 5, 5), (2, 3, 3)],
        [sk.nn.CenterCrop3D, (2, 5, 5, 5), (2, 3, 3, 3)],
    ],
)
def test_center_crop(layer, shape, crop_shape):
    x = jnp.arange(1, math.prod(shape) + 1).reshape(shape)

    assert layer(3)(x).shape == crop_shape
    spatial_ndim = len(shape[1:])
    leading_dim = shape[0]
    assert layer(0)(x).shape == (leading_dim, *[0] * spatial_ndim)
    # x[:, 1:-1, 1:-1]
    start = [0] + [1] * spatial_ndim
    end = [leading_dim] + [3] * spatial_ndim
    npt.assert_allclose(layer(3)(x), jax.lax.dynamic_slice(x, start, end))
