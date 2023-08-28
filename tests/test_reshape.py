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

import jax.numpy as jnp
import numpy.testing as npt

from serket.nn import (
    Pad1D,
    Pad2D,
    Pad3D,
    RandomZoom1D,
    RandomZoom2D,
    RandomZoom3D,
    Resize1D,
    Resize2D,
    Resize3D,
    Upsample1D,
    Upsample2D,
    Upsample3D,
)
from serket.nn.reshape import (
    Crop1D,
    Crop2D,
    Crop3D,
    Flatten,
    RandomCrop1D,
    RandomCrop2D,
    RandomCrop3D,
    Unflatten,
)


def test_flatten():
    assert Flatten(0, 1)(jnp.ones([1, 2, 3, 4, 5])).shape == (2, 3, 4, 5)
    assert Flatten(0, 2)(jnp.ones([1, 2, 3, 4, 5])).shape == (6, 4, 5)
    assert Flatten(1, 2)(jnp.ones([1, 2, 3, 4, 5])).shape == (1, 6, 4, 5)
    assert Flatten(-1, -1)(jnp.ones([1, 2, 3, 4, 5])).shape == (1, 2, 3, 4, 5)
    assert Flatten(-2, -1)(jnp.ones([1, 2, 3, 4, 5])).shape == (1, 2, 3, 20)
    assert Flatten(-3, -1)(jnp.ones([1, 2, 3, 4, 5])).shape == (1, 2, 60)


def test_unflatten():
    assert Unflatten(0, (1, 2, 3))(jnp.ones([6])).shape == (1, 2, 3)
    assert Unflatten(0, (1, 2, 3))(jnp.ones([6, 4])).shape == (1, 2, 3, 4)


def test_crop_1d():
    x = jnp.arange(10)[None, :]
    assert jnp.all(Crop1D(5, 0)(x)[0] == jnp.arange(5))
    assert jnp.all(Crop1D(5, 5)(x)[0] == jnp.arange(5, 10))
    assert jnp.all(Crop1D(5, 2)(x)[0] == jnp.arange(2, 7))
    # this is how jax.lax.dynamic_slice handles it
    assert jnp.all(Crop1D(5, 7)(x)[0] == jnp.array([5, 6, 7, 8, 9]))


def test_crop_2d():
    x = jnp.arange(25).reshape(1, 5, 5)
    y = jnp.array([[0, 1, 2], [5, 6, 7], [10, 11, 12]])
    assert jnp.all(Crop2D((3, 3), (0, 0))(x)[0] == y)

    y = jnp.array([[2, 3, 4], [7, 8, 9], [12, 13, 14]])
    assert jnp.all(Crop2D((3, 3), (0, 2))(x)[0] == y)

    y = jnp.array([[10, 11, 12], [15, 16, 17], [20, 21, 22]])
    assert jnp.all(Crop2D((3, 3), (2, 0))(x)[0] == y)

    y = jnp.array([[12, 13, 14], [17, 18, 19], [22, 23, 24]])
    assert jnp.all(Crop2D((3, 3), (2, 2))(x)[0] == y)

    y = jnp.array([[12, 13, 14], [17, 18, 19], [22, 23, 24]])
    assert jnp.all(Crop2D((3, 3), (2, 2))(x)[0] == y)

    y = jnp.array([[12, 13, 14], [17, 18, 19], [22, 23, 24]])
    assert jnp.all(Crop2D((3, 3), (2, 2))(x)[0] == y)


def test_crop_3d():
    x = jnp.arange(125).reshape(1, 5, 5, 5)
    y = jnp.array(
        [
            [[0, 1, 2], [5, 6, 7], [10, 11, 12]],
            [[25, 26, 27], [30, 31, 32], [35, 36, 37]],
            [[50, 51, 52], [55, 56, 57], [60, 61, 62]],
        ]
    )
    assert jnp.all(Crop3D((3, 3, 3), (0, 0, 0))(x)[0] == y)


def test_random_crop_1d():
    x = jnp.arange(10)[None, :]
    assert RandomCrop1D(size=5)(x).shape == (1, 5)


def test_random_crop_2d():
    x = jnp.arange(25).reshape(1, 5, 5)
    assert RandomCrop2D(size=(3, 3))(x).shape == (1, 3, 3)


def test_random_crop_3d():
    x = jnp.arange(125).reshape(1, 5, 5, 5)
    assert RandomCrop3D(size=(3, 3, 3))(x).shape == (1, 3, 3, 3)


def test_resize1d():
    assert Resize1D(4)(jnp.ones([1, 2])).shape == (1, 4)


def test_resize2d():
    assert Resize2D(4)(jnp.ones([1, 2, 2])).shape == (1, 4, 4)


def test_resize3d():
    assert Resize3D(4)(jnp.ones([1, 2, 2, 2])).shape == (1, 4, 4, 4)


def test_upsample1d():
    assert Upsample1D(2)(jnp.ones([1, 2])).shape == (1, 4)


def test_upsample2d():
    assert Upsample2D(2)(jnp.ones([1, 2, 2])).shape == (1, 4, 4)
    assert Upsample2D((2, 3))(jnp.ones([1, 2, 2])).shape == (1, 4, 6)


def test_upsample3d():
    assert Upsample3D(2)(jnp.ones([1, 2, 2, 2])).shape == (1, 4, 4, 4)
    assert Upsample3D((2, 3, 4))(jnp.ones([1, 2, 2, 2])).shape == (1, 4, 6, 8)


def test_padding1d():
    layer = Pad1D(padding=1)
    assert layer(jnp.ones((1, 1))).shape == (1, 3)


def test_padding2d():
    layer = Pad2D(padding=1)
    assert layer(jnp.ones((1, 1, 1))).shape == (1, 3, 3)

    layer = Pad2D(padding=((1, 2), (3, 4)))
    assert layer(jnp.ones((1, 1, 1))).shape == (1, 4, 8)


def test_padding3d():
    layer = Pad3D(padding=1)
    assert layer(jnp.ones((1, 1, 1, 1))).shape == (1, 3, 3, 3)

    layer = Pad3D(padding=((1, 2), (3, 4), (5, 6)))
    assert layer(jnp.ones((1, 1, 1, 1))).shape == (1, 4, 8, 12)


def test_random_zoom():
    npt.assert_allclose(RandomZoom1D((0, 0))(jnp.ones((10, 5))), jnp.ones((10, 5)))

    npt.assert_allclose(
        RandomZoom2D((0.5, 0.5))(jnp.ones((10, 5, 5))).shape, (10, 5, 5)
    )

    npt.assert_allclose(
        RandomZoom3D((0.5, 0.5))(jnp.ones((10, 5, 5, 5))).shape, (10, 5, 5, 5)
    )
