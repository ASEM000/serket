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
