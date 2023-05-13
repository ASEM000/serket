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

from serket.nn import Pad1D, Pad2D, Pad3D


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
