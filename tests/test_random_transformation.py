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

import jax.numpy as jnp
import numpy.testing as npt

from serket.nn import RandomCutout1D, RandomCutout2D


def test_random_cutout_1d():
    layer = RandomCutout1D(3, 1)
    x = jnp.ones((1, 10))
    y = layer(x)
    npt.assert_equal(y.shape, (1, 10))


def test_random_cutout_2d():
    layer = RandomCutout2D((3, 3), 1)
    x = jnp.ones((1, 10, 10))
    y = layer(x)
    npt.assert_equal(y.shape, (1, 10, 10))
