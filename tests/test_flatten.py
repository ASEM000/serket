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

from serket.nn.flatten import Flatten, Unflatten


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
