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

from __future__ import annotations

import jax
import jax.random as jr

import serket as sk


def test_sequential_without_key():
    layer = sk.Sequential(lambda x: x + 1, lambda x: x * 2)
    assert layer(1, key=jax.random.PRNGKey(0)) == 4


def test_sequential_with_key():
    layer = sk.Sequential(lambda x: x + 1, lambda x: x * 2)
    assert layer(1, key=jr.PRNGKey(0)) == 4


def test_random_apply():
    layer = sk.RandomApply(lambda x: x + 1, rate=1.0)
    assert layer(1, key=jr.PRNGKey(0)) == 2
    layer = sk.RandomApply(lambda x: x + 1, rate=0.0)
    assert layer(1, key=jr.PRNGKey(0)) == 1

    assert sk.tree_eval(layer)(1) == 2


def test_random_choice():
    layer = sk.RandomChoice(lambda x: x + 2, lambda x: x * 2)
    key = jr.PRNGKey(0)
    assert layer(1, key=key) == 3.0
    key = jr.PRNGKey(10)
    assert layer(1, key=key) == 2.0

    # convert all choices to sequential
    assert sk.tree_eval(layer)(1, key=jax.random.PRNGKey(0)) == (1.0 + 2.0) * 2.0
