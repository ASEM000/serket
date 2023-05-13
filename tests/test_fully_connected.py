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

import jax
import numpy.testing as npt

from serket.nn import FNN, MLP


def test_FNN():
    layer = FNN([1, 2, 3, 4], act_func="relu")
    assert not layer.act_funcs[0] is layer.act_funcs[1]
    assert not layer.layers[0] is layer.layers[1]


def test_mlp():
    fnn = FNN(layers=[2, 4, 4, 2], act_func="relu")
    mlp = MLP(2, 2, hidden_size=4, num_hidden_layers=2, act_func="relu")
    x = jax.random.normal(jax.random.PRNGKey(0), (10, 2))
    npt.assert_allclose(fnn(x), mlp(x))
