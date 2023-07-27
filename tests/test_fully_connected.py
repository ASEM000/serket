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
import jax.numpy as jnp
import numpy.testing as npt

from serket.nn import FNN, MLP


def test_fnn():
    layer = FNN([1, 2, 3, 4], act_func=("relu", "tanh"))
    assert layer.act_func[0] is not layer.act_func[1]
    assert layer.layers[0] is not layer.layers[1]

    x = jax.random.normal(jax.random.PRNGKey(0), (10, 1))
    w1 = jax.random.normal(jax.random.PRNGKey(1), (1, 5))
    w2 = jax.random.normal(jax.random.PRNGKey(2), (5, 3))
    w3 = jax.random.normal(jax.random.PRNGKey(3), (3, 4))

    y = x @ w1
    y = jnp.tanh(y)
    y = y @ w2
    y = jax.nn.relu(y)
    y = y @ w3

    l1 = FNN([1, 5, 3, 4], act_func=("tanh", "relu"), bias_init=None)
    l1 = l1.at["layers"].at[0].at["weight"].set(w1)
    l1 = l1.at["layers"].at[1].at["weight"].set(w2)
    l1 = l1.at["layers"].at[2].at["weight"].set(w3)

    npt.assert_allclose(l1(x), y)


def test_mlp():
    layer = MLP(
        1,
        4,
        hidden_size=10,
        num_hidden_layers=2,
        act_func=("relu", "tanh"),
        bias_init=None,
    )

    x = jax.random.normal(jax.random.PRNGKey(0), (10, 1))
    w1 = jax.random.normal(jax.random.PRNGKey(1), (1, 10))
    w2 = jax.random.normal(jax.random.PRNGKey(2), (10, 10))
    w3 = jax.random.normal(jax.random.PRNGKey(3), (10, 4))

    y = x @ w1
    y = jax.nn.relu(y)
    y = y @ w2
    y = jnp.tanh(y)
    y = y @ w3

    layer = layer.at["layers"].at[0].at["weight"].set(w1)
    layer = layer.at["layers"].at[1].at["weight"].set(w2[None])
    layer = layer.at["layers"].at[2].at["weight"].set(w3)

    # breakpoint()
    print(layer(x).shape)

    npt.assert_allclose(layer(x), y)


def test_fnn_mlp():
    fnn = FNN(layers=[2, 4, 4, 2], act_func="relu")
    mlp = MLP(2, 2, hidden_size=4, num_hidden_layers=2, act_func="relu")
    x = jax.random.normal(jax.random.PRNGKey(0), (10, 2))
    npt.assert_allclose(fnn(x), mlp(x))
