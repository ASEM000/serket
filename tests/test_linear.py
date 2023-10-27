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

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy.testing as npt
import pytest

import serket as sk


def test_embed():
    table = sk.nn.Embedding(10, 3, key=jax.random.PRNGKey(0))
    x = jnp.array([9])
    npt.assert_allclose(table(x), jnp.array([[0.43810904, 0.35078037, 0.13254273]]))

    with pytest.raises(TypeError):
        table(jnp.array([9.0]))


def test_linear():
    x = jnp.linspace(0, 1, 100)[:, None]
    y = x**3 + jax.random.uniform(jax.random.PRNGKey(0), (100, 1)) * 0.01

    @jax.value_and_grad
    def loss_func(NN, x, y):
        NN = sk.tree_unmask(NN)
        return jnp.mean((NN(x) - y) ** 2)

    @jax.jit
    def update(NN, x, y):
        value, grad = loss_func(NN, x, y)
        return value, jtu.tree_map(lambda x, g: x - 1e-3 * g, NN, grad)

    nn = sk.nn.FNN(
        [1, 128, 128, 1],
        act="relu",
        weight_init="he_normal",
        bias_init="ones",
        key=jax.random.PRNGKey(0),
    )

    nn = sk.tree_mask(nn)

    for _ in range(20_000):
        value, nn = update(nn, x, y)

    npt.assert_allclose(jnp.array(4.933563e-05), value, atol=1e-3)

    layer = sk.nn.Linear(1, 1, bias_init=None, key=jax.random.PRNGKey(0))
    w = jnp.array([[-0.31568417]])
    layer = layer.at["weight"].set(w)
    y = jnp.array([[-0.31568417]])
    npt.assert_allclose(layer(jnp.array([[1.0]])), y)

    layer = sk.nn.Linear(None, 1, bias_init="zeros", key=jax.random.PRNGKey(0))
    _, layer = layer.at["__call__"](jnp.ones([100, 2]))
    assert layer.in_features == (2,)


def test_bilinear():
    W = jnp.array(
        [
            [[-0.246, -0.3016], [-0.5532, 0.4251], [0.0983, 0.4425], [-0.1003, 0.1923]],
            [[0.4584, -0.5352], [-0.449, 0.1154], [-0.3347, 0.3776], [0.2751, -0.0284]],
            [
                [-0.4469, 0.3681],
                [-0.2142, -0.0545],
                [-0.5095, -0.2242],
                [-0.4428, 0.2033],
            ],
        ]
    )

    x1 = jnp.array([[-0.7676, -0.7205, -0.0586]])
    x2 = jnp.array([[0.4600, -0.2508, 0.0115, 0.6155]])
    y = jnp.array([[-0.3001916, 0.28336674]])
    layer = sk.nn.Linear((3, 4), 2, bias_init=None, key=jax.random.PRNGKey(0))
    layer = layer.at["weight"].set(W)

    npt.assert_allclose(y, layer(x1, x2), atol=1e-4)

    layer = sk.nn.Linear((3, 4), 2, bias_init="zeros", key=jax.random.PRNGKey(0))
    layer = layer.at["weight"].set(W)

    npt.assert_allclose(y, layer(x1, x2), atol=1e-4)


def test_identity():
    x = jnp.array([[1, 2, 3], [4, 5, 6]])
    layer = sk.nn.Identity()
    npt.assert_allclose(x, layer(x))


def test_multi_linear():
    x = jnp.linspace(0, 1, 100)[:, None]
    lhs = sk.nn.Linear(1, 10, key=jax.random.PRNGKey(0))
    rhs = sk.nn.Linear((1,), 10, key=jax.random.PRNGKey(0))
    npt.assert_allclose(lhs(x), rhs(x), atol=1e-4)

    with pytest.raises(TypeError):
        sk.nn.Linear([1, 2], 10, key=jax.random.PRNGKey(0))


def test_general_linear():
    x = jnp.ones([1, 2, 3, 4])
    layer = sk.nn.GeneralLinear(
        in_features=(1, 2), in_axes=(0, 1), out_features=5, key=jax.random.PRNGKey(0)
    )
    assert layer(x).shape == (3, 4, 5)

    x = jnp.ones([1, 2, 3, 4])
    layer = sk.nn.GeneralLinear(
        in_features=(1, 2), in_axes=(0, 1), out_features=5, key=jax.random.PRNGKey(0)
    )
    assert layer(x).shape == (3, 4, 5)

    x = jnp.ones([1, 2, 3, 4])
    layer = sk.nn.GeneralLinear(
        in_features=(1, 2), in_axes=(0, -3), out_features=5, key=jax.random.PRNGKey(0)
    )
    assert layer(x).shape == (3, 4, 5)

    x = jnp.ones([1, 2, 3, 4])
    layer = sk.nn.GeneralLinear(
        in_features=(2, 3), in_axes=(1, -2), out_features=5, key=jax.random.PRNGKey(0)
    )
    assert layer(x).shape == (1, 4, 5)

    with pytest.raises(TypeError):
        sk.nn.GeneralLinear(
            in_features=2, in_axes=(1, -2), out_features=5, key=jax.random.PRNGKey(0)
        )

    with pytest.raises(TypeError):
        sk.nn.GeneralLinear(
            in_features=(2, 3), in_axes=2, out_features=5, key=jax.random.PRNGKey(0)
        )

    with pytest.raises(ValueError):
        sk.nn.GeneralLinear(
            in_features=(1,), in_axes=(0, -3), out_features=5, key=jax.random.PRNGKey(0)
        )

    with pytest.raises(TypeError):
        sk.nn.GeneralLinear(
            in_features=(1, "s"),
            in_axes=(0, -3),
            out_features=5,
            key=jax.random.PRNGKey(0),
        )

    with pytest.raises(TypeError):
        sk.nn.GeneralLinear(
            in_features=(1, 2),
            in_axes=(0, "s"),
            out_features=3,
            key=jax.random.PRNGKey(0),
        )


def test_mlp():
    x = jnp.linspace(0, 1, 100)[:, None]

    fnn = sk.nn.FNN([1, 2, 1], key=jax.random.PRNGKey(0))
    mlp = sk.nn.MLP(
        in_features=1,
        out_features=1,
        hidden_features=2,
        num_hidden_layers=1,
        key=jax.random.PRNGKey(0),
    )

    npt.assert_allclose(fnn(x), mlp(x), atol=1e-4)

    fnn = sk.nn.FNN([1, 2, 2, 1], act="tanh", key=jax.random.PRNGKey(0))
    mlp = sk.nn.MLP(
        in_features=1,
        out_features=1,
        hidden_features=2,
        num_hidden_layers=2,
        act="tanh",
        key=jax.random.PRNGKey(0),
    )

    npt.assert_allclose(fnn(x), mlp(x), atol=1e-4)

    x = jax.random.normal(jax.random.PRNGKey(0), (10, 1))
    w1 = jax.random.normal(jax.random.PRNGKey(1), (1, 10))
    w2 = jax.random.normal(jax.random.PRNGKey(2), (10, 10))
    w3 = jax.random.normal(jax.random.PRNGKey(3), (10, 4))

    y = x @ w1
    y = jax.nn.tanh(y)
    y = y @ w2
    y = jax.nn.tanh(y)
    y = y @ w3

    layer = sk.nn.FNN(
        [1, 10, 10, 4],
        act="tanh",
        bias_init=None,
        key=jax.random.PRNGKey(0),
    )
    layer = (
        layer.at["linear_0"]["weight"]
        .set(w1)
        .at["linear_1"]["weight"]
        .set(w2)
        .at["linear_2"]["weight"]
        .set(w3)
    )

    npt.assert_allclose(layer(x), y)

    layer = sk.nn.MLP(
        1,
        4,
        hidden_features=10,
        num_hidden_layers=2,
        act="tanh",
        bias_init=None,
        key=jax.random.PRNGKey(0),
    )

    layer = layer.at["linear_i"]["weight"].set(w1)
    layer = layer.at["linear_h"]["weight"].set(w2[None])
    layer = layer.at["linear_o"]["weight"].set(w3)

    npt.assert_allclose(layer(x), y)


def test_fnn_mlp():
    fnn = sk.nn.FNN(layers=[2, 4, 4, 2], act="relu", key=jax.random.PRNGKey(0))
    mlp = sk.nn.MLP(
        2,
        2,
        hidden_features=4,
        num_hidden_layers=2,
        act="relu",
        key=jax.random.PRNGKey(0),
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (10, 2))
    npt.assert_allclose(fnn(x), mlp(x))
