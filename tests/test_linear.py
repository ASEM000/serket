import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy.testing as npt
import pytest
import pytreeclass as pytc

from serket.nn import (
    FNN,
    Bilinear,
    Embedding,
    GeneralLinear,
    Identity,
    Linear,
    MergeLinear,
    Multilinear,
)


def test_embed():
    table = Embedding(10, 3)
    x = jnp.array([9])
    npt.assert_allclose(table(x), jnp.array([[0.43810904, 0.35078037, 0.13254273]]))

    with pytest.raises(TypeError):
        table(jnp.array([9.0]))


def test_linear():
    x = jnp.linspace(0, 1, 100)[:, None]
    y = x**3 + jax.random.uniform(jax.random.PRNGKey(0), (100, 1)) * 0.01

    @jax.value_and_grad
    def loss_func(NN, x, y):
        NN = NN.at[...].apply(pytc.unfreeze, is_leaf=pytc.is_frozen)
        return jnp.mean((NN(x) - y) ** 2)

    @jax.jit
    def update(NN, x, y):
        value, grad = loss_func(NN, x, y)
        return value, jtu.tree_map(lambda x, g: x - 1e-3 * g, NN, grad)

    NN = FNN([1, 128, 128, 1])

    # NN = jtu.tree_map(lambda x: pytc.freeze(x) if pytc.is_nondiff(x) else x, NN)
    NN = NN.at[pytc.bcmap(pytc.is_nondiff)(NN)].apply(pytc.freeze)

    # print(pytc.tree_diagram(NN))
    for _ in range(20_000):
        value, NN = update(NN, x, y)

    npt.assert_allclose(jnp.array(4.933563e-05), value, atol=1e-3)

    layer = Linear(1, 1, bias_init_func=None)
    w = jnp.array([[-0.31568417]])
    layer = layer.at["weight"].set(w)
    y = jnp.array([[-0.31568417]])
    npt.assert_allclose(layer(jnp.array([[1.0]])), y)


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
    layer = Bilinear(3, 4, 2, bias_init_func=None)
    layer = layer.at["weight"].set(W)

    npt.assert_allclose(y, layer(x1, x2), atol=1e-4)

    layer = Bilinear(3, 4, 2, bias_init_func="zeros")
    layer = layer.at["weight"].set(W)

    npt.assert_allclose(y, layer(x1, x2), atol=1e-4)


def test_identity():
    x = jnp.array([[1, 2, 3], [4, 5, 6]])
    layer = Identity()
    npt.assert_allclose(x, layer(x))


def test_multi_linear():
    x = jnp.linspace(0, 1, 100)[:, None]
    lhs = Linear(1, 10)
    rhs = Multilinear((1,), 10)
    npt.assert_allclose(lhs(x), rhs(x), atol=1e-4)

    with pytest.raises(ValueError):
        Multilinear([1, 2], 10)


def test_general_linear():
    x = jnp.ones([1, 2, 3, 4])
    layer = GeneralLinear(in_features=(1, 2), in_axes=(0, 1), out_features=5)
    assert layer(x).shape == (3, 4, 5)

    x = jnp.ones([1, 2, 3, 4])
    layer = GeneralLinear(in_features=(1, 2), in_axes=(0, 1), out_features=5)
    assert layer(x).shape == (3, 4, 5)

    x = jnp.ones([1, 2, 3, 4])
    layer = GeneralLinear(in_features=(1, 2), in_axes=(0, -3), out_features=5)
    assert layer(x).shape == (3, 4, 5)

    x = jnp.ones([1, 2, 3, 4])
    layer = GeneralLinear(in_features=(2, 3), in_axes=(1, -2), out_features=5)
    assert layer(x).shape == (1, 4, 5)

    with pytest.raises(TypeError):
        GeneralLinear(in_features=2, in_axes=(1, -2), out_features=5)

    with pytest.raises(TypeError):
        GeneralLinear(in_features=(2, 3), in_axes=2, out_features=5)

    with pytest.raises(ValueError):
        GeneralLinear(in_features=(1,), in_axes=(0, -3), out_features=5)


def test_merge_linear():
    layer1 = Linear(5, 6)  # 5 input features, 6 output features
    layer2 = Linear(7, 6)  # 7 input features, 6 output features
    merged_layer = MergeLinear(layer1, layer2)  # 12 input features, 6 output features
    x1 = jnp.ones([1, 5])  # 1 sample, 5 features
    x2 = jnp.ones([1, 7])  # 1 sample, 7 features
    y = merged_layer(x1, x2)
    z = layer1(x1) + layer2(x2)
    npt.assert_allclose(y, z, atol=1e-6)

    with pytest.raises(ValueError):
        # output features of layer1 and layer2 mismatch
        l1 = Linear(5, 6)
        l2 = Linear(7, 8)
        MergeLinear(l1, l2)
