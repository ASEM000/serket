import jax.numpy as jnp

# import jax.tree_util as jtu
import numpy.testing as npt
import pytest

from serket.nn import Bilinear, GeneralLinear, Identity, Linear, Multilinear  # , FNN

# import pytreeclass as pytc


# def test_linear():
#     x = jnp.linspace(0, 1, 100)[:, None]
#     y = x**3 + jax.random.uniform(jax.random.PRNGKey(0), (100, 1)) * 0.01

#     @jax.value_and_grad
#     def loss_func(model, x, y):
#         return jnp.mean((model(x) - y) ** 2)

#     @jax.jit
#     def update(model, x, y):
#         value, grad = loss_func(model, x, y)
#         return value, model - 1e-3 * grad

#     model = FNN([1, 128, 128, 1])

#     model = jtu.tree_map(lambda x: pytc.freeze(x) if pytc.is_nondiff(x) else x, model)
#     print(pytc.tree_diagram(model))
#     for _ in range(20_000):
#         value, model = update(model, x, y)

#     npt.assert_allclose(jnp.array(4.933563e-05), value, atol=1e-3)

#     layer = Linear(1, 1, bias_init_func=None)
#     w = jnp.array([[-0.31568417]])
#     layer = layer.at["weight"].set(w)
#     y = jnp.array([[-0.31568417]])
#     npt.assert_allclose(layer(jnp.array([[1.0]])), y)


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
