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
