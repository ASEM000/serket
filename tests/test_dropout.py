import jax.numpy as jnp
import numpy.testing as npt
import pytest
import pytreeclass as pytc

from serket.nn import Dropout, Dropout1D, Dropout2D, Dropout3D, MaxPool2D, RandomApply


def test_dropout():

    x = jnp.array([1, 2, 3, 4, 5])

    npt.assert_allclose(Dropout(1.0)(x), jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    npt.assert_allclose(Dropout(0.0)(x), x)

    layer = Dropout(0.5)
    layer = layer.at[layer == "eval"].set(True, is_leaf=lambda x: x is None)
    assert pytc.is_treeclass_equal(layer, Dropout(0.5, eval=True))
    npt.assert_allclose(layer(x), x)

    with pytest.raises(ValueError):
        Dropout(1.1)

    with pytest.raises(ValueError):
        Dropout(-0.1)

    with pytest.raises(ValueError):
        Dropout(0.5, eval=1)


def test_dropout1d():
    layer = Dropout1D(0.5)
    assert layer(jnp.ones((1, 10))).shape == (1, 10)

    with pytest.raises(ValueError):
        Dropout1D(1.1)

    with pytest.raises(ValueError):
        Dropout1D(-0.1)

    with pytest.raises(ValueError):
        Dropout1D(0.5, eval=1)


def test_dropout2d():
    layer = Dropout2D(0.5)
    assert layer(jnp.ones((1, 10, 10))).shape == (1, 10, 10)

    with pytest.raises(ValueError):
        Dropout2D(1.1)

    with pytest.raises(ValueError):
        Dropout2D(-0.1)

    with pytest.raises(ValueError):
        Dropout2D(0.5, eval=1)


def test_dropout3d():
    layer = Dropout3D(0.5)
    assert layer(jnp.ones((1, 10, 10, 10))).shape == (1, 10, 10, 10)

    with pytest.raises(ValueError):
        Dropout3D(1.1)

    with pytest.raises(ValueError):
        Dropout3D(-0.1)

    with pytest.raises(ValueError):
        Dropout3D(0.5, eval=1)


def test_random_apply():
    layer = RandomApply(MaxPool2D(kernel_size=2, strides=2), p=0.0)
    assert layer(jnp.ones((1, 10, 10))).shape == (1, 10, 10)

    layer = RandomApply(MaxPool2D(kernel_size=2, strides=2), p=1.0)
    assert layer(jnp.ones((1, 10, 10))).shape == (1, 5, 5)
