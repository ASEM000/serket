import jax.numpy as jnp
import numpy.testing as npt
import pytest
import pytreeclass as pytc

from serket.nn import Dropout


def test_dropout():
    x = jnp.array([1, 2, 3, 4, 5])

    layer = Dropout(1.0)
    npt.assert_allclose(layer(x), jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]))

    layer = layer.at["p"].set(0.0, is_leaf=pytc.is_frozen)
    npt.assert_allclose(layer(x), x)

    with pytest.raises(ValueError):
        Dropout(1.1)

    with pytest.raises(ValueError):
        Dropout(-0.1)
