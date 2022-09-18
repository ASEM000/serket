import jax.numpy as jnp
import numpy.testing as npt
from pytreeclass._src.tree_util import is_treeclass_equal

from serket.nn import Dropout


def test_dropout():

    x = jnp.array([1, 2, 3, 4, 5])

    npt.assert_allclose(Dropout(1.0)(x), jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    npt.assert_allclose(Dropout(0.0)(x), x)

    layer = Dropout(0.5)
    layer = layer.at[layer == "eval"].set(True, is_leaf=lambda x: x is None)
    assert is_treeclass_equal(layer, Dropout(0.5, eval=True))
    npt.assert_allclose(layer(x), x)
