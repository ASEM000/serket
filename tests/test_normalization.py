import jax.numpy as jnp
import numpy.testing as npt

from serket.nn import LayerNorm


def test_LayerNorm():

    layer = LayerNorm((5, 2), affine=False)

    x = jnp.array(
        [
            [10, 12],
            [20, 22],
            [30, 32],
            [40, 42],
            [50, 52],
        ]
    )

    y = jnp.array(
        [
            [-1.4812257, -1.3401566],
            [-0.7758801, -0.63481104],
            [-0.07053456, 0.07053456],
            [0.63481104, 0.7758801],
            [1.3401566, 1.4812257],
        ]
    )

    npt.assert_allclose(layer(x), y, atol=1e-5)
