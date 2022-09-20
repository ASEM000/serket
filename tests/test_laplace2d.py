import jax.numpy as jnp
import numpy.testing as npt

from serket.nn import Laplace2D


def test_laplace2d():

    x = Laplace2D()(jnp.arange(1, 26).reshape([1, 5, 5]).astype(jnp.float32))

    y = jnp.array(
        [
            [
                [4.0, 3.0, 2.0, 1.0, -6.0],
                [-5.0, 0.0, 0.0, 0.0, -11.0],
                [-10.0, 0.0, 0.0, 0.0, -16.0],
                [-15.0, 0.0, 0.0, 0.0, -21.0],
                [-46.0, -27.0, -28.0, -29.0, -56.0],
            ]
        ]
    )

    npt.assert_allclose(x, y)
