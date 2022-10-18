import jax
import jax.numpy as jnp
import numpy.testing as npt

from serket.nn import RandomCutout1D, RandomCutout2D


def test_random_cutout_1d():

    assert jnp.all(
        RandomCutout1D(5)(jnp.ones((1, 10)) * 100, key=jax.random.PRNGKey(0))
        == jnp.array([[100.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]])
    )


def test_random_cutout_2d():
    x = jnp.ones((1, 10, 10)) * 100
    y = jnp.array(
        [
            [
                [100.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
                [100.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
                [100.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
                [100.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
                [100.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
                [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            ]
        ]
    )

    npt.assert_allclose(
        RandomCutout2D((5, 5))(x, key=jax.random.PRNGKey(0)), y, atol=1e-5
    )

    npt.assert_allclose(RandomCutout2D(5)(x, key=jax.random.PRNGKey(0)), y, atol=1e-5)
