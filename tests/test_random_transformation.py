import jax.numpy as jnp
import numpy.testing as npt

from serket.nn import RandomCutout1D, RandomCutout2D


def test_random_cutout_1d():
    layer = RandomCutout1D(3, 1)
    x = jnp.ones((1, 10))
    y = layer(x)
    npt.assert_equal(y.shape, (1, 10))


def test_random_cutout_2d():
    layer = RandomCutout2D((3, 3), 1)
    x = jnp.ones((1, 10, 10))
    y = layer(x)
    npt.assert_equal(y.shape, (1, 10, 10))
