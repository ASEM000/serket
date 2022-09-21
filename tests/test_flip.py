import jax.numpy as jnp
import numpy.testing as npt

from serket.nn import FlipLeftRight2D, FlipUpDown2D


def test_flip_left_right_2d():
    flip = FlipLeftRight2D()
    x = jnp.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    y = flip(x)
    npt.assert_allclose(y, jnp.array([[[3, 2, 1], [6, 5, 4], [9, 8, 7]]]))


def test_flip_up_down_2d():
    flip = FlipUpDown2D()
    x = jnp.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    y = flip(x)
    npt.assert_allclose(y, jnp.array([[[7, 8, 9], [4, 5, 6], [1, 2, 3]]]))
