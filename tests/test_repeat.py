import jax.numpy as jnp

from serket.nn import Repeat1D, Repeat2D, Repeat3D


def test_repeat1d():
    assert Repeat1D(2)(jnp.ones([1, 2])).shape == (1, 4)


def test_repeat2d():
    assert Repeat2D(2)(jnp.ones([1, 2, 2])).shape == (1, 4, 4)


def test_repeat3d():
    assert Repeat3D(2)(jnp.ones([1, 2, 2, 2])).shape == (1, 4, 4, 4)
