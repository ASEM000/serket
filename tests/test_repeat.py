import jax.numpy as jnp

from serket.nn import Resize1D, Resize2D, Resize3D, Upsample1D, Upsample2D, Upsample3D


def test_resize1d():
    assert Resize1D(4)(jnp.ones([1, 2])).shape == (1, 4)


def test_resize2d():
    assert Resize2D(4)(jnp.ones([1, 2, 2])).shape == (1, 4, 4)


def test_resize3d():
    assert Resize3D(4)(jnp.ones([1, 2, 2, 2])).shape == (1, 4, 4, 4)


def test_upsample1d():
    assert Upsample1D(2)(jnp.ones([1, 2])).shape == (1, 4)


def test_upsample2d():
    assert Upsample2D(2)(jnp.ones([1, 2, 2])).shape == (1, 4, 4)
    assert Upsample2D((2, 3))(jnp.ones([1, 2, 2])).shape == (1, 4, 6)


def test_upsample3d():
    assert Upsample3D(2)(jnp.ones([1, 2, 2, 2])).shape == (1, 4, 4, 4)
    assert Upsample3D((2, 3, 4))(jnp.ones([1, 2, 2, 2])).shape == (1, 4, 6, 8)
