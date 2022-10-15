import jax.numpy as jnp

from serket.nn import (
    Repeat1D,
    Repeat2D,
    Repeat3D,
    Resize1D,
    Resize2D,
    Resize3D,
    Upsampling1D,
    Upsampling2D,
    Upsampling3D,
)


def test_repeat1d():
    assert Repeat1D(2)(jnp.ones([1, 2])).shape == (1, 4)


def test_repeat2d():
    assert Repeat2D(2)(jnp.ones([1, 2, 2])).shape == (1, 4, 4)


def test_repeat3d():
    assert Repeat3D(2)(jnp.ones([1, 2, 2, 2])).shape == (1, 4, 4, 4)


def test_resize1d():
    assert Resize1D(4)(jnp.ones([1, 2])).shape == (1, 4)


def test_resize2d():
    assert Resize2D(4)(jnp.ones([1, 2, 2])).shape == (1, 4, 4)


def test_resize3d():
    assert Resize3D(4)(jnp.ones([1, 2, 2, 2])).shape == (1, 4, 4, 4)


def test_upsampling1d():
    assert Upsampling1D(2)(jnp.ones([1, 2])).shape == (1, 4)


def test_upsampling2d():
    assert Upsampling2D(2)(jnp.ones([1, 2, 2])).shape == (1, 4, 4)
    assert Upsampling2D((2, 3))(jnp.ones([1, 2, 2])).shape == (1, 4, 6)


def test_upsampling3d():
    assert Upsampling3D(2)(jnp.ones([1, 2, 2, 2])).shape == (1, 4, 4, 4)
    assert Upsampling3D((2, 3, 4))(jnp.ones([1, 2, 2, 2])).shape == (1, 4, 6, 8)
