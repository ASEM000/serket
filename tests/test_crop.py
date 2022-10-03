import jax.numpy as jnp

from serket.nn.crop import Crop1D, Crop2D


def test_crop_1d():
    x = jnp.arange(10)[None, :]
    assert jnp.all(Crop1D(5, 0)(x)[0] == jnp.arange(5))
    assert jnp.all(Crop1D(5, 5)(x)[0] == jnp.arange(5, 10))
    assert jnp.all(Crop1D(5, 2)(x)[0] == jnp.arange(2, 7))
    # this is how jax.lax.dynamic_slice handles it
    assert jnp.all(Crop1D(5, 7)(x)[0] == jnp.array([5, 6, 7, 8, 9]))


def test_crop_2d():
    x = jnp.arange(25).reshape(1, 5, 5)
    y = jnp.array([[0, 1, 2], [5, 6, 7], [10, 11, 12]])
    assert jnp.all(Crop2D(3, 3, 0, 0)(x)[0] == y)

    y = jnp.array([[2, 3, 4], [7, 8, 9], [12, 13, 14]])
    assert jnp.all(Crop2D(3, 3, 0, 2)(x)[0] == y)

    y = jnp.array([[10, 11, 12], [15, 16, 17], [20, 21, 22]])
    assert jnp.all(Crop2D(3, 3, 2, 0)(x)[0] == y)

    y = jnp.array([[12, 13, 14], [17, 18, 19], [22, 23, 24]])
    assert jnp.all(Crop2D(3, 3, 2, 2)(x)[0] == y)

    y = jnp.array([[12, 13, 14], [17, 18, 19], [22, 23, 24]])
    assert jnp.all(Crop2D(3, 3, 2, 2)(x)[0] == y)

    y = jnp.array([[12, 13, 14], [17, 18, 19], [22, 23, 24]])
    assert jnp.all(Crop2D(3, 3, 2, 2)(x)[0] == y)
