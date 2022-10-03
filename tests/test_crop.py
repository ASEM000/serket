import jax.numpy as jnp

from serket.nn.crop import Crop1D, Crop2D, _crop_1d, _crop_2d


def test_crop_1d():
    x = jnp.arange(10)[None, :]
    assert jnp.all(_crop_1d(x, 5, 0)[0] == jnp.arange(5))
    assert jnp.all(_crop_1d(x, 5, 5)[0] == jnp.arange(5, 10))
    assert jnp.all(_crop_1d(x, 5, 2)[0] == jnp.arange(2, 7))
    assert jnp.all(_crop_1d(x, 5, 7)[0] == jnp.arange(7, 10))
    assert jnp.all(
        _crop_1d(x, 5, 7, pad_if_needed=True)[0] == jnp.array([7, 8, 9, 0, 0])
    )


def test_crop_2d():
    x = jnp.arange(25).reshape(1, 5, 5)
    assert jnp.all(
        _crop_2d(x, 3, 3, 0, 0)[0] == jnp.array([[0, 1, 2], [5, 6, 7], [10, 11, 12]])
    )
    assert jnp.all(
        _crop_2d(x, 3, 3, 0, 2)[0] == jnp.array([[2, 3, 4], [7, 8, 9], [12, 13, 14]])
    )
    assert jnp.all(
        _crop_2d(x, 3, 3, 2, 0)[0]
        == jnp.array([[10, 11, 12], [15, 16, 17], [20, 21, 22]])
    )
    assert jnp.all(
        _crop_2d(x, 3, 3, 2, 2)[0]
        == jnp.array([[12, 13, 14], [17, 18, 19], [22, 23, 24]])
    )
    assert jnp.all(
        _crop_2d(x, 3, 3, 2, 2, pad_if_needed=True)[0]
        == jnp.array([[12, 13, 14], [17, 18, 19], [22, 23, 24]])
    )
    assert jnp.all(
        _crop_2d(x, 3, 3, 2, 2, pad_if_needed=True, padding_mode="constant")[0]
        == jnp.array([[12, 13, 14], [17, 18, 19], [22, 23, 24]])
    )


def test_Crop1D():
    layer = Crop1D(5, 2)
    x = jnp.arange(10)[None, :]
    assert jnp.all(layer(x) == jnp.arange(2, 7))


def test_Crop2D():
    layer = Crop2D(3, 3, 2, 2)
    x = jnp.arange(25).reshape(1, 5, 5)
    assert jnp.all(layer(x) == jnp.array([[12, 13, 14], [17, 18, 19], [22, 23, 24]]))
