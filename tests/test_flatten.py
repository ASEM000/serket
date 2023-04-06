import jax.numpy as jnp

from serket.nn.flatten import Flatten, Unflatten


def test_flatten():
    assert Flatten(0, 1)(jnp.ones([1, 2, 3, 4, 5])).shape == (2, 3, 4, 5)
    assert Flatten(0, 2)(jnp.ones([1, 2, 3, 4, 5])).shape == (6, 4, 5)
    assert Flatten(1, 2)(jnp.ones([1, 2, 3, 4, 5])).shape == (1, 6, 4, 5)
    assert Flatten(-1, -1)(jnp.ones([1, 2, 3, 4, 5])).shape == (1, 2, 3, 4, 5)
    assert Flatten(-2, -1)(jnp.ones([1, 2, 3, 4, 5])).shape == (1, 2, 3, 20)
    assert Flatten(-3, -1)(jnp.ones([1, 2, 3, 4, 5])).shape == (1, 2, 60)


def test_unflatten():
    assert Unflatten(0, (1, 2, 3))(jnp.ones([6])).shape == (1, 2, 3)
    assert Unflatten(0, (1, 2, 3))(jnp.ones([6, 4])).shape == (1, 2, 3, 4)
