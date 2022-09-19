import jax.numpy as jnp

from serket.nn.flatten import Flatten


def test_flatten():

    assert Flatten(0, 1)(jnp.ones([1, 2, 3, 4, 5])).shape == (2, 3, 4, 5)

    assert Flatten(0, 2)(jnp.ones([1, 2, 3, 4, 5])).shape == (6, 4, 5)

    assert Flatten(1, 2)(jnp.ones([1, 2, 3, 4, 5])).shape == (1, 6, 4, 5)

    assert Flatten(-1, -1)(jnp.ones([1, 2, 3, 4, 5])).shape == (1, 2, 3, 4, 5)

    assert Flatten(-2, -1)(jnp.ones([1, 2, 3, 4, 5])).shape == (1, 2, 3, 20)

    assert Flatten(-3, -1)(jnp.ones([1, 2, 3, 4, 5])).shape == (1, 2, 60)
