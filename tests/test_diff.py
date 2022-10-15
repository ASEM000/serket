import jax
import jax.numpy as jnp

from serket.operators import diff, diff_and_grad


def test_diff():
    def func(x):
        return x + 2

    f1 = jax.grad(func)
    f2 = diff(func)
    args = (1.0,)
    F1, F2 = f1(*args), f2(*args)
    assert F1 == F2

    f3 = jax.value_and_grad(func)
    f4 = diff_and_grad(func)
    args = (1.0,)
    F3, F4 = f3(*args), f4(*args)
    assert F3 == F4

    assert jnp.array_equal(
        diff(func)(jnp.array([1.0, 2.0, 3.0])), jnp.array([1.0, 1.0, 1.0])
    )

    def func(x, y):
        return x + y

    f1 = jax.grad(func)
    f2 = diff(func)
    args = (1.0, 2.0)
    F1, F2 = f1(*args), f2(*args)
    assert F1 == F2

    def func(x, y, z):
        return x + y + z

    f1 = jax.grad(func)
    f2 = diff(func)
    args = (1.0, 2.0, 3.0)
    F1, F2 = f1(*args), f2(*args)
    assert F1 == F2
