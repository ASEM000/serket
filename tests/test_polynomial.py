import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from serket.nn import Polynomial


def test_polynomial():
    t = Polynomial(1, 1, degree=2)
    t1 = t.linears[0]
    t1 = t1.at["weight"].set(jnp.array([[10.0]]))
    t2 = t.linears[1]
    t2 = t2.at["weight"].set(jnp.array([[20.0]]))
    t = t.at["linears"].set((t1, t2))
    x = jnp.array([1, 2, 3, 4, 5]).reshape(5, 1)
    y = jnp.array([[211.0], [821.0], [1831.0], [3241.0], [5051.0]])

    npt.assert_allclose(t(x), y)


def test_lazy_poly():
    t = Polynomial(None, 1, degree=2)

    with pytest.raises(ValueError):
        jax.vmap(t)(jnp.array([1, 2, 3, 4, 5]).reshape(5, 1))

    assert t(jnp.ones([1, 5])).shape == (1, 1)
