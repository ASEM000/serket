import jax
import jax.numpy as jnp
import numpy.testing as npt

from serket.nn import Sequential,Lambda


def test_sequential():

    model = Sequential([Lambda(lambda x:x)])
    assert model(1.) == 1.

    model = Sequential([Lambda(lambda x:x+1),Lambda(lambda x:x+1)])
    assert model(1.) == 3.

    