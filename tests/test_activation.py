import jax.numpy as jnp
import numpy.testing as npt

from serket.nn import ThresholdedReLU


def test_thresholded_relu():
    x = jnp.array([-1, 0, 1])
    theta = 0.5
    expected = jnp.array([0, 0, 1])
    actual = ThresholdedReLU(theta)(x)
    npt.assert_allclose(actual, expected)
