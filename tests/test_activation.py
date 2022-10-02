import jax.numpy as jnp
import numpy.testing as npt

from serket.nn import (
    AdaptiveLeakyReLU,
    AdaptiveReLU,
    AdaptiveSigmoid,
    AdaptiveTanh,
    ThresholdedReLU,
)


def test_thresholded_relu():
    x = jnp.array([-1, 0, 1])
    theta = 0.5
    expected = jnp.array([0, 0, 1])
    actual = ThresholdedReLU(theta)(x)
    npt.assert_allclose(actual, expected)


def test_AdaptiveReLU():
    npt.assert_allclose(
        AdaptiveReLU(1.0)(jnp.array([1.0, 2.0, 3.0])), jnp.array([1.0, 2.0, 3.0])
    )
    npt.assert_allclose(
        AdaptiveReLU(0.0)(jnp.array([1.0, 2.0, 3.0])), jnp.array([0.0, 0.0, 0.0])
    )
    npt.assert_allclose(
        AdaptiveReLU(0.5)(jnp.array([1.0, 2.0, 3.0])), jnp.array([0.5, 1.0, 1.5])
    )


def test_AdaptiveLeakyReLU():
    npt.assert_allclose(
        AdaptiveLeakyReLU(0.0, 1.0)(jnp.array([1.0, 2.0, 3.0])),
        jnp.array([0, 0, 0]),
    )
    npt.assert_allclose(
        AdaptiveLeakyReLU(0.0, 0.5)(jnp.array([1.0, 2.0, 3.0])),
        jnp.array([0, 0, 0]),
    )
    npt.assert_allclose(
        AdaptiveLeakyReLU(1.0, 0.0)(jnp.array([1.0, 2.0, 3.0])), jnp.array([1, 2, 3])
    )
    npt.assert_allclose(
        AdaptiveLeakyReLU(1.0, 0.5)(jnp.array([1.0, 2.0, 3.0])),
        jnp.array([1, 2, 3]),
    )


def test_AdaptiveSigmoid():
    npt.assert_allclose(
        AdaptiveSigmoid(1.0)(jnp.array([1.0, 2.0, 3.0])),
        jnp.array([0.7310586, 0.880797, 0.95257413]),
    )
    npt.assert_allclose(
        AdaptiveSigmoid(0.0)(jnp.array([1.0, 2.0, 3.0])), jnp.array([0.5, 0.5, 0.5])
    )


def test_AdaptiveTanh():
    npt.assert_allclose(
        AdaptiveTanh(1.0)(jnp.array([1.0, 2.0, 3.0])),
        jnp.array([0.7615942, 0.9640276, 0.9950547]),
    )
    npt.assert_allclose(
        AdaptiveTanh(0.0)(jnp.array([1.0, 2.0, 3.0])), jnp.array([0, 0, 0])
    )
    npt.assert_allclose(
        AdaptiveTanh(0.5)(jnp.array([1.0, 2.0, 3.0])),
        jnp.array([0.46211714, 0.7615942, 0.9051482]),
    )
