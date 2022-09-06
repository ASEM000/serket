import jax.numpy as jnp
import numpy.testing as npt

import serket as sk


def test_AdaptiveReLU():
    npt.assert_allclose(
        sk.nn.AdaptiveReLU(1.0)(jnp.array([1.0, 2.0, 3.0])), jnp.array([1.0, 2.0, 3.0])
    )
    npt.assert_allclose(
        sk.nn.AdaptiveReLU(0.0)(jnp.array([1.0, 2.0, 3.0])), jnp.array([0.0, 0.0, 0.0])
    )
    npt.assert_allclose(
        sk.nn.AdaptiveReLU(0.5)(jnp.array([1.0, 2.0, 3.0])), jnp.array([0.5, 1.0, 1.5])
    )


def test_AdaptiveLeakyReLU():
    npt.assert_allclose(
        sk.nn.AdaptiveLeakyReLU(0, 1.0)(jnp.array([1.0, 2.0, 3.0])),
        jnp.array([0, 0, 0]),
    )
    npt.assert_allclose(
        sk.nn.AdaptiveLeakyReLU(0, 0.5)(jnp.array([1.0, 2.0, 3.0])),
        jnp.array([0, 0, 0]),
    )
    npt.assert_allclose(
        sk.nn.AdaptiveLeakyReLU(1, 0)(jnp.array([1.0, 2.0, 3.0])), jnp.array([1, 2, 3])
    )
    npt.assert_allclose(
        sk.nn.AdaptiveLeakyReLU(1, 0.5)(jnp.array([1.0, 2.0, 3.0])),
        jnp.array([1, 2, 3]),
    )


def test_AdaptiveSigmoid():
    npt.assert_allclose(
        sk.nn.AdaptiveSigmoid(1.0)(jnp.array([1.0, 2.0, 3.0])),
        jnp.array([0.7310586, 0.880797, 0.95257413]),
    )
    npt.assert_allclose(
        sk.nn.AdaptiveSigmoid(0)(jnp.array([1.0, 2.0, 3.0])), jnp.array([0.5, 0.5, 0.5])
    )


def test_AdaptiveTanh():
    npt.assert_allclose(
        sk.nn.AdaptiveTanh(1)(jnp.array([1.0, 2.0, 3.0])),
        jnp.array([0.7615942, 0.9640276, 0.9950547]),
    )
    npt.assert_allclose(
        sk.nn.AdaptiveTanh(0)(jnp.array([1.0, 2.0, 3.0])), jnp.array([0, 0, 0])
    )
    npt.assert_allclose(
        sk.nn.AdaptiveTanh(0.5)(jnp.array([1.0, 2.0, 3.0])),
        jnp.array([0.46211714, 0.7615942, 0.9051482]),
    )
