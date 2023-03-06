from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt

from serket.nn import (
    ELU,
    GELU,
    GLU,
    SILU,
    AdaptiveLeakyReLU,
    AdaptiveReLU,
    AdaptiveSigmoid,
    AdaptiveTanh,
    CeLU,
    HardShrink,
    HardSigmoid,
    HardSILU,
    HardSwish,
    HardTanh,
    LeakyReLU,
    LogSigmoid,
    LogSoftmax,
    Mish,
    PReLU,
    ReLU,
    ReLU6,
    SeLU,
    Sigmoid,
    Snake,
    SoftPlus,
    SoftShrink,
    SoftSign,
    Swish,
    Tanh,
    TanhShrink,
    ThresholdedReLU,
)


def test_thresholded_relu():
    x = jnp.array([-1.0, 0, 1])
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


def test_prelu():
    x = jnp.array([-1.0, 0, 1])
    expected = x
    actual = PReLU(1.0)(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_relu():
    x = jnp.array([-1.0, 0, 1])
    expected = jax.nn.relu(x)
    actual = ReLU()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_relu6():
    x = jnp.array([-1.0, 0, 1])
    expected = jax.nn.relu6(x)
    actual = ReLU6()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_sigmoid():
    x = jnp.array([-1.0, 0, 1])
    expected = jax.nn.sigmoid(x)
    actual = Sigmoid()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_softplus():
    x = jnp.array([-1.0, 0, 1])
    expected = jax.nn.softplus(x)
    actual = SoftPlus()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_logsoftmax():
    x = jnp.array([1.0, 2, 3])
    expected = jax.nn.log_softmax(x)
    actual = LogSoftmax()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_logsigmoid():
    x = jnp.array([-1.0, 0, 1])
    expected = jax.nn.log_sigmoid(x)
    actual = LogSigmoid()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_leakyrelu():
    x = jnp.array([-1.0, 0, 1])
    expected = jax.nn.leaky_relu(x)
    actual = LeakyReLU()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_hardtanh():
    x = jnp.array([-1.0, 0, 1])
    expected = jnp.array([-1.0, 0, 1])
    actual = HardTanh()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_hard_silu():
    x = jnp.array([-1.0, 0, 1])
    expected = jax.nn.hard_silu(x)
    actual = HardSILU()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_hardsigmoid():
    x = jnp.array([-1.0, 0, 1])
    expected = jnp.array([0.3333, 0.5000, 0.6667])
    actual = HardSigmoid()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_hardswish():
    x = jnp.array([-1.0, 0, 1])
    expected = jnp.array([-0.3333, 0.0000, 0.6667])
    actual = HardSwish()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_hardshrink():
    x = jnp.array([-1.0, 0, 1])
    expected = jnp.array([-1.0, 0, 1])
    actual = HardShrink()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_softshrink():
    x = jnp.array([-1.0, 0, 1])
    expected = jnp.array([-0.5, 0, 0.5])
    actual = SoftShrink()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_tanhshrink():
    x = jnp.array([-1.0, 0, 1])
    expected = jnp.array([-0.2384, 0.0000, 0.2384])
    actual = TanhShrink()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_softsign():
    x = jnp.array([-1.0, 0, 1])
    expected = jnp.array([-0.5, 0, 0.5])
    actual = SoftSign()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_swish():
    x = jnp.array([-1.0, 0, 1])
    expected = jax.nn.swish(x)
    actual = Swish()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_tanh():
    x = jnp.array([-1.0, 0, 1])
    expected = jax.nn.tanh(x)
    actual = Tanh()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_silu():
    x = jnp.array([-1.0, 0, 1])
    expected = jax.nn.silu(x)
    actual = SILU()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_celu():
    x = jnp.array([-1.0, 0, 1])
    expected = jnp.array([-0.63212055, 0, 1])
    actual = CeLU()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_elu():
    x = jnp.array([-1.0, 0, 1])
    expected = jnp.array([-0.63212055, 0, 1])
    actual = ELU()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_gelu():
    x = jnp.array([-1.0, 0, 1])
    expected = jnp.array([-0.15865525, 0, 0.8413447])
    actual = GELU(approximate=False)(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_glu():
    x = jnp.array([[-1.0, 0, 1, 2]])
    expected = jnp.array([[-0.7311, 0.0000]])
    actual = GLU()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_selu():
    x = jnp.array([-1.0, 0, 1])
    expected = jax.nn.selu(x)
    actual = SeLU()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_hard_sigmoid():
    x = jnp.array([-1.0, 0, 1])
    expected = jax.nn.hard_sigmoid(x)
    actual = HardSigmoid()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_mish():
    x = jnp.array([-1.0, 0, 1])
    expected = jnp.array([-0.3034, 0.0000, 0.8651])
    actual = Mish()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_snake():
    x = jnp.array([-1.0, 0, 1])
    expected = jnp.array([-0.29192656, 0.0, 1.7080734])
    actual = Snake()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)
