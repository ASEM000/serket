# Copyright 2024 serket authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from serket._src.nn.activation import (
    ELU,
    GELU,
    GLU,
    CeLU,
    HardShrink,
    HardSigmoid,
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
    SoftPlus,
    SoftShrink,
    SoftSign,
    SquarePlus,
    Swish,
    Tanh,
    TanhShrink,
    ThresholdedReLU,
    resolve_act,
)


def test_thresholded_relu():
    x = jnp.array([-1.0, 0, 1])
    theta = 0.5
    expected = jnp.array([0, 0, 1])
    actual = ThresholdedReLU(theta)(x)
    npt.assert_allclose(actual, expected)


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


def test_square_plus():
    x = jnp.array([-1.0, 0, 1])
    expected = 0.5 * (x + jnp.sqrt(x**2 + 4))
    actual = SquarePlus()(x)
    npt.assert_allclose(actual, expected, atol=1e-4)


def test_resolving():
    with pytest.raises(ValueError):
        resolve_act("nonexistent")


def test_invalid_act_sig():
    with pytest.raises(AssertionError):
        resolve_act(lambda x, y: x)
