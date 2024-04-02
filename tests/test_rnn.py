# Copyright 2023 serket authors
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

import os

os.environ["KERAS_BACKEND"] = "jax"
from itertools import product

import jax.numpy as jnp
import jax.random as jr
import keras
import numpy.testing as npt
import pytest

import serket as sk


def test_simple_rnn():
    key = jr.PRNGKey(0)
    time_step = 3
    in_features = 2
    hidden_features = 3
    input = jr.uniform(key, (time_step, in_features))
    keras_rnn = keras.layers.SimpleRNN(
        hidden_features,
        return_sequences=True,
        return_state=True,
    )
    serket_cell = sk.nn.SimpleRNNCell(in_features, hidden_features, key=key)
    keras_output, *keras_state = keras_rnn(input[None])
    keras_output = keras_output[0]  # drop batch dimension
    i2h, h2h, b = keras_rnn.weights
    serket_cell = (
        serket_cell.at["in_hidden_to_hidden"]["weight"]
        .set(jnp.concatenate([i2h.numpy().T, h2h.numpy().T], axis=-1))
        .at["in_hidden_to_hidden"]["bias"]
        .set(b.numpy())
    )
    serket_rnn = sk.nn.scan_cell(serket_cell)
    serket_output, serket_state = serket_rnn(input, sk.tree_state(serket_cell))
    npt.assert_allclose(keras_output, serket_output, atol=1e-6)
    npt.assert_allclose(keras_state[0][0], serket_state.hidden_state, atol=1e-6)


def test_lstm():
    key = jr.PRNGKey(0)
    time_step = 3
    in_features = 2
    hidden_features = 3
    input = jr.uniform(key, (time_step, in_features))
    keras_rnn = keras.layers.LSTM(
        hidden_features,
        return_sequences=True,
        return_state=True,
    )
    serket_cell = sk.nn.LSTMCell(in_features, hidden_features, key=key)
    keras_output, *keras_state = keras_rnn(input[None])
    keras_output = keras_output[0]  # drop batch dimension
    i2h, h2h, b = keras_rnn.weights

    # serket combines the input and hidden weights
    serket_cell = (
        serket_cell.at["in_hidden_to_hidden"]["weight"]
        .set(jnp.concatenate([i2h.numpy().T, h2h.numpy().T], axis=-1))
        .at["in_hidden_to_hidden"]["bias"]
        .set(b.numpy())
    )
    serket_rnn = sk.nn.scan_cell(serket_cell)
    serket_output, serket_state = serket_rnn(input, sk.tree_state(serket_cell))
    npt.assert_allclose(keras_output, serket_output, atol=1e-6)
    npt.assert_allclose(keras_state[0][0], serket_state.hidden_state, atol=1e-6)
    npt.assert_allclose(keras_state[1][0], serket_state.cell_state, atol=1e-6)


def test_bilstm():
    key = jr.PRNGKey(0)
    time_step = 3
    in_features = 2
    hidden_features = 3
    input = jr.uniform(key, (time_step, in_features))
    keras_rnn = keras.layers.LSTM(
        hidden_features,
        return_sequences=True,
        return_state=True,
    )

    keras_rnn = keras.layers.Bidirectional(
        keras.layers.LSTM(
            hidden_features,
            return_sequences=True,
            return_state=False,
        )
    )

    keras_output = keras_rnn(input[None])

    i2hf, h2hf, bf, i2hb, h2hb, bb = keras_rnn.weights

    i2hf = i2hf.numpy().T
    h2hf = h2hf.numpy().T
    ih2hf = jnp.concatenate([i2hf, h2hf], axis=-1)
    bf = bf.numpy()
    i2hb = i2hb.numpy().T
    h2hb = h2hb.numpy().T
    ih2hb = jnp.concatenate([i2hb, h2hb], axis=-1)
    bb = bb.numpy()

    serket_cell = sk.nn.LSTMCell(in_features, hidden_features, key=key)

    forward_cell = (
        serket_cell.at["in_hidden_to_hidden"]["weight"]
        .set(ih2hf)
        .at["in_hidden_to_hidden"]["bias"]
        .set(bf)
    )
    backward_cell = (
        serket_cell.at["in_hidden_to_hidden"]["weight"]
        .set(ih2hb)
        .at["in_hidden_to_hidden"]["bias"]
        .set(bb)
    )

    state1 = sk.tree_state(forward_cell)
    output1, _ = sk.nn.scan_cell(forward_cell)(input, state1)
    state2 = sk.tree_state(backward_cell)
    output2, _ = sk.nn.scan_cell(backward_cell, reverse=True)(input, state2)
    serket_output = jnp.concatenate([output1, output2], axis=1)

    npt.assert_allclose(keras_output[0], serket_output, atol=1e-6)


@pytest.mark.parametrize(
    ("sk_layer", "keras_layer", "ndim"),
    [
        *product(
            [sk.nn.ConvLSTM1DCell, sk.nn.FFTConvLSTM1DCell],
            [keras.layers.ConvLSTM1D],
            [1],
        ),
        *product(
            [sk.nn.ConvLSTM2DCell, sk.nn.FFTConvLSTM2DCell],
            [keras.layers.ConvLSTM2D],
            [2],
        ),
        *product(
            [sk.nn.ConvLSTM3DCell, sk.nn.FFTConvLSTM3DCell],
            [keras.layers.ConvLSTM3D],
            [3],
        ),
    ],
)
def test_conv_lstm(sk_layer, keras_layer, ndim):
    key = jr.PRNGKey(0)
    time_step = 3
    in_features = 2
    spatial = [5] * ndim
    kernel_size = 3
    hidden_features = 3
    input = jr.uniform(key, (time_step, in_features, *spatial))
    keras_rnn = keras_layer(
        hidden_features,
        kernel_size,
        data_format="channels_first",
        padding="same",
        use_bias=False,
        return_sequences=True,
        return_state=True,
    )
    keras_output, *keras_state = keras_rnn(input[None])
    serket_cell = sk_layer(
        in_features,
        hidden_features,
        kernel_size,
        padding="same",
        key=key,
        bias_init=None,
        recurrent_act="sigmoid",
    )
    w1, w2 = keras_rnn.weights
    serket_cell = (
        serket_cell.at["in_to_hidden"]["weight"]
        .set(jnp.transpose(w1.numpy(), (-1, -2, *range(ndim))))
        .at["hidden_to_hidden"]["weight"]
        .set(jnp.transpose(w2.numpy(), (-1, -2, *range(ndim))))
    )

    state = sk.tree_state(serket_cell, input=input[0])
    sekret_output, serket_state = sk.nn.scan_cell(serket_cell)(input, state)

    npt.assert_allclose(keras_output[0], sekret_output, atol=1e-6)
    npt.assert_allclose(keras_state[0][0], serket_state.hidden_state, atol=1e-6)
    npt.assert_allclose(keras_state[1][0], serket_state.cell_state, atol=1e-6)


def test_dense_cell():
    cell = sk.nn.LinearCell(
        in_features=10,
        hidden_features=10,
        act=lambda x: x,
        weight_init="ones",
        bias_init=None,
        key=jr.PRNGKey(0),
    )
    input = jnp.ones([10, 10])
    state = sk.tree_state(cell)
    output, _ = sk.nn.scan_cell(cell)(input, state)
    # 1x10 @ 10x10 => 1x10
    npt.assert_allclose(output[-1], jnp.ones([10]) * 10.0)
