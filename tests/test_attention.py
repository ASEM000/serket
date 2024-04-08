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

import jax.numpy as jnp
import jax.random as jr
import pytest

import serket as sk


def test_attention_shape():
    batch = 3
    num_heads = 2
    qkv_features = 4
    q_length = 4
    kv_length = 2
    mask = jr.uniform(jr.PRNGKey(2), (batch, num_heads, q_length, kv_length))
    mask = (mask > 0.5).astype(jnp.float32)
    q = jr.uniform(jr.PRNGKey(0), (batch, q_length, qkv_features))
    k = jr.uniform(jr.PRNGKey(1), (batch, kv_length, qkv_features))
    v = jr.uniform(jr.PRNGKey(2), (batch, kv_length, qkv_features))
    layer = sk.nn.MultiHeadAttention(
        num_heads,
        qkv_features,
        drop_rate=0.0,
        key=jr.PRNGKey(0),
    )
    assert (layer(q, k, v, mask=mask, key=jr.PRNGKey(0)).shape) == (3, 4, 4)

    with pytest.raises(ValueError):
        sk.nn.MultiHeadAttention(10, 2, key=jr.PRNGKey(0))

    with pytest.raises(ValueError):
        sk.nn.MultiHeadAttention(4, 4, 10, key=jr.PRNGKey(0))

    with pytest.raises(ValueError):
        sk.nn.MultiHeadAttention(4, 4, 4, 10, key=jr.PRNGKey(0))

    with pytest.raises(ValueError):
        sk.nn.MultiHeadAttention(4, 4, 4, 4, 10, key=jr.PRNGKey(0))
