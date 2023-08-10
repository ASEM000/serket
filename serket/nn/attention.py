# Copyright 2023 Serket authors
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

import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
from typing_extensions import Annotated

import serket as sk
from serket.nn.initialization import InitType
from serket.nn.utils import maybe_lazy_call, maybe_lazy_init

"""Defines attention layers."""


def split_heads(array: jax.Array, num_heads: int) -> jax.Array:
    return array.reshape(*array.shape[:-1], num_heads, -1)


def merge_heads(array: jax.Array) -> jax.Array:
    return array.reshape(*array.shape[:-2], -1)


def is_lazy_call(instance, *_, **__) -> bool:
    return getattr(instance, "q_features", False) is None


def is_lazy_init(_, num_heads, q_features, *__, **___) -> bool:
    return q_features is None


def infer_q_features(_, q_array, *__, **___) -> int:
    return q_array.shape[-1]


def infer_k_features(_, __, k_array, *___, **____) -> int:
    return k_array.shape[-1]


def infer_v_features(_, __, ___, v_array, *____, **_____) -> int:
    return v_array.shape[-1]


attention_updates = dict(
    q_features=infer_q_features,
    k_features=infer_k_features,
    v_features=infer_v_features,
)


def calculate_attention(
    q_heads: jax.Array,
    k_heads: jax.Array,
    v_heads: jax.Array,
    mask: jax.Array,
    num_heads: int,
    drop_layer: sk.nn.GeneralDropout,
    key: jr.KeyArray = jr.PRNGKey(0),
) -> jax.Array:
    """Applies multi-head attention to the given inputs.

    Args:
        q_array: Query array. [..., q_length, q_features]
        k_array: Key array. [..., k_length, k_features]
        mask: Mask array. [..., num_heads, q_length, kv_length]
        num_heads: Number of attention heads.

    Reference:
        - https://github.com/keras-team/keras/blob/v2.13.1/keras/layers/attention/multi_head_attention.py
        - https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/attention.py
        - https://flax.readthedocs.io/en/latest/_modules/flax/linen/attention.html
    """
    k_depth = k_heads.shape[-1]
    # [..., q_length, head_features*num_heads] -> [..., q_length, num_heads, head_features]
    q_heads = split_heads(q_heads, num_heads)
    # [..., k_length, head_features*num_heads] -> [..., k_length, num_heads, head_features]
    k_heads = split_heads(k_heads, num_heads)
    # [..., v_length, head_features*num_heads] -> [..., v_length, num_heads, head_features]
    v_heads = split_heads(v_heads, num_heads)

    logits = jnp.einsum("...qhd,...khd->...hqk", q_heads, k_heads)
    logits /= jnp.sqrt(k_depth // num_heads)

    # handle mask
    min_num = jnp.finfo(logits.dtype).min
    logits = jnp.where(mask, logits, min_num) if mask is not None else logits

    attention_weight = jax.nn.softmax(logits)
    attention = jnp.einsum("...hqk,...khd->...qhd", attention_weight, v_heads)
    return merge_heads(drop_layer(attention, key=key))


class MultiHeadAttention(sk.TreeClass):
    """Multi-head attention module.

    The module consists of linear projections for query, key, and value
    (q, k, v), followed by the application of scaled dot-product attention
    with optional masking and dropout. It supports multi-head attention where
    the input features are divided into multiple heads to capture different
    aspects of relationships between the input vectors.


    Args:
        num_heads: Number of attention heads.
        q_features: Number of features for the query.
        k_features: Number of features for the key.
        v_features: Number of features for the value.
        out_features: Number of features for the output.
        q_weight_init: Initializer for the query weight. Defaults to glorot_uniform.
        q_bias_init: Initializer for the query bias. Defaults to zeros. use
            ``None`` to disable bias.
        k_weight_init: Initializer for the key weight. Defaults to glorot_uniform.
        k_bias_init: Initializer for the key bias. Defaults to zeros. use
            ``None`` to disable bias.
        v_weight_init: Initializer for the value weight. Defaults to glorot_uniform.
        v_bias_init: Initializer for the value bias. Defaults to zeros. use
            ``None`` to disable bias.
        out_weight_init: Initializer for the output weight. Defaults to glorot_uniform.
        out_bias_init: Initializer for the output bias. Defaults to zeros. use
            ``None`` to disable bias.
        drop_rate: Dropout rate. defaults to 0.0.
        drop_broadcast: Whether to broadcast the dropout mask across the batch
            dimension and the heads dimension. Defaults to False.
        key: Key for the random number generator.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> batch = 3
        >>> num_heads = 2
        >>> q_features = 4
        >>> k_features = 8
        >>> v_features = 6
        >>> q_length = 4
        >>> kv_length = 2
        >>> mask = jr.uniform(jr.PRNGKey(2), (batch, num_heads, q_length, kv_length))
        >>> mask = (mask > 0.5).astype(jnp.float32)
        >>> q = jr.uniform(jr.PRNGKey(0), (batch, q_length, q_features))
        >>> k = jr.uniform(jr.PRNGKey(1), (batch, kv_length, k_features))
        >>> v = jr.uniform(jr.PRNGKey(2), (batch, kv_length, v_features))
        >>> layer = sk.nn.MultiHeadAttention(
        ...    num_heads,
        ...    q_features,
        ...    k_features,
        ...    v_features,
        ...    drop_rate=0.0,
        ... )
        >>> print(layer(q, k, v, mask=mask, key=jr.PRNGKey(0)).shape)
        (3, 4, 4)

    Note:
        - If ``k_features``, ``v_features``, ``out_features`` are not specified,
          they are set to ``q_features``.
        - To disable attention :class:`.Dropout`, use :func:`.tree_eval` on the 
          instantiated layer.

        >>> import serket as sk
        >>> layer = sk.nn.MultiHeadAttention(1, 1)
        >>> print(repr(layer.dropout))
        GeneralDropout(drop_rate=0.0, drop_axes=Ellipsis)
        >>> print(repr(sk.tree_eval(layer).dropout))
        Identity()

    Note:
        :class:`.MultiHeadAttention` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``q_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import jax.random as jr
        >>> import serket as sk
        >>> q = jr.uniform(jr.PRNGKey(0), (3, 2, 6))
        >>> k = jr.uniform(jr.PRNGKey(1), (3, 2, 6))
        >>> v = jr.uniform(jr.PRNGKey(2), (3, 2, 6))
        >>> lazy_layer = sk.nn.MultiHeadAttention(2, None)
        >>> _, materialized_layer = lazy_layer.at["__call__"](q, k, v)
        >>> materialized_layer(q, k, v).shape
        (3, 2, 6)

    Reference:
        - https://github.com/keras-team/keras/blob/v2.13.1/keras/layers/attention/multi_head_attention.py
        - https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/attention.py
        - https://flax.readthedocs.io/en/latest/_modules/flax/linen/attention.html
        - https://arxiv.org/abs/1706.03762
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        num_heads: int,
        q_features: int,
        k_features: int | None = None,
        v_features: int | None = None,
        out_features: int | None = None,
        q_weight_init: InitType = "glorot_uniform",
        q_bias_init: InitType = "zeros",
        k_weight_init: InitType = "glorot_uniform",
        k_bias_init: InitType = "zeros",
        v_weight_init: InitType = "glorot_uniform",
        v_bias_init: InitType = "zeros",
        out_weight_init: InitType = "glorot_uniform",
        out_bias_init: InitType = "zeros",
        drop_rate: float = 0.0,
        drop_broadcast: bool = False,
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        k_features = q_features if k_features is None else k_features
        v_features = q_features if v_features is None else v_features
        out_features = q_features if out_features is None else out_features

        if q_features % num_heads != 0:
            raise ValueError(f"{q_features=} % {num_heads=} != 0.")

        if k_features % num_heads != 0:
            raise ValueError(f"{k_features=} % {num_heads=} != 0.")

        if v_features % num_heads != 0:
            raise ValueError(f"{v_features=} % {num_heads=} != 0.")
        
        if out_features % num_heads != 0:
            raise ValueError(f"{out_features=} % {num_heads=} != 0.")

        head_features = q_features // num_heads
        qkey, kkey, vkey, okey = jr.split(key, 4)

        self.num_heads = num_heads
        drop_axes = (-1, -2) if drop_broadcast else ...
        self.dropout = sk.nn.GeneralDropout(drop_rate, drop_axes)

        self.q_projection = sk.nn.Linear(
            in_features=q_features,
            out_features=head_features * num_heads,
            weight_init=q_weight_init,
            bias_init=q_bias_init,
            key=qkey,
        )

        self.k_projection = sk.nn.Linear(
            in_features=k_features,
            out_features=head_features * num_heads,
            weight_init=k_weight_init,
            bias_init=k_bias_init,
            key=kkey,
        )

        self.v_projection = sk.nn.Linear(
            in_features=v_features,
            out_features=head_features * num_heads,
            weight_init=v_weight_init,
            bias_init=v_bias_init,
            key=vkey,
        )

        self.out_projection = sk.nn.Linear(
            in_features=head_features * num_heads,
            out_features=out_features,
            weight_init=out_weight_init,
            bias_init=out_bias_init,
            key=okey,
        )

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=attention_updates)
    def __call__(
        self,
        q_array: Annotated[jax.Array, "..., q_length, q_features"],
        k_array: Annotated[jax.Array, "..., kv_length, k_features"],
        v_array: Annotated[jax.Array, "..., kv_length, v_features"],
        mask: Annotated[jax.Array, "..., num_heads, q_length, kv_length"] | None = None,
        key: jr.KeyArray = jr.PRNGKey(0),
    ) -> Annotated[jax.Array, "..., q_length, out_features"]:
        """Applies multi-head attention to the given inputs.

        Args:
            q_array: Query array. [..., q_length, q_features]
            k_array: Key array. [..., kv_length, k_features]
            v_array: Value array. [..., kv_length, v_features]
            mask: Mask array. [..., num_heads, q_length, kv_length]
            key: Key for the random number generator.
        """

        # [..., q_length, q_features] -> [..., q_length, head_features*num_heads]
        q_heads = self.q_projection(q_array)
        # [..., k_length, k_features] -> [..., k_length, head_features*num_heads]
        k_heads = self.k_projection(k_array)
        # [..., v_length, v_features] -> [..., v_length, head_features*num_heads]
        v_heads = self.v_projection(v_array)

        attention = calculate_attention(
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
            mask=mask,
            num_heads=self.num_heads,
            drop_layer=self.dropout,
            key=key,
        )

        return self.out_projection(attention)
