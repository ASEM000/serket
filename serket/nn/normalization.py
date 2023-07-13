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

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap

import serket as sk
from serket.nn.utils import IsInstance, Range, ScalarLike, positive_int_cb


def layer_norm(
    x: jax.Array,
    *,
    gamma: jax.Array,
    beta: jax.Array,
    eps: float,
    normalized_shape: int | tuple[int],
) -> jax.Array:
    """Layer Normalization
    See: https://nn.labml.ai/normalization/layer_norm/index.html
    transform the input by scaling and shifting to have zero mean and unit variance.

    Args:
        x: input array
        gamma: scale
        beta: shift
        eps: a value added to the denominator for numerical stability.
        normalized_shape: the shape of the input to be normalized.
    """
    dims = tuple(range(len(x.shape) - len(normalized_shape), len(x.shape)))

    μ = jnp.mean(x, axis=dims, keepdims=True)
    σ_2 = jnp.var(x, axis=dims, keepdims=True)
    x̂ = (x - μ) * jax.lax.rsqrt((σ_2 + eps))

    if gamma is not None and beta is not None:
        return x̂ * gamma + beta
    return x̂


def group_norm(
    x: jax.Array,
    *,
    gamma: jax.Array,
    beta: jax.Array,
    eps: float,
    groups: int,
) -> jax.Array:
    """Group Normalization
    See: https://nn.labml.ai/normalization/group_norm/index.html
    transform the input by scaling and shifting to have zero mean and unit variance.

    Args:
        x: input array
        gamma: scale Array
        beta: shift Array
        eps: a value added to the denominator for numerical stability.
        groups: number of groups to separate the channels into
    """
    # split channels into groups
    xx = x.reshape(groups, -1)
    μ = jnp.mean(xx, axis=-1, keepdims=True)
    σ_2 = jnp.var(xx, axis=-1, keepdims=True)
    x̂ = (xx - μ) * jax.lax.rsqrt((σ_2 + eps))
    x̂ = x̂.reshape(*x.shape)

    if gamma is not None and beta is not None:
        gamma = jnp.expand_dims(gamma, axis=range(1, x.ndim))
        beta = jnp.expand_dims(beta, axis=range(1, x.ndim))
        x̂ = x̂ * gamma + beta
    return x̂


class LayerNorm(sk.TreeClass):
    """Layer Normalization
    See: https://nn.labml.ai/normalization/layer_norm/index.html
    transform the input by scaling and shifting to have zero mean and unit variance.

    Args:
        normalized_shape: the shape of the input to be normalized.
        eps: a value added to the denominator for numerical stability.
        affine: a boolean value that when set to True, this module has learnable affine parameters.
    """

    eps: float = sk.field(callbacks=[Range(0), ScalarLike()])

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        *,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        self.normalized_shape = (
            normalized_shape
            if isinstance(normalized_shape, tuple)
            else (normalized_shape,)
        )
        self.eps = eps
        self.affine = affine

        # make gamma and beta trainable
        self.gamma = jnp.ones(normalized_shape) if self.affine else None
        self.beta = jnp.zeros(normalized_shape) if self.affine else None

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        return layer_norm(
            x,
            gamma=self.gamma,
            beta=self.beta,
            eps=self.eps,
            normalized_shape=self.normalized_shape,
        )


class GroupNorm(sk.TreeClass):
    """Group Normalization
    See: https://nn.labml.ai/normalization/group_norm/index.html
    transform the input by scaling and shifting to have zero mean and unit variance.

    Args:
        in_features : the shape of the input to be normalized.
        groups : number of groups to separate the channels into.
        eps : a value added to the denominator for numerical stability.
        affine : a boolean value that when set to True, this module has learnable affine parameters.
    """

    eps: float = sk.field(callbacks=[Range(0), ScalarLike()])

    def __init__(
        self,
        in_features,
        *,
        groups: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        self.in_features = positive_int_cb(in_features)
        self.groups = positive_int_cb(groups)
        self.affine = affine
        self.eps = eps

        # needs more info for checking
        if in_features % groups != 0:
            msg = f"in_features must be divisible by groups. Got {in_features} and {groups}"
            raise ValueError(msg)

        # make gamma and beta trainable
        self.gamma = jnp.ones(self.in_features) if self.affine else None
        self.beta = jnp.zeros(self.in_features) if self.affine else None

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return group_norm(
            x=x,
            gamma=self.gamma,
            beta=self.beta,
            eps=self.eps,
            groups=self.groups,
        )


class InstanceNorm(GroupNorm):
    """Instance Normalization
    See: https://nn.labml.ai/normalization/instance_norm/index.html
    transform the input by scaling and shifting to have zero mean and unit variance.

    Args:
        in_features : the shape of the input to be normalized.
        eps : a value added to the denominator for numerical stability.
        affine : a boolean value that when set to True, this module has learnable affine parameters.
    """

    def __init__(
        self,
        in_features: int,
        *,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__(
            in_features=in_features,
            groups=in_features,
            eps=eps,
            affine=affine,
        )


class BatchNormState(NamedTuple):
    running_mean: jax.Array
    running_var: jax.Array


@custom_vmap
def batchnorm(
    x: jax.Array,
    state: tuple[jax.Array, jax.Array],
    *,
    momentum: float = 0.1,
    eps: float = 1e-5,
    gamma: jax.Array | None = None,
    beta: jax.Array | None = None,
    track_running_stats: bool = False,
):
    del momentum, eps, gamma, beta, track_running_stats
    return x, state


@batchnorm.def_vmap
def _(
    axis_size,
    in_batched,
    x: jax.Array,
    state: tuple[jax.Array, jax.Array],
    *,
    momentum: float = 0.1,
    eps: float = 1e-5,
    track_running_stats: bool = True,
):
    run_mean, run_var = state

    axes = [0] + list(range(2, x.ndim))

    batch_mean, batch_var = jnp.mean(x, axis=axes), jnp.var(x, axis=axes)

    run_mean = jnp.where(
        track_running_stats,
        (1 - momentum) * run_mean + momentum * batch_mean,
        batch_mean,
    )

    run_var = jnp.where(
        track_running_stats,
        (1 - momentum) * run_var + momentum * batch_var * (axis_size / (axis_size - 1)),
        batch_var,
    )
    x_normalized = (x - batch_mean) * jax.lax.rsqrt(batch_var + eps)
    return (x_normalized, (run_mean, run_var)), (True, (True, True))


class BatchNorm(sk.TreeClass):
    in_features: int = sk.field(callbacks=[IsInstance(int), Range(1)])
    momentum: float = sk.field(callbacks=[Range(0, 1), ScalarLike()])
    eps: float = sk.field(callbacks=[Range(0), ScalarLike()])
    track_running_stats: bool = sk.field(callbacks=[IsInstance(bool)])

    def __post_init__(self):
        self.state = BatchNormState(
            jnp.zeros(self.in_features), jnp.ones(self.in_features)
        )
