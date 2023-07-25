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

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.custom_batching import custom_vmap

import serket as sk
from serket.nn.evaluation import tree_evaluation
from serket.nn.initialization import InitType, resolve_init_func
from serket.nn.state import tree_state
from serket.nn.utils import Range, ScalarLike, positive_int_cb


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

    if gamma is not None:
        x̂ = x̂ * gamma

    if beta is not None:
        x̂ = x̂ + beta

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

    if gamma is not None:
        gamma = jnp.expand_dims(gamma, axis=range(1, x.ndim))
        x̂ *= gamma

    if beta is not None:
        beta = jnp.expand_dims(beta, axis=range(1, x.ndim))
        x̂ += beta
    return x̂


class LayerNorm(sk.TreeClass):
    """Layer Normalization

    Transform the input by scaling and shifting to have zero mean and unit variance.

    Args:
        normalized_shape: the shape of the input to be normalized.
        eps: a value added to the denominator for numerical stability.
        gamma_init_func: a function to initialize the scale. Defaults to ones.
            if None, the scale is not trainable.
        beta_init_func: a function to initialize the shift. Defaults to zeros.
            if None, the shift is not trainable.
        key: a random key for initialization. Defaults to jax.random.PRNGKey(0).

    Note:
        https://nn.labml.ai/normalization/layer_norm/index.html
    """

    eps: float = sk.field(callbacks=[Range(0, min_inclusive=False), ScalarLike()])

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        *,
        eps: float = 1e-5,
        gamma_init_func: InitType = "ones",
        beta_init_func: InitType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        self.normalized_shape = (
            normalized_shape
            if isinstance(normalized_shape, tuple)
            else (normalized_shape,)
        )
        self.eps = eps
        self.gamma = resolve_init_func(gamma_init_func)(key, self.normalized_shape)
        self.beta = resolve_init_func(beta_init_func)(key, self.normalized_shape)

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

    Transform the input by scaling and shifting to have zero mean and unit variance.

    Args:
        in_features: the shape of the input to be normalized.
        groups: number of groups to separate the channels into.
        eps: a value added to the denominator for numerical stability.
        gamma_init_func: a function to initialize the scale. Defaults to ones.
            if None, the scale is not trainable.
        beta_init_func: a function to initialize the shift. Defaults to zeros.
            if None, the shift is not trainable.
        key: a random key for initialization. Defaults to jax.random.PRNGKey(0).

    Note:
        https://nn.labml.ai/normalization/group_norm/index.html
    """

    eps: float = sk.field(callbacks=[Range(0), ScalarLike()])

    def __init__(
        self,
        in_features,
        *,
        groups: int,
        eps: float = 1e-5,
        gamma_init_func: InitType = "ones",
        beta_init_func: InitType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        self.in_features = positive_int_cb(in_features)
        self.groups = positive_int_cb(groups)
        self.eps = eps

        # needs more info for checking
        if in_features % groups != 0:
            raise ValueError(f"{in_features} must be divisible by {groups=}.")

        self.gamma = resolve_init_func(gamma_init_func)(key, (in_features,))
        self.beta = resolve_init_func(beta_init_func)(key, (in_features,))

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

    Transform the input by scaling and shifting to have zero mean and unit variance.

    Args:
        in_features: the shape of the input to be normalized.
        eps: a value added to the denominator for numerical stability.
        gamma_init_func: a function to initialize the scale. Defaults to ones.
            if None, the scale is not trainable.
        beta_init_func: a function to initialize the shift. Defaults to zeros.
            if None, the shift is not trainable.
        key: a random key for initialization. Defaults to jax.random.PRNGKey(0).

    Note:
        https://nn.labml.ai/normalization/instance_norm/index.html
    """

    def __init__(
        self,
        in_features: int,
        *,
        eps: float = 1e-5,
        gamma_init_func: InitType = "ones",
        beta_init_func: InitType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        super().__init__(
            in_features=in_features,
            groups=in_features,
            eps=eps,
            gamma_init_func=gamma_init_func,
            beta_init_func=beta_init_func,
            key=key,
        )


@sk.autoinit
class BatchNormState(sk.TreeClass):
    running_mean: jax.Array
    running_var: jax.Array


def _batchnorm_impl(
    x: jax.Array,
    state: BatchNormState,
    momentum: float = 0.1,
    eps: float = 1e-3,
    gamma: jax.Array = None,
    beta: jax.Array = None,
    evalution: bool = False,
    axis: int = 1,
):
    # reduce over axis=1
    broadcast_shape = [1] * x.ndim
    broadcast_shape[axis] = x.shape[axis]

    def bn_eval_step(x, state):
        run_mean, run_var = state.running_mean, state.running_var
        run_mean = jnp.reshape(run_mean, broadcast_shape)
        run_var = jnp.reshape(run_var, broadcast_shape)
        output = (x - run_mean) / jnp.sqrt(run_var + eps)

        return output, state

    def bn_train_step(x, state):
        # maybe support axes option
        run_mean, run_var = state.running_mean, state.running_var
        axes = list(range(x.ndim))
        with jax.ensure_compile_time_eval():
            del axes[axis]
        batch_mean = jnp.mean(x, axis=axes, keepdims=True)
        batch_var = jnp.mean(jnp.square(x), axis=axes, keepdims=True) - batch_mean**2
        output = (x - batch_mean) / jnp.sqrt(batch_var + eps)
        run_mean = momentum * run_mean + (1 - momentum) * jnp.squeeze(batch_mean)
        run_var = momentum * run_var + (1 - momentum) * jnp.squeeze(batch_var)
        return output, BatchNormState(run_mean, run_var)

    output, state = jax.lax.cond(evalution, bn_eval_step, bn_train_step, x, state)

    state = jax.lax.stop_gradient(state)

    x = jax.lax.cond(evalution, jax.lax.stop_gradient, lambda x: x, x)

    if gamma is not None:
        output *= jnp.reshape(gamma, broadcast_shape)

    if beta is not None:
        output += jnp.reshape(beta, broadcast_shape)

    return output, state


@custom_vmap
def batchnorm(
    x: jax.Array,
    state: BatchNormState,
    momentum: float = 0.1,
    eps: float = 1e-5,
    gamma: jax.Array | None = None,
    beta: jax.Array | None = None,
    evaluation: bool = False,
    axis: int = 1,
) -> tuple[jax.Array, BatchNormState]:
    del momentum, eps, gamma, beta, evaluation, axis
    # no-op when unbatched
    return x, state


@batchnorm.def_vmap
def _(
    axis_size,
    in_batched,
    x: jax.Array,
    state: BatchNormState,
    momentum: float = 0.99,
    eps: float = 1e-5,
    gamma: jax.Array | None = None,
    beta: jax.Array | None = None,
    evaluation: bool = True,
    axis: int = 1,
) -> tuple[jax.Array, BatchNormState]:
    output = _batchnorm_impl(
        x=x,
        state=state,
        momentum=momentum,
        eps=eps,
        gamma=gamma,
        beta=beta,
        evalution=evaluation,
        axis=axis,
    )
    return output, (True, BatchNormState(True, True))


class BatchNorm(sk.TreeClass):
    """Applies normalization over batched inputs`

    .. warning::
        Works under
            - ``jax.vmap(BatchNorm(...), in_axes=(0, None))(x, state)``
            - ``jax.vmap(BatchNorm(...))(x)``

        otherwise will be a no-op.

    Evaluation behavior:
        - ``output = (x - running_mean) / sqrt(running_var + eps)``

    Training behavior:
        - ``output = (x - batch_mean) / sqrt(batch_var + eps)``
        - ``running_mean = momentum * running_mean + (1 - momentum) * batch_mean``
        - ``running_var = momentum * running_var + (1 - momentum) * batch_var``

    Args:
        in_features: the shape of the input to be normalized.
        momentum: the value used for the ``running_mean`` and ``running_var``
            computation. must be a number between ``0`` and ``1``.
        eps: a value added to the denominator for numerical stability.
        gamma_init_func: a function to initialize the scale. Defaults to ones.
            if None, the scale is not trainable.
        beta_init_func: a function to initialize the shift. Defaults to zeros.
            if None, the shift is not trainable.
        axis: the axis that should be normalized. Defaults to 1.
        evaluation: a boolean value that when set to True, this module will run in
            evaluation mode. In this case, this module will always use the running
            estimates of the batch statistics during training.

    Note:
        https://keras.io/api/layers/normalization_layers/batch_normalization/
    """

    def __init__(
        self,
        in_features: int,
        *,
        momentum: float = 0.99,
        eps: float = 1e-5,
        gamma_init_func: InitType = "ones",
        beta_init_func: InitType = "zeros",
        axis: int = 1,
        evaluation: bool = False,
        key: jr.KeyArray = jr.PRNGKey(0),
    ) -> None:
        self.in_features = in_features
        self.momentum = momentum
        self.eps = eps
        self.gamma = resolve_init_func(gamma_init_func)(key, (in_features,))
        self.beta = resolve_init_func(beta_init_func)(key, (in_features,))
        self.axis = axis
        self.evaluation = evaluation

    def __call__(
        self,
        x: jax.Array,
        state: BatchNormState | None = None,
        **k,
    ) -> jax.Array:
        state = sk.tree_state(self) if state is None else state

        x, state = batchnorm(
            x,
            state,
            self.momentum,
            self.eps,
            self.gamma,
            self.beta,
            self.evaluation,
            self.axis,
        )
        return x, state


@tree_evaluation.def_evalutation(BatchNorm)
def _(batchnorm: BatchNorm) -> BatchNorm:
    return batchnorm.at["evaluation"].set(True)


@tree_state.def_state(BatchNorm)
def batchnorm_init_state(batchnorm: BatchNorm, _) -> BatchNormState:
    running_mean = jnp.zeros([batchnorm.in_features])
    running_var = jnp.ones([batchnorm.in_features])
    return BatchNormState(running_mean, running_var)
