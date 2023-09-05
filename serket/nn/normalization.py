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
from jax.custom_batching import custom_vmap

import serket as sk
from serket.custom_transform import tree_eval, tree_state
from serket.nn.initialization import DType, InitType, resolve_init
from serket.utils import (
    Range,
    ScalarLike,
    maybe_lazy_call,
    maybe_lazy_init,
    positive_int_cb,
)


def layer_norm(
    x: jax.Array,
    *,
    gamma: jax.Array,
    beta: jax.Array,
    eps: float,
    normalized_shape: int | tuple[int],
) -> jax.Array:
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


def is_lazy_call(instance, *_, **__) -> bool:
    return instance.normalized_shape is None


def is_lazy_init(_, normalized_shape, *__, **___) -> bool:
    return normalized_shape is None


def infer_normalized_shape(instance, x, *_, **__) -> int:
    return x.shape


updates = dict(normalized_shape=infer_normalized_shape)


class LayerNorm(sk.TreeClass):
    """Layer Normalization

    Transform the input by scaling and shifting to have zero mean and unit variance.

    Args:
        normalized_shape: the shape of the input to be normalized.
        eps: a value added to the denominator for numerical stability.
        weight_init: a function to initialize the scale. Defaults to ones.
            if None, the scale is not trainable.
        bias_init: a function to initialize the shift. Defaults to zeros.
            if None, the shift is not trainable.
        key: a random key for initialization. Defaults to jax.random.PRNGKey(0).
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Note:
        :class:`.LayerNorm` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``normalized_shape`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((5,10))
        >>> _, layer = sk.nn.LayerNorm(None).at['__call__'](x)
        >>> layer(x).shape
        (5, 10)

    Reference:
        https://nn.labml.ai/normalization/layer_norm/index.html
    """

    eps: float = sk.field(callbacks=[Range(0, min_inclusive=False), ScalarLike()])

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        normalized_shape: int | tuple[int, ...] | None,
        *,
        eps: float = 1e-5,
        weight_init: InitType = "ones",
        bias_init: InitType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
        dtype: DType = jnp.float32,
    ):
        self.normalized_shape = (
            normalized_shape
            if isinstance(normalized_shape, tuple)
            else (normalized_shape,)
        )
        self.eps = eps
        self.weight_init = weight_init
        self.bias_init = bias_init

        self.gamma = resolve_init(weight_init)(key, self.normalized_shape, dtype)
        self.beta = resolve_init(bias_init)(key, self.normalized_shape, dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return layer_norm(
            x,
            gamma=self.gamma,
            beta=self.beta,
            eps=self.eps,
            normalized_shape=self.normalized_shape,
        )


def is_lazy_call(instance, *_, **__) -> bool:
    return instance.in_features is None


def is_lazy_init(_, in_features, *__, **___) -> bool:
    return in_features is None


def infer_in_features(instance, x, *_, **__) -> int:
    return x.shape[0]


updates = dict(in_features=infer_in_features)


class GroupNorm(sk.TreeClass):
    """Group Normalization

    Transform the input by scaling and shifting to have zero mean and unit variance.

    Args:
        in_features: the shape of the input to be normalized.
        groups: number of groups to separate the channels into.
        eps: a value added to the denominator for numerical stability.
        weight_init: a function to initialize the scale. Defaults to ones.
            if None, the scale is not trainable.
        bias_init: a function to initialize the shift. Defaults to zeros.
            if None, the shift is not trainable.
        key: a random key for initialization. Defaults to jax.random.PRNGKey(0).
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((5,10))
        >>> _, layer = sk.nn.GroupNorm(5, groups=1).at['__call__'](x)
        >>> layer(x).shape
        (5, 10)

    Note:
        :class:`.GroupNorm` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((5,10))
        >>> _, layer = sk.nn.GroupNorm(None, groups=5).at['__call__'](x)
        >>> layer(x).shape
        (5, 10)

    Reference:
        https://nn.labml.ai/normalization/group_norm/index.html
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features,
        *,
        groups: int,
        eps: float = 1e-5,
        weight_init: InitType = "ones",
        bias_init: InitType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
        dtype: DType = jnp.float32,
    ):
        self.in_features = positive_int_cb(in_features)
        self.groups = positive_int_cb(groups)
        self.eps = sk.field(callbacks=[Range(0), ScalarLike()])(eps)

        # needs more info for checking
        if in_features % groups != 0:
            raise ValueError(f"{in_features} must be divisible by {groups=}.")

        self.weight = resolve_init(weight_init)(key, (in_features,), dtype)
        self.bias = resolve_init(bias_init)(key, (in_features,), dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return group_norm(
            x=x,
            gamma=self.weight,
            beta=self.bias,
            eps=self.eps,
            groups=self.groups,
        )


class InstanceNorm(GroupNorm):
    """Instance Normalization

    Transform the input by scaling and shifting to have zero mean and unit variance.

    Args:
        in_features: the shape of the input to be normalized.
        eps: a value added to the denominator for numerical stability.
        weight_init: a function to initialize the scale. Defaults to ones.
            if None, the scale is not trainable.
        bias_init: a function to initialize the shift. Defaults to zeros.
            if None, the shift is not trainable.
        key: a random key for initialization. Defaults to jax.random.PRNGKey(0).
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((5,10))
        >>> _, layer = sk.nn.InstanceNorm(5).at['__call__'](x)
        >>> layer(x).shape
        (5, 10)

    Note:
        :class:`.InstanceNorm` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((5,10))
        >>> _, layer = sk.nn.InstanceNorm(None).at['__call__'](x)
        >>> layer(x).shape
        (5, 10)

    Reference:
        https://nn.labml.ai/normalization/instance_norm/index.html
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int,
        *,
        eps: float = 1e-5,
        weight_init: InitType = "ones",
        bias_init: InitType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
        dtype: DType = jnp.float32,
    ):
        super().__init__(
            in_features=in_features,
            groups=in_features,
            eps=eps,
            weight_init=weight_init,
            bias_init=bias_init,
            key=key,
            dtype=dtype,
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

    if gamma is not None:
        output *= jnp.reshape(gamma, broadcast_shape)

    if beta is not None:
        output += jnp.reshape(beta, broadcast_shape)

    output = jax.lax.cond(evalution, jax.lax.stop_gradient, lambda x: x, output)

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


def infer_in_features(instance, x, *_, **__) -> int:
    return x.shape[instance.axis]


updates = dict(in_features=infer_in_features)


class BatchNorm(sk.TreeClass):
    """Applies normalization over batched inputs`

    .. warning::
        Works under
            - ``jax.vmap(BatchNorm(...), in_axes=(0, None))(x, state)``
            - ``jax.vmap(BatchNorm(...))(x)``

        otherwise will be a no-op.

    Training behavior:
        - ``output = (x - batch_mean) / sqrt(batch_var + eps)``
        - ``running_mean = momentum * running_mean + (1 - momentum) * batch_mean``
        - ``running_var = momentum * running_var + (1 - momentum) * batch_var``

    For evaluation, use :func:`.tree_eval` to convert the layer to
    :class:`nn.EvalNorm`.


    Args:
        in_features: the shape of the input to be normalized.
        momentum: the value used for the ``running_mean`` and ``running_var``
            computation. must be a number between ``0`` and ``1``.
        eps: a value added to the denominator for numerical stability.
        weight_init: a function to initialize the scale. Defaults to ones.
            if None, the scale is not trainable.
        bias_init: a function to initialize the shift. Defaults to zeros.
            if None, the shift is not trainable.
        axis: the axis that should be normalized. Defaults to 1.
        key: a random key to initialize the parameters.
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import jax
        >>> import serket as sk
        >>> bn = sk.nn.BatchNorm(10)
        >>> state = sk.tree_state(bn)
        >>> x = jax.random.uniform(jax.random.PRNGKey(0), shape=(5, 10))
        >>> x, state = jax.vmap(bn, in_axes=(0, None))(x, state)

    Example:
        >>> # working with multiple states
        >>> import jax
        >>> import serket as sk
        >>> @sk.autoinit
        ... class Tree(sk.TreeClass):
        ...    bn1: sk.nn.BatchNorm = sk.nn.BatchNorm(10)
        ...    bn2: sk.nn.BatchNorm = sk.nn.BatchNorm(10)
        ...    def __call__(self, x, state):
        ...        x, bn1 = self.bn1(x, state.bn1)
        ...        x, bn2 = self.bn2(x, state.bn2)
        ...        # update the output state
        ...        state = state.at["bn1"].set(bn1).at["bn2"].set(bn2)
        ...        return x, state
        >>> tree = Tree()
        >>> # initialize state as the same structure as tree
        >>> state = sk.tree_state(tree)
        >>> x = jax.random.uniform(jax.random.PRNGKey(0), shape=(5, 10))
        >>> x, state = jax.vmap(tree, in_axes=(0, None))(x, state)

    Note:
        :class:`.BatchNorm` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((5,10))
        >>> _, layer = sk.nn.BatchNorm(None).at['__call__'](x)
        >>> x, state = jax.vmap(layer)(x)
        >>> x.shape
        (5, 10)

    Reference:
        https://keras.io/api/layers/normalization_layers/batch_normalization/
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int,
        *,
        momentum: float = 0.99,
        eps: float = 1e-5,
        weight_init: InitType = "ones",
        bias_init: InitType = "zeros",
        axis: int = 1,
        key: jr.KeyArray = jr.PRNGKey(0),
        dtype: DType = jnp.float32,
    ) -> None:
        self.in_features = in_features
        self.momentum = momentum
        self.eps = eps
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.axis = axis

        self.weight = resolve_init(weight_init)(key, (in_features,), dtype=dtype)
        self.bias = resolve_init(bias_init)(key, (in_features,), dtype=dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    def __call__(
        self,
        x: jax.Array,
        state: BatchNormState | None = None,
    ) -> tuple[jax.Array, BatchNormState]:
        state = sk.tree_state(self) if state is None else state

        x, state = batchnorm(
            x,
            state,
            self.momentum,
            self.eps,
            self.weight,
            self.bias,
            False,
            self.axis,
        )
        return x, state


class EvalNorm(sk.TreeClass):
    """Applies normalization evlaution step over batched inputs`

    This layer intended to be the evaluation step of :class:`nn.BatchNorm`.
    and to be used with ``jax.vmap``. It will be a no-op when unbatched.

    .. warning::
        Works under
            - ``jax.vmap(BatchNorm(...), in_axes=(0, None))(x, state)``
            - ``jax.vmap(BatchNorm(...))(x)``

        otherwise will be a no-op.

    Evaluation behavior:
        - ``output = (x - running_mean) / sqrt(running_var + eps)``

    Args:
        in_features: the shape of the input to be normalized.
        momentum: the value used for the ``running_mean`` and ``running_var``
            computation. must be a number between ``0`` and ``1``. this value
            is ignored in evaluation mode, but kept for conversion to
            :class:`nn.BatchNorm`.
        eps: a value added to the denominator for numerical stability.
        weight_init: a function to initialize the scale. Defaults to ones.
            if None, the scale is not trainable.
        bias_init: a function to initialize the shift. Defaults to zeros.
            if None, the shift is not trainable.
        axis: the axis that should be normalized. Defaults to 1.
        key: a random key to initialize the parameters.
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import jax
        >>> import serket as sk
        >>> bn = sk.nn.BatchNorm(10)
        >>> state = sk.tree_state(bn)
        >>> x = jax.random.uniform(jax.random.PRNGKey(0), shape=(5, 10))
        >>> x, state = jax.vmap(bn, in_axes=(0, None))(x, state)
        >>> # convert to evaluation mode
        >>> bn = sk.tree_eval(bn)
        >>> x, state = jax.vmap(bn, in_axes=(0, None))(x, state)

    Reference:
        https://keras.io/api/layers/normalization_layers/batch_normalization/
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int,
        *,
        momentum: float = 0.99,
        eps: float = 1e-5,
        weight_init: InitType = "ones",
        bias_init: InitType = "zeros",
        axis: int = 1,
        key: jr.KeyArray = jr.PRNGKey(0),
        dtype: DType = jnp.float32,
    ) -> None:
        self.in_features = in_features
        self.momentum = momentum
        self.eps = eps
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.axis = axis
        self.weight = resolve_init(weight_init)(key, (in_features,), dtype)
        self.bias = resolve_init(bias_init)(key, (in_features,), dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    def __call__(
        self,
        x: jax.Array,
        state: BatchNormState | None = None,
    ) -> tuple[jax.Array, BatchNormState]:
        state = sk.tree_state(self) if state is None else state

        x, state = batchnorm(
            x,
            state,
            0.0,
            self.eps,
            self.weight,
            self.bias,
            True,
            self.axis,
        )
        return x, state


@tree_eval.def_eval(BatchNorm)
def _(batchnorm: BatchNorm) -> EvalNorm:
    return EvalNorm(
        in_features=batchnorm.in_features,
        momentum=batchnorm.momentum,  # ignored
        eps=batchnorm.eps,
        weight_init=lambda *_: batchnorm.weight,
        bias_init=lambda *_: batchnorm.bias,
        axis=batchnorm.axis,
    )


@tree_state.def_state(BatchNorm)
def batchnorm_init_state(
    batchnorm: BatchNorm,
    array: jax.Array | None,
) -> BatchNormState:
    del array
    running_mean = jnp.zeros([batchnorm.in_features])
    running_var = jnp.ones([batchnorm.in_features])
    return BatchNormState(running_mean, running_var)
