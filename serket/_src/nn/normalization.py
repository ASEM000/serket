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

from __future__ import annotations

import functools as ft
from typing import TypeVar

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap

import serket as sk
from serket._src.custom_transform import tree_eval, tree_state
from serket._src.nn.initialization import DType, InitType, resolve_init
from serket._src.utils import (
    Range,
    ScalarLike,
    maybe_lazy_call,
    maybe_lazy_init,
    positive_int_cb,
)


def layer_norm(
    array: jax.Array,
    *,
    gamma: jax.Array,
    beta: jax.Array,
    eps: float,
    normalized_shape: int | tuple[int],
) -> jax.Array:
    dims = tuple(range(len(array.shape) - len(normalized_shape), len(array.shape)))

    μ = jnp.mean(array, axis=dims, keepdims=True)
    σ_2 = jnp.var(array, axis=dims, keepdims=True)
    x̂ = (array - μ) * jax.lax.rsqrt((σ_2 + eps))

    if gamma is not None:
        x̂ = x̂ * gamma

    if beta is not None:
        x̂ = x̂ + beta

    return x̂


def group_norm(
    array: jax.Array,
    *,
    gamma: jax.Array,
    beta: jax.Array,
    eps: float,
    groups: int,
) -> jax.Array:
    # split channels into groups
    xx = array.reshape(groups, -1)
    μ = jnp.mean(xx, axis=-1, keepdims=True)
    σ_2 = jnp.var(xx, axis=-1, keepdims=True)
    x̂ = (xx - μ) * jax.lax.rsqrt((σ_2 + eps))
    x̂ = x̂.reshape(*array.shape)

    if gamma is not None:
        gamma = jnp.expand_dims(gamma, axis=range(1, array.ndim))
        x̂ *= gamma

    if beta is not None:
        beta = jnp.expand_dims(beta, axis=range(1, array.ndim))
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

    .. image:: ../_static/norm_figure.png

    Transform the input by scaling and shifting to have zero mean and unit variance.

    Args:
        normalized_shape: the shape of the input to be normalized.
        key: a random key for initialization.
        eps: a value added to the denominator for numerical stability.
        weight_init: a function to initialize the scale. Defaults to ones.
            if None, the scale is not trainable.
        bias_init: a function to initialize the shift. Defaults to zeros.
            if None, the shift is not trainable.
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Note:
        :class:`.LayerNorm` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``normalized_shape`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((5,10))
        >>> _, layer = sk.nn.LayerNorm(None).at['__call__'](x)
        >>> layer(x).shape
        (5, 10)

    Reference:
        - https://nn.labml.ai/normalization/layer_norm/index.html
        - https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html
    """

    eps: float = sk.field(on_setattr=[Range(0, min_inclusive=False), ScalarLike()])

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        normalized_shape: int | tuple[int, ...] | None,
        *,
        key: jax.Array,
        eps: float = 1e-5,
        weight_init: InitType = "ones",
        bias_init: InitType = "zeros",
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
    def __call__(self, array: jax.Array) -> jax.Array:
        return layer_norm(
            array=array,
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

    .. image:: ../_static/norm_figure.png

    Transform the input by scaling and shifting to have zero mean and unit variance.

    Args:
        in_features: the shape of the input to be normalized.
        key: a random key for initialization.
        groups: number of groups to separate the channels into.
        eps: a value added to the denominator for numerical stability.
        weight_init: a function to initialize the scale. Defaults to ones.
            if None, the scale is not trainable.
        bias_init: a function to initialize the shift. Defaults to zeros.
            if None, the shift is not trainable.
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> x = jnp.ones((5,10))
        >>> _, layer = sk.nn.GroupNorm(5, groups=1, key=jr.PRNGKey(0)).at['__call__'](x)
        >>> layer(x).shape
        (5, 10)

    Note:
        :class:`.GroupNorm` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((5,10))
        >>> _, layer = sk.nn.GroupNorm(None, groups=5).at['__call__'](x)
        >>> layer(x).shape
        (5, 10)

    Reference:
        - https://nn.labml.ai/normalization/group_norm/index.html
        - https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html
    """

    eps: float = sk.field(on_setattr=[Range(0), ScalarLike()])

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features,
        *,
        key: jax.Array,
        groups: int,
        eps: float = 1e-5,
        weight_init: InitType = "ones",
        bias_init: InitType = "zeros",
        dtype: DType = jnp.float32,
    ):
        self.in_features = positive_int_cb(in_features)
        self.groups = positive_int_cb(groups)
        self.eps = eps

        # needs more info for checking
        if in_features % groups != 0:
            raise ValueError(f"{in_features} must be divisible by {groups=}.")

        self.weight = resolve_init(weight_init)(key, (in_features,), dtype)
        self.bias = resolve_init(bias_init)(key, (in_features,), dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    def __call__(self, array: jax.Array) -> jax.Array:
        return group_norm(
            array=array,
            gamma=self.weight,
            beta=self.bias,
            eps=self.eps,
            groups=self.groups,
        )


class InstanceNorm(GroupNorm):
    """Instance Normalization

    .. image:: ../_static/norm_figure.png

    Transform the input by scaling and shifting to have zero mean and unit variance.

    Args:
        in_features: the shape of the input to be normalized.
        key: a random key for initialization.
        eps: a value added to the denominator for numerical stability.
        weight_init: a function to initialize the scale. Defaults to ones.
            if None, the scale is not trainable.
        bias_init: a function to initialize the shift. Defaults to zeros.
            if None, the shift is not trainable.
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> x = jnp.ones((5,10))
        >>> _, layer = sk.nn.InstanceNorm(5, key=jr.PRNGKey(0)).at['__call__'](x)
        >>> layer(x).shape
        (5, 10)

    Note:
        :class:`.InstanceNorm` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((5,10))
        >>> _, layer = sk.nn.InstanceNorm(None).at['__call__'](x)
        >>> layer(x).shape
        (5, 10)

    Reference:
        - https://nn.labml.ai/normalization/instance_norm/index.html
        - https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int,
        *,
        key: jax.Array,
        eps: float = 1e-5,
        weight_init: InitType = "ones",
        bias_init: InitType = "zeros",
        dtype: DType = jnp.float32,
    ):
        super().__init__(
            in_features=in_features,
            key=key,
            groups=in_features,
            eps=eps,
            weight_init=weight_init,
            bias_init=bias_init,
            dtype=dtype,
        )


@sk.autoinit
class BatchNormState(sk.TreeClass):
    running_mean: jax.Array = sk.field(on_getattr=[jax.lax.stop_gradient_p.bind])
    running_var: jax.Array = sk.field(on_getattr=[jax.lax.stop_gradient_p.bind])


def batchnorm(
    array: jax.Array,
    state: BatchNormState,
    momentum: float = 0.1,
    eps: float = 1e-3,
    gamma: jax.Array = None,
    beta: jax.Array = None,
    axis: int = 1,
    axis_name: str | None = None,
):
    broadcast_shape = [1] * array.ndim
    broadcast_shape[axis] = array.shape[axis]
    axes = list(range(array.ndim))
    with jax.ensure_compile_time_eval():
        del axes[axis]
    batch_mean = jnp.mean(array, axis=axes, keepdims=True)
    if axis_name is not None:
        batch_mean = jax.lax.pmean(batch_mean, axis_name)
    batch_var = jnp.mean(jnp.square(array), axis=axes, keepdims=True)
    batch_var -= batch_mean**2
    if axis_name is not None:
        batch_var = jax.lax.pmean(batch_var, axis_name)
    output = (array - batch_mean) / jnp.sqrt(batch_var + eps)
    run_mean = momentum * state.running_mean + (1 - momentum) * jnp.squeeze(batch_mean)
    run_var = momentum * state.running_var + (1 - momentum) * jnp.squeeze(batch_var)

    state = BatchNormState(run_mean, run_var)
    if gamma is not None:
        output *= jnp.reshape(gamma, broadcast_shape)
    if beta is not None:
        output += jnp.reshape(beta, broadcast_shape)
    return output, state


def evalnorm(
    array: jax.Array,
    state: BatchNormState,
    momentum: float = 0.1,
    eps: float = 1e-3,
    gamma: jax.Array = None,
    beta: jax.Array = None,
    axis: int = 1,
    axis_name: str | None = None,
):
    broadcast_shape = [1] * array.ndim
    broadcast_shape[axis] = array.shape[axis]
    run_mean, run_var = state.running_mean, state.running_var
    run_mean = jnp.reshape(run_mean, broadcast_shape)
    run_var = jnp.reshape(run_var, broadcast_shape)
    output = (array - run_mean) * jax.lax.rsqrt(run_var + eps)

    if gamma is not None:
        output *= jnp.reshape(gamma, broadcast_shape)
    if beta is not None:
        output += jnp.reshape(beta, broadcast_shape)
    return output, state


def infer_in_features(instance, x, *_, **__) -> int:
    return x.shape[instance.axis]


updates = dict(in_features=infer_in_features)


class BatchNorm(sk.TreeClass):
    """Applies normalization over batched inputs`

    .. image:: ../_static/norm_figure.png

    .. warning::
        Works under
            - ``jax.vmap(BatchNorm(...), in_axes=(0, None), out_axes=(0, None))(x, state)``
            - ``jax.vmap(BatchNorm(...), out_axes=(0, None))(x)``

        otherwise will be a no-op.

    Training behavior:
        - ``output = (x - batch_mean) / sqrt(batch_var + eps)``
        - ``running_mean = momentum * running_mean + (1 - momentum) * batch_mean``
        - ``running_var = momentum * running_var + (1 - momentum) * batch_var``

    For evaluation, use :func:`.tree_eval` to convert the layer to
    :class:`nn.EvalNorm`.


    Args:
        in_features: the shape of the input to be normalized.
        key: a random key to initialize the parameters.
        momentum: the value used for the ``running_mean`` and ``running_var``
            computation. must be a number between ``0`` and ``1``.
        eps: a value added to the denominator for numerical stability.
        weight_init: a function to initialize the scale. Defaults to ones.
            if None, the scale is not trainable.
        bias_init: a function to initialize the shift. Defaults to zeros.
            if None, the shift is not trainable.
        axis: the feature axis that should be normalized. Defaults to 1. i.e.
            the other axes are reduced over.
        axis_name: the axis name passed to ``jax.lax.pmean``. Defaults to None.
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import jax
        >>> import serket as sk
        >>> import jax.random as jr
        >>> bn = sk.nn.BatchNorm(10, key=jr.PRNGKey(0))
        >>> state = sk.tree_state(bn)
        >>> x = jax.random.uniform(jax.random.PRNGKey(0), shape=(5, 10))
        >>> x, state = jax.vmap(bn, in_axes=(0, None))(x, state)

    Example:
        Working with :class:`.BatchNorm` with threading the state.

        >>> import jax
        >>> import serket as sk
        >>> import jax.random as jr
        >>> import jax.numpy as jnp
        >>> class ThreadedBatchNorm(sk.TreeClass):
        ...    def __init__(self, *, key: jax.Array):
        ...        k1, k2 = jax.random.split(key)
        ...        self.bn1 = sk.nn.BatchNorm(5, axis=-1, key=k1)
        ...        self.bn2 = sk.nn.BatchNorm(5, axis=-1, key=k2)
        ...    def __call__(self, x, state):
        ...        x, bn1 = self.bn1(x, state.bn1)
        ...        x = x + 1.0
        ...        x, bn2 = self.bn2(x, state.bn2)
        ...        # update the output state
        ...        state = state.at["bn1"].set(bn1).at["bn2"].set(bn2)
        ...        return x, state
        >>> net: ThreadedBatchNorm = ThreadedBatchNorm(key=jr.PRNGKey(0))
        >>> # initialize state as the same structure as tree
        >>> state: ThreadedBatchNorm = sk.tree_state(net)
        >>> x = jnp.linspace(-jnp.pi, jnp.pi, 50 * 20).reshape(20, 10, 5)
        >>> for xi in x:
        ...    out, state = jax.vmap(net, in_axes=(0, None), out_axes=(0, None))(xi, state)
        >>> print(state)
        ThreadedBatchNorm(
          bn1=BatchNormState(
            running_mean=[0.01683246 0.01797775 0.01912302 0.02026827 0.02141353],
            running_var=[0.819393  0.8193929 0.819393  0.819393  0.819393 ]
          ),
          bn2=BatchNormState(
            running_mean=[0.18209304 0.18209311 0.18209308 0.18209308 0.18209304],
            running_var=[0.9997767  0.99978393 0.99978125 0.99977994 0.99978155]
          )
        )

    Example:
        Working with :class:`.BatchNorm` without threading the state.

        Instead of threading a state in and out of the layer's ``__call__`` method as
        previously shown, this example demonstrates a state that is integrated
        within the layer, akin to the approaches used in Keras or PyTorch. However,
        since the state is embedded in the layer, some additional work is required
        to make sure the layer works within the functional paradigm by using the
        ``at`` functionality, which is illustrated in the example below.

        >>> import jax
        >>> import serket as sk
        >>> import jax.random as jr
        >>> import functools as ft
        >>> class UnthreadedBatchNorm(sk.TreeClass):
        ...    def __init__(self, *, key: jax.Array):
        ...        k1, k2 = jax.random.split(key)
        ...        self.bn1 = sk.nn.BatchNorm(5, axis=-1, key=k1)
        ...        self.bn1_state = sk.tree_state(self.bn1)
        ...        self.bn2 = sk.nn.BatchNorm(5, axis=-1, key=k2)
        ...        self.bn2_state = sk.tree_state(self.bn2)
        ...    def _call(self, x):
        ...        # this method will raise `AttributeError` if used directly
        ...        # because this method mutates the state
        ...        # instead, use `at["_call"]` to call this method to
        ...        # return the output and updated state in a functional manner
        ...        x, self.bn1_state = self.bn1(x, self.bn1_state)
        ...        x = x + 1.0
        ...        x, self.bn2_state = self.bn2(x, self.bn2_state)
        ...        return x
        ...    def __call__(self, x):
        ...        return self.at["_call"](x)
        >>> # define a function to mask and unmask the net across `vmap`
        >>> # this is necessary because `vmap` needs the output to be of inexact
        >>> def mask_vmap(net, x):
        ...    @ft.partial(jax.vmap, out_axes=(0, None))
        ...    def forward(x):
        ...        return sk.tree_mask(net(x))
        ...    return sk.tree_unmask(forward(x))
        >>> net: UnthreadedBatchNorm = UnthreadedBatchNorm(key=jr.PRNGKey(0))
        >>> x = jnp.linspace(-jnp.pi, jnp.pi, 50 * 20).reshape(20, 10, 5)
        >>> for xi in x:
        ...    out, net = mask_vmap(net, xi)
        >>> print(net.bn1_state)
        BatchNormState(
          running_mean=[0.01683246 0.01797775 0.01912302 0.02026827 0.02141353],
          running_var=[0.819393  0.8193929 0.819393  0.819393  0.819393 ]
        )
        >>> print(net.bn2_state)
        BatchNormState(
          running_mean=[0.18209304 0.18209311 0.18209308 0.18209308 0.18209304],
          running_var=[0.9997767  0.99978393 0.99978125 0.99977994 0.99978155]
        )

    Note:
        :class:`.BatchNorm` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["__call__"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> x = jnp.ones((5,10))
        >>> _, layer = sk.nn.BatchNorm(None, key=jr.PRNGKey(0)).at['__call__'](x)
        >>> x, state = jax.vmap(layer)(x)
        >>> x.shape
        (5, 10)

    Note:
        If ``axis_name`` is specified, then ``axis_name`` argument must be passed
        to ``jax.vmap`` or ``jax.pmap``.

    Reference:
        - https://keras.io/api/layers/normalization_layers/batch_normalization/
        - https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int,
        *,
        key: jax.Array,
        momentum: float = 0.99,
        eps: float = 1e-5,
        weight_init: InitType = "ones",
        bias_init: InitType = "zeros",
        axis: int = 1,
        axis_name: str | None = None,
        dtype: DType = jnp.float32,
    ) -> None:
        self.in_features = in_features
        self.momentum = momentum
        self.eps = eps
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.axis = axis
        self.axis_name = axis_name
        self.weight = resolve_init(weight_init)(key, (in_features,), dtype=dtype)
        self.bias = resolve_init(bias_init)(key, (in_features,), dtype=dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    def __call__(
        self,
        array: jax.Array,
        state: BatchNormState | None = None,
    ) -> tuple[jax.Array, BatchNormState]:
        state = sk.tree_state(self) if state is None else state
        batchnorm_impl = custom_vmap(lambda x, state: (x, state))
        momentum, eps = jax.lax.stop_gradient((self.momentum, self.eps))

        @batchnorm_impl.def_vmap
        def _(_, batch_tree, array: jax.Array, state: BatchNormState):
            output = batchnorm(
                array=array,
                state=state,
                momentum=momentum,
                eps=eps,
                gamma=self.weight,
                beta=self.bias,
                axis=self.axis,
                axis_name=self.axis_name,
            )
            return output, tuple(batch_tree)

        return batchnorm_impl(array, state)


class EvalNorm(sk.TreeClass):
    """Applies normalization evlaution step over batched inputs`

    This layer intended to be the evaluation step of :class:`nn.BatchNorm`.
    and to be used with ``jax.vmap``. It will be a no-op when unbatched.

    .. warning::
        Works under
            - ``jax.vmap(BatchNorm(...), in_axes=(0, None), out_axes=(0, None))(x, state)``
            - ``jax.vmap(BatchNorm(...), out_axes=(0, None))(x)``

        otherwise will be a no-op.

    Evaluation behavior:
        - ``output = (x - running_mean) / sqrt(running_var + eps)``

    Args:
        in_features: the shape of the input to be normalized.
        key: a random key to initialize the parameters.
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
        axis_name: the axis name passed to ``jax.lax.pmean``. Defaults to None.
        dtype: dtype of the weights and biases. defaults to ``jnp.float32``.

    Example:
        >>> import jax
        >>> import serket as sk
        >>> import jax.random as jr
        >>> bn = sk.nn.BatchNorm(10, key=jr.PRNGKey(0))
        >>> state = sk.tree_state(bn)
        >>> x = jax.random.uniform(jax.random.PRNGKey(0), shape=(5, 10))
        >>> x, state = jax.vmap(bn, in_axes=(0, None))(x, state)
        >>> # convert to evaluation mode
        >>> bn = sk.tree_eval(bn)
        >>> x, state = jax.vmap(bn, in_axes=(0, None))(x, state)

    Note:
        If ``axis_name`` is specified, then ``axis_name`` argument must be passed
        to ``jax.vmap`` or ``jax.pmap``.

    Reference:
        https://keras.io/api/layers/normalization_layers/batch_normalization/
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int,
        *,
        key: jax.Array,
        momentum: float = 0.99,
        eps: float = 1e-5,
        weight_init: InitType = "ones",
        bias_init: InitType = "zeros",
        axis: int = 1,
        axis_name: str | None = None,
        dtype: DType = jnp.float32,
    ) -> None:
        self.in_features = in_features
        self.momentum = momentum
        self.eps = eps
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.axis = axis
        self.axis_name = axis_name
        self.weight = resolve_init(weight_init)(key, (in_features,), dtype)
        self.bias = resolve_init(bias_init)(key, (in_features,), dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    def __call__(
        self,
        x: jax.Array,
        state: BatchNormState | None = None,
    ) -> tuple[jax.Array, BatchNormState]:
        state = sk.tree_state(self) if state is None else state
        evalnorm_impl = custom_vmap(lambda x, state: (x, state))
        eps = jax.lax.stop_gradient(self.eps)

        @evalnorm_impl.def_vmap
        def _(_, batch_tree, array: jax.Array, state: BatchNormState):
            output = evalnorm(
                array=array,
                state=state,
                momentum=0.0,
                eps=eps,
                gamma=self.weight,
                beta=self.bias,
                axis=self.axis,
                axis_name=self.axis_name,
            )
            return output, tuple(batch_tree)

        return evalnorm_impl(x, state)


@tree_eval.def_eval(BatchNorm)
def _(batchnorm: BatchNorm) -> EvalNorm:
    return EvalNorm(
        in_features=batchnorm.in_features,
        momentum=batchnorm.momentum,  # ignored
        eps=batchnorm.eps,
        weight_init=lambda *_: batchnorm.weight,
        bias_init=None if batchnorm.bias is None else lambda *_: batchnorm.bias,
        axis=batchnorm.axis,
        axis_name=batchnorm.axis_name,
        key=None,
    )


@tree_state.def_state(BatchNorm)
def _(batchnorm: BatchNorm, **_) -> BatchNormState:
    running_mean = jnp.zeros([batchnorm.in_features])
    running_var = jnp.ones([batchnorm.in_features])
    return BatchNormState(running_mean, running_var)


T = TypeVar("T")


def weight_norm(
    leaf: T,
    axis: int | None = -1,
    eps: float = 1e-12,
) -> T:
    """Apply L2 weight normalization to an array.

    Args:
        leaf: the array to be normalized. If ``leaf`` is not an array, then it will
            be returned as is.
        axis: the feature axis to be normalized. defaults to -1.
        eps: the epsilon value to be added to the denominator. defaults to 1e-12.

    Example:
        Normalize ``weight`` arrays of two-layer linear network but not ``bias``

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> class Net(sk.TreeClass):
        ...     def __init__(self, *, key: jax.Array):
        ...         k1, k2 = jax.random.split(key)
        ...         self.l1 = sk.nn.Linear(2, 4, key=k1)
        ...         self.l2 = sk.nn.Linear(4, 2, key=k2)
        >>>     def __call__(self, inputs: jax.Array) -> jax.Array:
        ...         # `...` selects all the first level nodes of `Net` (e.g. `l1`, `l2`)
        ...         # then the `weight` attribute of each layer at the second level
        ...         self = self.at[...]["weight"].apply(sk.nn.weight_norm)
        ...         return self.l2(self.l1(inputs))

    Reference:
        - https://arxiv.org/pdf/1602.07868.pdf
    """
    if not (hasattr(leaf, "ndim") and hasattr(leaf, "shape")):
        return leaf
    if axis is not None:
        reduction_axes = list(range(leaf.ndim))
        with jax.ensure_compile_time_eval():
            del reduction_axes[axis]
    ssum = jnp.sum(jnp.square(leaf), axis=reduction_axes, keepdims=True)
    return leaf * jax.lax.rsqrt(ssum + eps)
