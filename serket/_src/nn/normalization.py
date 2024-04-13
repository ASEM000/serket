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

import functools as ft
from typing import Sequence

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap

from serket import TreeClass, autoinit, field
from serket._src.custom_transform import tree_eval, tree_state
from serket._src.nn.initialization import resolve_init
from serket._src.utils.convert import tuplify
from serket._src.utils.lazy import maybe_lazy_call, maybe_lazy_init
from serket._src.utils.typing import DType, InitType, T
from serket._src.utils.validate import (
    Range,
    ScalarLike,
    validate_in_features_shape,
    validate_pos_int,
)


def layer_norm(
    input: jax.Array,
    weight: jax.Array | None,
    bias: jax.Array | None,
    normalized_shape: Sequence[int],
    eps: float | None = None,
) -> jax.Array:
    """Layer Normalization

    Normalizes the input by scaling and shifting to have zero mean and unit variance
    over the last ``len(normalized_shape)`` dimensions.

    Args:
        input: the input to be normalized.
        weight: the scale. if None, the scale is not trainable.
        bias: the shift. if None, the shift is not trainable.
        normalized_shape: the shape of the input to be normalized. Accepts a tuple
            of integers to denote the shape of the input to be normalized.
            The last ``len(normalized_shape)`` dimensions will be normalized.
        eps: a value added to the denominator for numerical stability. defaults to
            ``jnp.finfo(input.dtype).eps`` if ``None`` is passed.
    """
    assert normalized_shape == input.shape[-len(normalized_shape) :]
    axes = tuple(range(len(input.shape) - len(normalized_shape), len(input.shape)))
    eps = jnp.finfo(input.dtype).eps if eps is None else eps
    μ = jnp.mean(input, axis=axes, keepdims=True)
    σ_2 = jnp.var(input, axis=axes, keepdims=True)
    output = (input - μ) * jax.lax.rsqrt((σ_2 + eps))
    output = output if weight is None else output * weight
    output = output if bias is None else output + bias
    return output


def group_norm(
    input: jax.Array,
    weight: jax.Array | None,
    bias: jax.Array | None,
    eps: float,
    groups: int,
) -> jax.Array:
    """Group Normalization

    Args:
        input: the input to be normalized.
        weight: the scale. if None, the scale is not trainable.
        bias: the shift. if None, the shift is not trainable.
        eps: a value added to the denominator for numerical stability.
        groups: number of groups to separate the features into.
    """
    # split channels into groups
    shape = input.shape
    axes = tuple(range(1, input.ndim))
    input = input.reshape(groups, -1)
    # calculate mean and variance over each group
    μ = jnp.mean(input, axis=-1, keepdims=True)
    σ_2 = jnp.var(input, axis=-1, keepdims=True)
    output = (input - μ) * jax.lax.rsqrt((σ_2 + eps))
    output = output.reshape(*shape)
    output = output if weight is None else output * jnp.expand_dims(weight, axis=axes)
    output = output if bias is None else output + jnp.expand_dims(bias, axis=axes)
    return output


def instance_norm(
    input: jax.Array,
    weight: jax.Array | None,
    bias: jax.Array | None,
    eps: float,
) -> jax.Array:
    """Instance Normalization

    Args:
        input: the input to be normalized.
        weight: the scale. if None, the scale is not trainable.
        bias: the shift. if None, the shift is not trainable.
        eps: a value added to the denominator for numerical stability.
    """
    return group_norm(input, weight, bias, eps, groups=input.shape[0])


def is_lazy_call(instance, *_1, **_2) -> bool:
    return instance.normalized_shape is None


def is_lazy_init(_1, normalized_shape, *_2, **_3) -> bool:
    return normalized_shape is None


def infer_normalized_shape(_1, input, *_2, **_3) -> int:
    return input.shape


updates = dict(normalized_shape=infer_normalized_shape)


class LayerNorm(TreeClass):
    """Layer Normalization

    .. image:: ../_static/norm_figure.png

    Transform the input by scaling and shifting to have zero mean and unit variance.

    Args:
        normalized_shape: the shape of the input to be normalized.
        key: a random key for initialization of the scale and shift.
        weight_init: a function to initialize the scale. Defaults to ones.
            if ``None``, the scale is not trainable.
        bias_init: a function to initialize the shift. Defaults to zeros.
            if ``None``, the shift is not trainable.
        dtype: dtype of the weights and biases. ``float32``
        eps: a value added to the denominator for numerical stability.

    Example:
        >>> import serket as sk
        >>> import jax.random as jr
        >>> import jax.numpy as jnp
        >>> import numpy.testing as npt
        >>> C, H, W = 4, 5, 6
        >>> k1, k2 = jr.split(jr.PRNGKey(0), 2)
        >>> input = jr.uniform(k1, shape=(C, H, W))
        >>> layer = sk.nn.LayerNorm((H, W), key=k2)
        >>> output = layer(input)
        >>> mean = jnp.mean(input, axis=(1, 2), keepdims=True)
        >>> var = jnp.var(input, axis=(1, 2), keepdims=True)
        >>> npt.assert_allclose((input - mean) / jnp.sqrt(var + 1e-5), output, atol=1e-5)

    Note:
        :class:`.LayerNorm` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``normalized_shape`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> input = jnp.ones((5,10))
        >>> key = jr.PRNGKey(0)
        >>> lazy = sk.nn.LayerNorm(None, key=key)
        >>> _, material = sk.value_and_tree(lambda lazy: lazy(input))(lazy)
        >>> material(input).shape
        (5, 10)

    Reference:
        - https://nn.labml.ai/normalization/layer_norm/index.html
        - https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html
    """

    eps: float = field(on_setattr=[Range(0, min_inclusive=False), ScalarLike()])

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        normalized_shape: int | tuple[int, ...] | None,
        *,
        key: jax.Array,
        weight_init: InitType = "ones",
        bias_init: InitType = "zeros",
        dtype: DType = jnp.float32,
        eps: float = 1e-5,
    ):
        self.normalized_shape = tuplify(normalized_shape)
        self.eps = eps
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.weight = resolve_init(weight_init)(key, self.normalized_shape, dtype)
        self.bias = resolve_init(bias_init)(key, self.normalized_shape, dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    def __call__(self, input: jax.Array) -> jax.Array:
        return layer_norm(
            input=input,
            weight=self.weight,
            bias=self.bias,
            normalized_shape=self.normalized_shape,
            eps=self.eps,
        )


def is_lazy_call(instance, *_1, **_2) -> bool:
    return instance.in_features is None


def is_lazy_init(_, in_features, *_1, **_2) -> bool:
    return in_features is None


def infer_in_features(instance, x, *_1, **_2) -> int:
    return x.shape[0]


updates = dict(in_features=infer_in_features)


class GroupNorm(TreeClass):
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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> key = jr.PRNGKey(0)
        >>> layer = sk.nn.GroupNorm(5, groups=1, key=key)
        >>> input = jnp.ones((5,10))
        >>> layer(input).shape
        (5, 10)

    Note:
        :class:`.GroupNorm` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> key = jr.PRNGKey(0)
        >>> lazy = sk.nn.GroupNorm(None, groups=5, key=key)
        >>> input = jnp.ones((5,10))
        >>> _, material = sk.value_and_tree(lambda lazy: lazy(input))(lazy)
        >>> material(input).shape
        (5, 10)

    Reference:
        - https://nn.labml.ai/normalization/group_norm/index.html
        - https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html
    """

    eps: float = field(on_setattr=[Range(0), ScalarLike()])

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
        self.in_features = validate_pos_int(in_features)
        self.groups = validate_pos_int(groups)
        self.eps = eps

        # needs more info for checking
        if in_features % groups != 0:
            raise ValueError(f"{in_features=} must be divisible by {groups=}.")

        self.weight = resolve_init(weight_init)(key, (in_features,), dtype)
        self.bias = resolve_init(bias_init)(key, (in_features,), dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_in_features_shape, axis=0)
    def __call__(self, input: jax.Array) -> jax.Array:
        return group_norm(
            input=input,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
            groups=self.groups,
        )


class InstanceNorm(TreeClass):
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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> key = jr.PRNGKey(0)
        >>> layer = sk.nn.InstanceNorm(5, key=key)
        >>> input = jnp.ones((5,10))
        >>> layer(input).shape
        (5, 10)

    Note:
        :class:`.InstanceNorm` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> key = jr.PRNGKey(0)
        >>> lazy = sk.nn.InstanceNorm(None, key=key)
        >>> input = jnp.ones((5,10))
        >>> _, material = sk.value_and_tree(lambda lazy: lazy(input))(lazy)
        >>> material(input).shape
        (5, 10)

    Reference:
        - https://nn.labml.ai/normalization/instance_norm/index.html
        - https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html
    """

    eps: float = field(on_setattr=[Range(0), ScalarLike()])

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features,
        *,
        key: jax.Array,
        eps: float = 1e-5,
        weight_init: InitType = "ones",
        bias_init: InitType = "zeros",
        dtype: DType = jnp.float32,
    ):
        self.in_features = validate_pos_int(in_features)
        self.eps = eps
        self.weight = resolve_init(weight_init)(key, (in_features,), dtype)
        self.bias = resolve_init(bias_init)(key, (in_features,), dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
    @ft.partial(validate_in_features_shape, axis=0)
    def __call__(self, input: jax.Array) -> jax.Array:
        return instance_norm(
            input=input,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )


# batch norm is possibly the most complicated layer in serket
# this is because it combines a state and a custom vmap behavior
# for state, per serket design, two layers are needed: one for always training
# and one for evaluation. The other aspect is the custom vmap behavior which
# is needed to have access to the batch dimension for the running mean and var
# computation.


@autoinit
class BatchNormState(TreeClass):
    # use primitive `bind` on jax array instead of `stop_gradient` to avoid `tree_map`
    running_mean: jax.Array = field(on_getattr=[jax.lax.stop_gradient_p.bind])
    running_var: jax.Array = field(on_getattr=[jax.lax.stop_gradient_p.bind])


def batch_norm(
    input: jax.Array,
    running_mean: jax.Array,
    running_var: jax.Array,
    momentum: float = 0.1,
    eps: float = 1e-3,
    weight: jax.Array = None,
    bias: jax.Array = None,
    axis: int = 1,
    axis_name: str | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Batch Normalization

    Computes:
        - ``output = (input - batch_mean) / sqrt(batch_var + eps)``
        - ``running_mean = momentum * running_mean + (1 - momentum) * batch_mean``
        - ``running_var = momentum * running_var + (1 - momentum) * batch_var``

    Args:
        input: the input to be normalized.
        running_mean: the running mean of the input.
        running_var: the running variance of the input.
        momentum: the value used for the ``running_mean`` and ``running_var``
            computation. must be a number between ``0`` and ``1``.
        eps: a value added to the denominator for numerical stability.
        weight: the scale. if None, the scale is not trainable.
        bias: the shift. if None, the shift is not trainable.
        axis: the axis that should be normalized. Defaults to 1.
        axis_name: the axis name passed to ``jax.lax.pmean``. Defaults to None.

    Returns:
        - ``normalized_input``
        - ``running_mean``
        - ``running_var``
    """
    broadcast_shape = [1] * input.ndim
    broadcast_shape[axis] = input.shape[axis]
    axes = list(range(input.ndim))
    with jax.ensure_compile_time_eval():
        del axes[axis]
    batch_mean = jnp.mean(input, axis=axes, keepdims=True)
    if axis_name is not None:
        batch_mean = jax.lax.pmean(batch_mean, axis_name)
    batch_var = jnp.mean(jnp.square(input), axis=axes, keepdims=True)
    batch_var -= batch_mean**2
    if axis_name is not None:
        batch_var = jax.lax.pmean(batch_var, axis_name)
    output = (input - batch_mean) / jnp.sqrt(batch_var + eps)
    running_mean = momentum * running_mean + (1 - momentum) * jnp.squeeze(batch_mean)
    running_var = momentum * running_var + (1 - momentum) * jnp.squeeze(batch_var)
    output = output if weight is None else output * jnp.reshape(weight, broadcast_shape)
    output = output if bias is None else output + jnp.reshape(bias, broadcast_shape)
    return output, running_mean, running_var


def eval_batch_norm(
    input: jax.Array,
    running_mean: jax.Array,
    running_var: jax.Array,
    momentum: float = 0.1,
    eps: float = 1e-3,
    weight: jax.Array = None,
    bias: jax.Array = None,
    axis: int = 1,
    axis_name: str | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Batch Normalization Evaluation Step

    Computes:
        - ``output = (input - running_mean) / sqrt(running_var + eps)``

    Args:
        input: the input to be normalized.
        running_mean: the running mean of the input.
        running_var: the running variance of the input.
        momentum: the value used for the ``running_mean`` and ``running_var``
            computation. must be a number between ``0`` and ``1``.
        eps: a value added to the denominator for numerical stability.
        weight: the scale. if None, the scale is not trainable.
        bias: the shift. if None, the shift is not trainable.
        axis: the axis that should be normalized. Defaults to 1.
        axis_name: the axis name passed to ``jax.lax.pmean``. Defaults to None.

    Returns:
        - ``normalized_input``
        - ``running_mean``
        - ``running_var``
    """
    del momentum, axis_name
    broadcast_shape = [1] * input.ndim
    broadcast_shape[axis] = input.shape[axis]
    running_mean = jnp.reshape(running_mean, broadcast_shape)
    running_var = jnp.reshape(running_var, broadcast_shape)
    output = (input - running_mean) * jax.lax.rsqrt(running_var + eps)
    output = output if weight is None else output * jnp.reshape(weight, broadcast_shape)
    output = output if bias is None else output + jnp.reshape(bias, broadcast_shape)
    return output, running_mean, running_var


def infer_in_features(instance, input, *_, **__) -> int:
    return input.shape[instance.axis]


updates = dict(in_features=infer_in_features)


class BatchNorm(TreeClass):
    """Applies normalization over batched inputs`

    .. image:: ../_static/norm_figure.png

    .. warning::
        Works under
            - ``jax.vmap(BatchNorm(...), in_axes=(0, None), out_axes=(0, None))(input, state)``

        otherwise will be a no-op.

    Training behavior:
        - ``output = (input - batch_mean) / sqrt(batch_var + eps)``
        - ``running_mean = momentum * running_mean + (1 - momentum) * batch_mean``
        - ``running_var = momentum * running_var + (1 - momentum) * batch_var``

    For evaluation, use :func:`.tree_eval` to convert the layer to :class:`.EvalBatchNorm`.

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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import jax
        >>> import serket as sk
        >>> import jax.random as jr
        >>> bn = sk.nn.BatchNorm(10, key=jr.PRNGKey(0))
        >>> state = sk.tree_state(bn)
        >>> key = jr.PRNGKey(0)
        >>> input = jr.uniform(key, shape=(5, 10))
        >>> output, state = jax.vmap(bn, in_axes=(0, None), out_axes=(0, None))(input, state)

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
        ...    def __call__(self, input, state):
        ...        input, bn1 = self.bn1(input, state.bn1)
        ...        input = input + 1.0
        ...        input, bn2 = self.bn2(input, state.bn2)
        ...        # update the output state
        ...        state = state.at["bn1"].set(bn1).at["bn2"].set(bn2)
        ...        return input, state
        >>> net: ThreadedBatchNorm = ThreadedBatchNorm(key=jr.PRNGKey(0))
        >>> # initialize state as the same structure as tree
        >>> state: ThreadedBatchNorm = sk.tree_state(net)
        >>> inputs = jnp.linspace(-jnp.pi, jnp.pi, 50 * 20).reshape(20, 10, 5)
        >>> for input in inputs:
        ...     output, state = jax.vmap(net, in_axes=(0, None), out_axes=(0, None))(input, state)

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
        ...    def __call__(self, input):
        ...        # this method will raise `AttributeError` if used directly
        ...        # because this method mutates the state
        ...        # instead, use `value_and_tree` to call the layer and return
        ...        # the output and updated state in a functional manner
        ...        input, self.bn1_state = self.bn1(input, self.bn1_state)
        ...        input = input + 1.0
        ...        input, self.bn2_state = self.bn2(input, self.bn2_state)
        ...        return input
        >>> # define a function to mask and unmask the net across `vmap`
        >>> # this is necessary because `vmap` needs the output to be of `jaxtype`
        >>> def mask_vmap(net, input):
        ...    @ft.partial(jax.vmap, out_axes=(0, None))
        ...    def forward(input):
        ...        output, new_net = sk.value_and_tree(lambda net: net(input))(net)
        ...        return output, sk.tree_mask(new_net)
        ...    return sk.tree_unmask(forward(input))
        >>> net: UnthreadedBatchNorm = UnthreadedBatchNorm(key=jr.PRNGKey(0))
        >>> inputs = jnp.linspace(-jnp.pi, jnp.pi, 50 * 20).reshape(20, 10, 5)
        >>> for input in inputs:
        ...    output, net = mask_vmap(net, input)

    Note:
        :class:`.BatchNorm` supports lazy initialization, meaning that the
        weights and biases are not initialized until the first call to the layer.
        This is useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use :func:`.value_and_tree` to call the layer and return the method
        output and the material layer.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> key = jr.PRNGKey(0)
        >>> lazy = sk.nn.BatchNorm(None, key=key)
        >>> input = jnp.ones((5,10))
        >>> _ , material = sk.value_and_tree(lambda lazy: lazy(input, None))(lazy)
        >>> state = sk.tree_state(material)
        >>> output, state = jax.vmap(material, in_axes=(0, None), out_axes=(0, None))(input, state)
        >>> output.shape
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
        input: jax.Array,
        state: BatchNormState,
    ) -> tuple[jax.Array, BatchNormState]:
        # simplify the call signature to avoid forcing the user to
        # to use a long `in_axes` and `out_axes` argument
        batch_norm_impl = custom_vmap(lambda input, state: (input, state))
        momentum, eps = jax.lax.stop_gradient((self.momentum, self.eps))

        @batch_norm_impl.def_vmap
        def _(_, batch_tree, input: jax.Array, state: BatchNormState):
            output, running_mean, running_var = batch_norm(
                input=input,
                running_mean=state.running_mean,
                running_var=state.running_var,
                momentum=momentum,
                eps=eps,
                weight=self.weight,
                bias=self.bias,
                axis=self.axis,
                axis_name=self.axis_name,
            )

            state = BatchNormState(running_mean, running_var)
            return (output, state), tuple(batch_tree)

        return batch_norm_impl(input, state)


class EvalBatchNorm(TreeClass):
    """Applies normalization evlaution step over batched inputs`

    This layer intended to be the evaluation step of :class:`.BatchNorm`.
    and to be used with ``jax.vmap``. It will be a no-op when unbatched.

    .. warning::
        Works under
            - ``jax.vmap(BatchNorm(...), in_axes=(0, None), out_axes=(0, None))(input, state)``

        otherwise will be a no-op.

    Evaluation behavior:
        - ``output = (input - running_mean) / sqrt(running_var + eps)``

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
        dtype: dtype of the weights and biases. ``float32``

    Example:
        >>> import jax
        >>> import serket as sk
        >>> import jax.random as jr
        >>> bn = sk.nn.BatchNorm(10, key=jr.PRNGKey(0))
        >>> state = sk.tree_state(bn)
        >>> input = jax.random.uniform(jr.PRNGKey(0), shape=(5, 10))
        >>> output, state = jax.vmap(bn, in_axes=(0, None), out_axes=(0, None))(input, state)
        >>> # convert to evaluation mode
        >>> bn = sk.tree_eval(bn)
        >>> output, state = jax.vmap(bn, in_axes=(0, None), out_axes=(0,None))(input, state)

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
        input: jax.Array,
        state: BatchNormState,
    ) -> tuple[jax.Array, BatchNormState]:
        # simplify the call signature to avoid forcing the user to
        # to use a long `in_axes` and `out_axes` argument
        eval_norm_impl = custom_vmap(lambda x, state: (x, state))
        eps = jax.lax.stop_gradient(self.eps)

        @eval_norm_impl.def_vmap
        def _(_, batch_tree, input: jax.Array, state: BatchNormState):
            output, running_mean, running_var = eval_batch_norm(
                input=input,
                running_mean=state.running_mean,
                running_var=state.running_var,
                momentum=0.0,
                eps=eps,
                weight=self.weight,
                bias=self.bias,
                axis=self.axis,
                axis_name=self.axis_name,
            )
            state = BatchNormState(running_mean, running_var)
            return (output, state), tuple(batch_tree)

        return eval_norm_impl(input, state)


@tree_eval.def_eval(BatchNorm)
def _(batch_norm: BatchNorm) -> EvalBatchNorm:
    return EvalBatchNorm(
        in_features=batch_norm.in_features,
        momentum=batch_norm.momentum,  # ignored
        eps=batch_norm.eps,
        weight_init=lambda *_: batch_norm.weight,
        bias_init=None if batch_norm.bias is None else lambda *_: batch_norm.bias,
        axis=batch_norm.axis,
        axis_name=batch_norm.axis_name,
        key=None,
    )


@tree_state.def_state(BatchNorm)
def _(batch_norm: BatchNorm) -> BatchNormState:
    running_mean = jnp.zeros([batch_norm.in_features])
    running_var = jnp.ones([batch_norm.in_features])
    return BatchNormState(running_mean, running_var)


def weight_norm(leaf: T, axis: int | None = -1, eps: float = 1e-12) -> T:
    """Apply L2 weight normalization to an input.

    Args:
        leaf: the input to be normalized. If ``leaf`` is not an array, then it will
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
        ...     def __call__(self, inputs: jax.Array) -> jax.Array:
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
