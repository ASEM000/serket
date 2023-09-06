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
from typing import Callable, Literal, TypeVar, Union, get_args

import jax
import jax.numpy as jnp
from jax import lax

import serket as sk
from serket.utils import IsInstance, Range, ScalarLike

T = TypeVar("T")


def adaptive_leaky_relu(
    x: jax.typing.ArrayLike,
    a: float = 1.0,
    v: float = 1.0,
) -> jax.Array:
    """Adaptive Leaky ReLU activation function

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """
    return jnp.maximum(0, a * x) - v * jnp.maximum(0, -a * x)


@sk.autoinit
class AdaptiveLeakyReLU(sk.TreeClass):
    """Leaky ReLU activation function

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """

    a: float = sk.field(default=1.0, on_setattr=[Range(0), ScalarLike()])
    v: float = sk.field(default=1.0, on_setattr=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array) -> jax.Array:
        v = jax.lax.stop_gradient(self.v)
        return adaptive_leaky_relu(x, self.a, v)


def adaptive_relu(x: jax.typing.ArrayLike, a: float = 1.0) -> jax.Array:
    """Adaptive ReLU activation function

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """
    return jnp.maximum(0, a * x)


@sk.autoinit
class AdaptiveReLU(sk.TreeClass):
    """ReLU activation function with learnable parameters

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """

    a: float = sk.field(default=1.0, on_setattr=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array) -> jax.Array:
        return adaptive_relu(x, self.a)


def adaptive_sigmoid(x: jax.typing.ArrayLike, a: float = 1.0) -> jax.Array:
    """Adaptive sigmoid activation function

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """
    return 1 / (1 + jnp.exp(-a * x))


@sk.autoinit
class AdaptiveSigmoid(sk.TreeClass):
    """Sigmoid activation function with learnable `a` parameter

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """

    a: float = sk.field(default=1.0, on_setattr=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array) -> jax.Array:
        return adaptive_sigmoid(x, self.a)


def adaptive_tanh(x: jax.typing.ArrayLike, a: float = 1.0) -> jax.Array:
    """Adaptive tanh activation function

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """
    return (jnp.exp(a * x) - jnp.exp(-a * x)) / (jnp.exp(a * x) + jnp.exp(-a * x))


@sk.autoinit
class AdaptiveTanh(sk.TreeClass):
    """Tanh activation function with learnable parameters

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """

    a: float = sk.field(default=1.0, on_setattr=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array) -> jax.Array:
        a = self.a
        return adaptive_tanh(x, a)


@sk.autoinit
class CeLU(sk.TreeClass):
    """Celu activation function"""

    alpha: float = sk.field(default=1.0, on_setattr=[ScalarLike()])

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.celu(x, alpha=lax.stop_gradient(self.alpha))


@sk.autoinit
class ELU(sk.TreeClass):
    """Exponential linear unit"""

    alpha: float = sk.field(default=1.0, on_setattr=[ScalarLike()])

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.elu(x, alpha=lax.stop_gradient(self.alpha))


@sk.autoinit
class GELU(sk.TreeClass):
    """Gaussian error linear unit"""

    approximate: bool = sk.field(default=False, on_setattr=[IsInstance(bool)])

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.gelu(x, approximate=self.approximate)


@sk.autoinit
class GLU(sk.TreeClass):
    """Gated linear unit"""

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.glu(x)


def hard_shrink(x: jax.typing.ArrayLike, alpha: float = 0.5) -> jax.Array:
    """Hard shrink activation function

    Reference:
        https://arxiv.org/pdf/1702.00783.pdf.
    """
    return jnp.where(x > alpha, x, jnp.where(x < -alpha, x, 0.0))


@sk.autoinit
class HardShrink(sk.TreeClass):
    """Hard shrink activation function"""

    alpha: float = sk.field(default=0.5, on_setattr=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array) -> jax.Array:
        alpha = lax.stop_gradient(self.alpha)
        return hard_shrink(x, alpha)


class HardSigmoid(sk.TreeClass):
    """Hard sigmoid activation function"""

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.hard_sigmoid(x)


class HardSwish(sk.TreeClass):
    """Hard swish activation function"""

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.hard_swish(x)


class HardTanh(sk.TreeClass):
    """Hard tanh activation function"""

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.hard_tanh(x)


class LogSigmoid(sk.TreeClass):
    """Log sigmoid activation function"""

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.log_sigmoid(x)


class LogSoftmax(sk.TreeClass):
    """Log softmax activation function"""

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.log_softmax(x)


@sk.autoinit
class LeakyReLU(sk.TreeClass):
    """Leaky ReLU activation function"""

    negative_slope: float = sk.field(default=0.01, on_setattr=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.leaky_relu(x, lax.stop_gradient(self.negative_slope))


class ReLU(sk.TreeClass):
    """ReLU activation function"""

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.relu(x)


class ReLU6(sk.TreeClass):
    """ReLU6 activation function"""

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.relu6(x)


class SeLU(sk.TreeClass):
    """Scaled Exponential Linear Unit"""

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.selu(x)


class Sigmoid(sk.TreeClass):
    """Sigmoid activation function"""

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.sigmoid(x)


class SoftPlus(sk.TreeClass):
    """SoftPlus activation function"""

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.softplus(x)


def softsign(x: jax.typing.ArrayLike) -> jax.Array:
    """SoftSign activation function"""
    return x / (1 + jnp.abs(x))


class SoftSign(sk.TreeClass):
    """SoftSign activation function"""

    def __call__(self, x: jax.Array) -> jax.Array:
        return softsign(x)


def softshrink(x: jax.typing.ArrayLike, alpha: float = 0.5) -> jax.Array:
    """Soft shrink activation function

    Reference:
        https://arxiv.org/pdf/1702.00783.pdf.
    """
    return jnp.where(
        x < -alpha,
        x + alpha,
        jnp.where(x > alpha, x - alpha, 0.0),
    )


@sk.autoinit
class SoftShrink(sk.TreeClass):
    """SoftShrink activation function"""

    alpha: float = sk.field(default=0.5, on_setattr=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array) -> jax.Array:
        alpha = lax.stop_gradient(self.alpha)
        return softshrink(x, alpha)


def squareplus(x: jax.typing.ArrayLike) -> jax.Array:
    """SquarePlus activation function

    Reference:
        https://arxiv.org/pdf/1908.08681.pdf.
    """
    return 0.5 * (x + jnp.sqrt(x * x + 4))


class SquarePlus(sk.TreeClass):
    """SquarePlus activation function"""

    def __call__(self, x: jax.Array) -> jax.Array:
        return squareplus(x)


class Swish(sk.TreeClass):
    """Swish activation function"""

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.swish(x)


class Tanh(sk.TreeClass):
    """Tanh activation function"""

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.tanh(x)


def tanh_shrink(x: jax.typing.ArrayLike) -> jax.Array:
    """TanhShrink activation function"""
    return x - jnp.tanh(x)


class TanhShrink(sk.TreeClass):
    """TanhShrink activation function"""

    def __call__(self, x: jax.Array) -> jax.Array:
        return tanh_shrink(x)


def thresholded_relu(x: jax.typing.ArrayLike, theta: float = 1.0) -> jax.Array:
    """Thresholded ReLU activation function

    Reference:
        https://arxiv.org/pdf/1911.09737.pdf.
    """
    return jnp.where(x > theta, x, 0)


@sk.autoinit
class ThresholdedReLU(sk.TreeClass):
    """Thresholded ReLU activation function."""

    theta: float = sk.field(default=1.0, on_setattr=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array) -> jax.Array:
        theta = lax.stop_gradient(self.theta)
        return thresholded_relu(x, theta)


def mish(x: jax.typing.ArrayLike) -> jax.Array:
    """Mish activation function https://arxiv.org/pdf/1908.08681.pdf."""
    return x * jax.nn.tanh(jax.nn.softplus(x))


class Mish(sk.TreeClass):
    """Mish activation function https://arxiv.org/pdf/1908.08681.pdf."""

    def __call__(self, x: jax.Array) -> jax.Array:
        return mish(x)


def prelu(x: jax.typing.ArrayLike, a: float = 0.25) -> jax.Array:
    """Parametric ReLU activation function"""
    return jnp.where(x >= 0, x, x * a)


@sk.autoinit
class PReLU(sk.TreeClass):
    """Parametric ReLU activation function"""

    a: float = sk.field(default=0.25, on_setattr=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array) -> jax.Array:
        return prelu(x, self.a)


def snake(x: jax.typing.ArrayLike, a: float = 1.0) -> jax.Array:
    """Snake activation function

    Args:
        a: scalar (frequency) parameter of the activation function, default is 1.0.

    Reference:
        https://arxiv.org/pdf/2006.08195.pdf.
    """
    return x + (1 - jnp.cos(2 * a * x)) / (2 * a)


@sk.autoinit
class Snake(sk.TreeClass):
    """Snake activation function

    Args:
        a: scalar (frequency) parameter of the activation function, default is 1.0.

    Reference:
        https://arxiv.org/pdf/2006.08195.pdf.
    """

    a: float = sk.field(on_setattr=[Range(0), ScalarLike()], default=1.0)

    def __call__(self, x: jax.Array) -> jax.Array:
        a = lax.stop_gradient(self.a)
        return snake(x, a)


# useful for building layers from configuration text
ActivationLiteral = Literal[
    "adaptive_leaky_relu",
    "adaptive_relu",
    "adaptive_sigmoid",
    "adaptive_tanh",
    "celu",
    "elu",
    "gelu",
    "glu",
    "hard_shrink",
    "hard_sigmoid",
    "hard_swish",
    "hard_tanh",
    "leaky_relu",
    "log_sigmoid",
    "log_softmax",
    "mish",
    "prelu",
    "relu",
    "relu6",
    "selu",
    "sigmoid",
    "snake",
    "softplus",
    "softshrink",
    "softsign",
    "squareplus",
    "swish",
    "tanh",
    "tanh_shrink",
    "thresholded_relu",
]


acts = [
    adaptive_leaky_relu,
    adaptive_relu,
    adaptive_sigmoid,
    adaptive_tanh,
    jax.nn.celu,
    jax.nn.elu,
    jax.nn.gelu,
    jax.nn.glu,
    hard_shrink,
    jax.nn.hard_sigmoid,
    jax.nn.hard_swish,
    jax.nn.hard_tanh,
    jax.nn.leaky_relu,
    jax.nn.log_sigmoid,
    jax.nn.log_softmax,
    mish,
    prelu,
    jax.nn.relu,
    jax.nn.relu6,
    jax.nn.selu,
    jax.nn.sigmoid,
    snake,
    jax.nn.softplus,
    softshrink,
    softsign,
    squareplus,
    jax.nn.swish,
    jax.nn.tanh,
    tanh_shrink,
    thresholded_relu,
]


ActivationFunctionType = Callable[[jax.typing.ArrayLike], jax.Array]
ActivationType = Union[ActivationLiteral, ActivationFunctionType]
act_map = dict(zip(get_args(ActivationLiteral), acts))


@ft.singledispatch
def resolve_activation(act: T) -> T:
    return act


@resolve_activation.register(str)
def _(act: str):
    try:
        return jax.tree_map(lambda x: x, act_map[act])
    except KeyError:
        raise ValueError(f"Unknown {act=}, available activations: {list(act_map)}")


def def_act_entry(key: str, act: ActivationFunctionType) -> None:
    """Register a custom activation function key for use in ``serket`` layers.

    Args:
        key: The key to register the function under.
        act: a callable object that takes a single argument and returns a ``jax``
            array.

    Note:
        The registered key can be used in any of ``serket`` ``act_*`` arguments as
        substitution for the function.

    Note:
        By design, activation functions can be passed directly to ``serket`` layers
        with the ``act`` argument. This function is useful if you want to
        represent activation functions as a string in a configuration file.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> @sk.autoinit
        ... class MyTrainableActivation(sk.TreeClass):
        ...    my_param: float = 10.0
        ...    def __call__(self, x):
        ...        return x * self.my_param
        >>> sk.def_act_entry("my_act", MyTrainableActivation())
        >>> x = jnp.ones((1, 1))
        >>> sk.nn.FNN([1, 1, 1], act="my_act", weight_init="ones", bias_init=None)(x)
        Array([[10.]], dtype=float32)
    """
    if key in act_map:
        raise ValueError(f"`init_key` {key=} already registered")

    if not callable(act):
        raise TypeError(f"{act=} must be a callable object")

    act_map[key] = act
