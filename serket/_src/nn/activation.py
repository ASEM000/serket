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
from typing import Callable, Literal, TypeVar, Union, get_args

import jax
import jax.numpy as jnp
from jax import lax

import serket as sk
from serket._src.utils import IsInstance, Range, ScalarLike

T = TypeVar("T")


def adaptive_leaky_relu(
    input: jax.typing.ArrayLike,
    a: float = 1.0,
    v: float = 1.0,
) -> jax.Array:
    """Adaptive Leaky ReLU activation function

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """
    return jnp.maximum(0, a * input) - v * jnp.maximum(0, -a * input)


@sk.autoinit
class AdaptiveLeakyReLU(sk.TreeClass):
    """Leaky ReLU activation function

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """

    a: float = sk.field(default=1.0, on_setattr=[Range(0), ScalarLike()])
    v: float = sk.field(
        default=1.0,
        on_setattr=[Range(0), ScalarLike()],
        on_getattr=[lax.stop_gradient_p.bind],
    )

    def __call__(self, input: jax.Array) -> jax.Array:
        return adaptive_leaky_relu(input, self.a, self.v)


def adaptive_relu(input: jax.typing.ArrayLike, a: float = 1.0) -> jax.Array:
    """Adaptive ReLU activation function

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """
    return jnp.maximum(0, a * input)


@sk.autoinit
class AdaptiveReLU(sk.TreeClass):
    """ReLU activation function with learnable parameters

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """

    a: float = sk.field(default=1.0, on_setattr=[Range(0), ScalarLike()])

    def __call__(self, input: jax.Array) -> jax.Array:
        return adaptive_relu(input, self.a)


def adaptive_sigmoid(input: jax.typing.ArrayLike, a: float = 1.0) -> jax.Array:
    """Adaptive sigmoid activation function

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """
    return 1 / (1 + jnp.exp(-a * input))


@sk.autoinit
class AdaptiveSigmoid(sk.TreeClass):
    """Sigmoid activation function with learnable `a` parameter

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """

    a: float = sk.field(default=1.0, on_setattr=[Range(0), ScalarLike()])

    def __call__(self, input: jax.Array) -> jax.Array:
        return adaptive_sigmoid(input, self.a)


def adaptive_tanh(input: jax.typing.ArrayLike, a: float = 1.0) -> jax.Array:
    """Adaptive tanh activation function

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """
    return (jnp.exp(a * input) - jnp.exp(-a * input)) / (
        jnp.exp(a * input) + jnp.exp(-a * input)
    )


@sk.autoinit
class AdaptiveTanh(sk.TreeClass):
    """Tanh activation function with learnable parameters

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """

    a: float = sk.field(default=1.0, on_setattr=[Range(0), ScalarLike()])

    def __call__(self, input: jax.Array) -> jax.Array:
        return adaptive_tanh(input, self.a)


@sk.autoinit
class CeLU(sk.TreeClass):
    """Celu activation function"""

    alpha: float = sk.field(
        default=1.0,
        on_setattr=[ScalarLike()],
        on_getattr=[lax.stop_gradient_p.bind],
    )

    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.nn.celu(input, alpha=self.alpha)


@sk.autoinit
class ELU(sk.TreeClass):
    """Exponential linear unit"""

    alpha: float = sk.field(
        default=1.0,
        on_setattr=[ScalarLike()],
        on_getattr=[lax.stop_gradient_p.bind],
    )

    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.nn.elu(input, alpha=self.alpha)


@sk.autoinit
class GELU(sk.TreeClass):
    """Gaussian error linear unit"""

    approximate: bool = sk.field(default=False, on_setattr=[IsInstance(bool)])

    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.nn.gelu(input, approximate=self.approximate)


@sk.autoinit
class GLU(sk.TreeClass):
    """Gated linear unit"""

    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.nn.glu(input)


def hard_shrink(input: jax.typing.ArrayLike, alpha: float = 0.5) -> jax.Array:
    """Hard shrink activation function

    Reference:
        https://arxiv.org/pdf/1702.00783.pdf.
    """
    return jnp.where(input > alpha, input, jnp.where(input < -alpha, input, 0.0))


@sk.autoinit
class HardShrink(sk.TreeClass):
    """Hard shrink activation function"""

    alpha: float = sk.field(
        default=0.5,
        on_setattr=[Range(0), ScalarLike()],
        on_getattr=[lax.stop_gradient_p.bind],
    )

    def __call__(self, input: jax.Array) -> jax.Array:
        return hard_shrink(input, self.alpha)


class HardSigmoid(sk.TreeClass):
    """Hard sigmoid activation function"""

    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.nn.hard_sigmoid(input)


class HardSwish(sk.TreeClass):
    """Hard swish activation function"""

    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.nn.hard_swish(input)


class HardTanh(sk.TreeClass):
    """Hard tanh activation function"""

    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.nn.hard_tanh(input)


class LogSigmoid(sk.TreeClass):
    """Log sigmoid activation function"""

    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.nn.log_sigmoid(input)


class LogSoftmax(sk.TreeClass):
    """Log softmax activation function"""

    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.nn.log_softmax(input)


@sk.autoinit
class LeakyReLU(sk.TreeClass):
    """Leaky ReLU activation function"""

    negative_slope: float = sk.field(
        default=0.01,
        on_setattr=[Range(0), ScalarLike()],
        on_getattr=[lax.stop_gradient_p.bind],
    )

    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.nn.leaky_relu(input, self.negative_slope)


class ReLU(sk.TreeClass):
    """ReLU activation function"""

    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.nn.relu(input)


class ReLU6(sk.TreeClass):
    """ReLU6 activation function"""

    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.nn.relu6(input)


class SeLU(sk.TreeClass):
    """Scaled Exponential Linear Unit"""

    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.nn.selu(input)


class Sigmoid(sk.TreeClass):
    """Sigmoid activation function"""

    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.nn.sigmoid(input)


class SoftPlus(sk.TreeClass):
    """SoftPlus activation function"""

    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.nn.softplus(input)


def softsign(x: jax.typing.ArrayLike) -> jax.Array:
    """SoftSign activation function"""
    return x / (1 + jnp.abs(x))


class SoftSign(sk.TreeClass):
    """SoftSign activation function"""

    def __call__(self, input: jax.Array) -> jax.Array:
        return softsign(input)


def softshrink(input: jax.typing.ArrayLike, alpha: float = 0.5) -> jax.Array:
    """Soft shrink activation function

    Reference:
        https://arxiv.org/pdf/1702.00783.pdf.
    """
    return jnp.where(
        input < -alpha,
        input + alpha,
        jnp.where(input > alpha, input - alpha, 0.0),
    )


@sk.autoinit
class SoftShrink(sk.TreeClass):
    """SoftShrink activation function"""

    alpha: float = sk.field(
        default=0.5,
        on_setattr=[Range(0), ScalarLike()],
        on_getattr=[lax.stop_gradient_p.bind],
    )

    def __call__(self, input: jax.Array) -> jax.Array:
        return softshrink(input, self.alpha)


def squareplus(input: jax.typing.ArrayLike) -> jax.Array:
    """SquarePlus activation function

    Reference:
        https://arxiv.org/pdf/1908.08681.pdf.
    """
    return 0.5 * (input + jnp.sqrt(input * input + 4))


class SquarePlus(sk.TreeClass):
    """SquarePlus activation function"""

    def __call__(self, input: jax.Array) -> jax.Array:
        return squareplus(input)


class Swish(sk.TreeClass):
    """Swish activation function"""

    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.nn.swish(input)


class Tanh(sk.TreeClass):
    """Tanh activation function"""

    def __call__(self, input: jax.Array) -> jax.Array:
        return jax.nn.tanh(input)


def tanh_shrink(input: jax.typing.ArrayLike) -> jax.Array:
    """TanhShrink activation function"""
    return input - jnp.tanh(input)


class TanhShrink(sk.TreeClass):
    """TanhShrink activation function"""

    def __call__(self, input: jax.Array) -> jax.Array:
        return tanh_shrink(input)


def thresholded_relu(input: jax.typing.ArrayLike, theta: float = 1.0) -> jax.Array:
    """Thresholded ReLU activation function

    Reference:
        https://arxiv.org/pdf/1911.09737.pdf.
    """
    return jnp.where(input > theta, input, 0)


@sk.autoinit
class ThresholdedReLU(sk.TreeClass):
    """Thresholded ReLU activation function."""

    theta: float = sk.field(
        default=1.0,
        on_setattr=[Range(0), ScalarLike()],
        on_getattr=[lax.stop_gradient_p.bind],
    )

    def __call__(self, input: jax.Array) -> jax.Array:
        return thresholded_relu(input, self.theta)


def mish(input: jax.typing.ArrayLike) -> jax.Array:
    """Mish activation function https://arxiv.org/pdf/1908.08681.pdf."""
    return input * jax.nn.tanh(jax.nn.softplus(input))


class Mish(sk.TreeClass):
    """Mish activation function https://arxiv.org/pdf/1908.08681.pdf."""

    def __call__(self, input: jax.Array) -> jax.Array:
        return mish(input)


def prelu(input: jax.typing.ArrayLike, a: float = 0.25) -> jax.Array:
    """Parametric ReLU activation function"""
    return jnp.where(input >= 0, input, input * a)


@sk.autoinit
class PReLU(sk.TreeClass):
    """Parametric ReLU activation function"""

    a: float = sk.field(default=0.25, on_setattr=[Range(0), ScalarLike()])

    def __call__(self, input: jax.Array) -> jax.Array:
        return prelu(input, self.a)


def snake(input: jax.typing.ArrayLike, a: float = 1.0) -> jax.Array:
    """Snake activation function

    Args:
        a: scalar (frequency) parameter of the activation function, default is 1.0.

    Reference:
        https://arxiv.org/pdf/2006.08195.pdf.
    """
    return input + (1 - jnp.cos(2 * a * input)) / (2 * a)


@sk.autoinit
class Snake(sk.TreeClass):
    """Snake activation function

    Args:
        a: scalar (frequency) parameter of the activation function, default is 1.0.

    Reference:
        https://arxiv.org/pdf/2006.08195.pdf.
    """

    a: float = sk.field(
        default=1.0,
        on_setattr=[Range(0), ScalarLike()],
        on_getattr=[lax.stop_gradient_p.bind],
    )

    def __call__(self, input: jax.Array) -> jax.Array:
        return snake(input, self.a)


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
        >>> import jax.random as jr
        >>> @sk.autoinit
        ... class MyTrainableActivation(sk.TreeClass):
        ...    my_param: float = 10.0
        ...    def __call__(self, x):
        ...        return x * self.my_param
        >>> sk.def_act_entry("my_act", MyTrainableActivation())
    """
    if key in act_map:
        raise ValueError(f"`init_key` {key=} already registered")

    if not callable(act):
        raise TypeError(f"{act=} must be a callable object")

    act_map[key] = act
