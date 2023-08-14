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

from typing import Callable, Literal, Protocol, Union, get_args

import jax
import jax.numpy as jnp
from jax import lax

import serket as sk
from serket.nn.utils import IsInstance, Range, ScalarLike


@sk.autoinit
class AdaptiveLeakyReLU(sk.TreeClass):
    """Leaky ReLU activation function

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """

    a: float = sk.field(default=1.0, callbacks=[Range(0), ScalarLike()])
    v: float = sk.field(default=1.0, callbacks=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        v = jax.lax.stop_gradient(self.v)
        return jnp.maximum(0, self.a * x) - v * jnp.maximum(0, -self.a * x)


@sk.autoinit
class AdaptiveReLU(sk.TreeClass):
    """ReLU activation function with learnable parameters

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """

    a: float = sk.field(default=1.0, callbacks=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jnp.maximum(0, self.a * x)


@sk.autoinit
class AdaptiveSigmoid(sk.TreeClass):
    """Sigmoid activation function with learnable `a` parameter

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """

    a: float = sk.field(default=1.0, callbacks=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return 1 / (1 + jnp.exp(-self.a * x))


@sk.autoinit
class AdaptiveTanh(sk.TreeClass):
    """Tanh activation function with learnable parameters

    Reference:
        https://arxiv.org/pdf/1906.01170.pdf.
    """

    a: float = sk.field(default=1.0, callbacks=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        a = self.a
        return (jnp.exp(a * x) - jnp.exp(-a * x)) / (jnp.exp(a * x) + jnp.exp(-a * x))


@sk.autoinit
class CeLU(sk.TreeClass):
    """Celu activation function"""

    alpha: float = sk.field(default=1.0, callbacks=[ScalarLike()])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.celu(x, alpha=lax.stop_gradient(self.alpha))


@sk.autoinit
class ELU(sk.TreeClass):
    """Exponential linear unit"""

    alpha: float = sk.field(default=1.0, callbacks=[ScalarLike()])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.elu(x, alpha=lax.stop_gradient(self.alpha))


@sk.autoinit
class GELU(sk.TreeClass):
    """Gaussian error linear unit"""

    approximate: bool = sk.field(default=1.0, callbacks=[IsInstance(bool)])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.gelu(x, approximate=self.approximate)


@sk.autoinit
class GLU(sk.TreeClass):
    """Gated linear unit"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.glu(x)


@sk.autoinit
class HardShrink(sk.TreeClass):
    """Hard shrink activation function"""

    alpha: float = sk.field(default=0.5, callbacks=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        alpha = lax.stop_gradient(self.alpha)
        return jnp.where(x > alpha, x, jnp.where(x < -alpha, x, 0.0))


class HardSigmoid(sk.TreeClass):
    """Hard sigmoid activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.hard_sigmoid(x)


class HardSwish(sk.TreeClass):
    """Hard swish activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.hard_swish(x)


class HardTanh(sk.TreeClass):
    """Hard tanh activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.hard_tanh(x)


class LogSigmoid(sk.TreeClass):
    """Log sigmoid activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.log_sigmoid(x)


class LogSoftmax(sk.TreeClass):
    """Log softmax activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.log_softmax(x)


@sk.autoinit
class LeakyReLU(sk.TreeClass):
    """Leaky ReLU activation function"""

    negative_slope: float = sk.field(default=0.01, callbacks=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.leaky_relu(x, lax.stop_gradient(self.negative_slope))


class ReLU(sk.TreeClass):
    """ReLU activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.relu(x)


class ReLU6(sk.TreeClass):
    """ReLU6 activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.relu6(x)


class SeLU(sk.TreeClass):
    """Scaled Exponential Linear Unit"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.selu(x)


class Sigmoid(sk.TreeClass):
    """Sigmoid activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.sigmoid(x)


class SoftPlus(sk.TreeClass):
    """SoftPlus activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.softplus(x)


class SoftSign(sk.TreeClass):
    """SoftSign activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return x / (1 + jnp.abs(x))


@sk.autoinit
class SoftShrink(sk.TreeClass):
    """SoftShrink activation function"""

    alpha: float = sk.field(default=0.5, callbacks=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        alpha = lax.stop_gradient(self.alpha)
        return jnp.where(
            x < -alpha,
            x + alpha,
            jnp.where(x > alpha, x - alpha, 0.0),
        )


class SquarePlus(sk.TreeClass):
    """SquarePlus activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return 0.5 * (x + jnp.sqrt(x * x + 4))


class Swish(sk.TreeClass):
    """Swish activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.swish(x)


class Tanh(sk.TreeClass):
    """Tanh activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.tanh(x)


class TanhShrink(sk.TreeClass):
    """TanhShrink activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return x - jax.nn.tanh(x)


@sk.autoinit
class ThresholdedReLU(sk.TreeClass):
    """Thresholded ReLU activation function."""

    theta: float = sk.field(default=1.0, callbacks=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        theta = lax.stop_gradient(self.theta)
        return jnp.where(x > theta, x, 0)


class Mish(sk.TreeClass):
    """Mish activation function https://arxiv.org/pdf/1908.08681.pdf."""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return x * jax.nn.tanh(jax.nn.softplus(x))


@sk.autoinit
class PReLU(sk.TreeClass):
    """Parametric ReLU activation function"""

    a: float = sk.field(default=0.25, callbacks=[Range(0), ScalarLike()])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jnp.where(x >= 0, x, x * self.a)


@sk.autoinit
class Snake(sk.TreeClass):
    """Snake activation function

    Args:
        a: scalar (frequency) parameter of the activation function, default is 1.0.

    Reference:
        https://arxiv.org/pdf/2006.08195.pdf.
    """

    a: float = sk.field(callbacks=[Range(0), ScalarLike()], default=1.0)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        a = lax.stop_gradient(self.a)
        return x + (1 - jnp.cos(2 * a * x)) / (2 * a)


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
    AdaptiveLeakyReLU,
    AdaptiveReLU,
    AdaptiveSigmoid,
    AdaptiveTanh,
    CeLU,
    ELU,
    GELU,
    GLU,
    HardShrink,
    HardSigmoid,
    HardSwish,
    HardTanh,
    LeakyReLU,
    LogSigmoid,
    LogSoftmax,
    Mish,
    PReLU,
    ReLU,
    ReLU6,
    SeLU,
    Sigmoid,
    Snake,
    SoftPlus,
    SoftShrink,
    SoftSign,
    SquarePlus,
    Swish,
    Tanh,
    TanhShrink,
    ThresholdedReLU,
]


act_map: dict[str, sk.TreeClass] = dict(zip(get_args(ActivationLiteral), acts))

ActivationFunctionType = Callable[[jax.typing.ArrayLike], jax.Array]
ActivationType = Union[ActivationLiteral, ActivationFunctionType]


class ActivationClassType(Protocol):
    def __call__(self, x: jax.typing.ArrayLike) -> jax.Array:
        ...


def resolve_activation(act: ActivationType) -> ActivationFunctionType:
    # in case the user passes a trainable activation function
    # we need to make a copy of it to avoid unpredictable side effects
    if isinstance(act, str):
        if act in act_map:
            return act_map[act]()
        raise ValueError(f"Unknown {act=}, available activations: {list(act_map)}")
    return act


def def_act_entry(key: str, act: ActivationClassType) -> None:
    """Register a custom activation function key for use in ``serket`` layers.

    Args:
        key: The key to register the function under.
        act: a class with a ``__call__`` method that takes a single argument
            and returns a ``jax`` array.

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
        >>> sk.def_act_entry("my_act", MyTrainableActivation)
        >>> x = jnp.ones((1, 1))
        >>> sk.nn.FNN([1, 1, 1], act="my_act", weight_init="ones", bias_init=None)(x)
        Array([[10.]], dtype=float32)
    """
    if key in act_map:
        raise ValueError(f"`init_key` {key=} already registered")

    if not isinstance(act, type):
        raise ValueError(f"Expected a class, got {act=}")
    if not callable(act):
        raise ValueError(f"Expected a class with a `__call__` method, got {act=}")

    act_map[key] = act
