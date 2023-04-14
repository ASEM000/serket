from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytreeclass as pytc
from jax.lax import stop_gradient

from serket.nn.callbacks import non_negative_scalar_cbs, scalar_like_cb

ActivationType = Callable[[Any], Any]


def adaptive_leaky_relu(x: jax.Array, a: float = 1.0, v: float = 1.0) -> jax.Array:
    return jnp.maximum(0, a * x) - v * jnp.maximum(0, -a * x)


def adaptive_relu(x: jax.Array, a: float = 1.0) -> jax.Array:
    return jnp.maximum(0, a * x)


def adaptive_sigmoid(x: jax.Array, a: float = 1.0) -> jax.Array:
    return 1 / (1 + jnp.exp(-a * x))


def adaptive_tanh(x: jax.Array, a: float = 1.0) -> jax.Array:
    return (jnp.exp(a * x) - jnp.exp(-a * x)) / (jnp.exp(a * x) + jnp.exp(-a * x))


def hard_shrink(x: jax.Array, alpha: float = 0.5) -> jax.Array:
    return jnp.where(x > alpha, x, jnp.where(x < -alpha, x, 0.0))


def parametric_relu(x: jax.Array, a: float = 0.25) -> jax.Array:
    return jnp.where(x >= 0, x, x * a)


def self_scalable_tanh(x: jax.Array, beta: float = 1.0) -> jax.Array:
    return jnp.tanh(x) * (1.0 + beta * x)


def soft_shrink(x: jax.Array, alpha: float = 0.5) -> jax.Array:
    return jnp.where(
        x < -alpha,
        x + alpha,
        jnp.where(x > alpha, x - alpha, 0.0),
    )


def silu(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)


def square_plus(x: jax.Array) -> jax.Array:
    return 0.5 * (x + jnp.sqrt(x * x + 4))


def soft_sign(x: jax.Array) -> jax.Array:
    return x / (1 + jnp.abs(x))


def thresholded_relu(x: jax.Array, theta: float = 1.0) -> jax.Array:
    return jnp.where(x > theta, x, 0)


def mish(x: jax.Array) -> jax.Array:
    return x * jax.nn.tanh(jax.nn.softplus(x))


def snake(x: jax.Array, frequency: float = 1.0) -> jax.Array:
    return x + (1 - jnp.cos(2 * frequency * x)) / (2 * frequency)


fkwargs = dict(repr=False, callbacks=[jtu.Partial])


@pytc.treeclass
class AdaptiveLeakyReLU:
    r"""Leaky ReLU activation function with learnable `a` parameter https://arxiv.org/pdf/1906.01170.pdf.

    .. math::
        \text{AdaptiveLeakyReLU}(x) = \max(0, a x) - v \max(0, -a x)
    """

    a: float = pytc.field(default=1.0, callbacks=[*non_negative_scalar_cbs])
    v: float = pytc.field(default=1.0, callbacks=[*non_negative_scalar_cbs])
    func: ActivationType = pytc.field(default=adaptive_leaky_relu, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x, self.a, stop_gradient(self.v))


@pytc.treeclass
class AdaptiveReLU:
    r"""ReLU activation function with learnable parameters https://arxiv.org/pdf/1906.01170.pdf.

    .. math::
        \text{AdaptiveReLU}(x) = \max(0, a x)
    """

    a: float = pytc.field(default=1.0, callbacks=[*non_negative_scalar_cbs])
    func: ActivationType = pytc.field(default=adaptive_relu, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x, self.a)


@pytc.treeclass
class AdaptiveSigmoid:
    r"""Sigmoid activation function with learnable `a` parameter https://arxiv.org/pdf/1906.01170.pdf.

    .. math::
        \text{AdaptiveSigmoid}(x) = \frac{1}{1 + \exp(-a x)}
    """

    a: float = pytc.field(default=1.0, callbacks=[*non_negative_scalar_cbs])
    func: ActivationType = pytc.field(default=adaptive_sigmoid, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x, self.a)


@pytc.treeclass
class AdaptiveTanh:
    r"""Tanh activation function with learnable parameters https://arxiv.org/pdf/1906.01170.pdf.

    .. math::
        \text{AdaptiveTanh}(x) = \frac{\exp(a x) - \exp(-a x)}{\exp(a x) + \exp(-a x)}
    """

    a: float = pytc.field(default=1.0, callbacks=[*non_negative_scalar_cbs])
    func: ActivationType = pytc.field(default=adaptive_tanh, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x, self.a)


@pytc.treeclass
class CeLU:
    r"""Celu activation function

    .. math::
        \text{CeLU}(x) = \max(0, x) + \min(0, \alpha \exp(x / \alpha) - 1)

    """

    alpha: float = pytc.field(default=1.0)
    func: ActivationType = pytc.field(default=jax.nn.celu, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x, alpha=stop_gradient(self.alpha))


@pytc.treeclass
class ELU:
    r"""Exponential linear unit"""

    alpha: float = pytc.field(default=1.0)
    func: ActivationType = pytc.field(default=jax.nn.elu, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x, alpha=stop_gradient(self.alpha))


@pytc.treeclass
class GELU:
    r"""Gaussian error linear unit

    .. math::
        \mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{erf} \left(
        \frac{x}{\sqrt{2}} \right) \right)
    """

    approximate: bool = pytc.field(default=True)
    func: ActivationType = pytc.field(default=jax.nn.gelu, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x, approximate=stop_gradient(self.approximate))


@pytc.treeclass
class GLU:
    r"""Gated linear unit

    .. math::
        \mathrm{glu}(x) = x_1 \odot \sigma(x_2)
    """
    func: ActivationType = pytc.field(default=jax.nn.glu, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class HardSILU:
    r"""Hard SILU activation function

    .. math::
        \mathrm{hard\_silu}(x) = x \cdot \mathrm{hard\_sigmoid}(x)
    """
    func: ActivationType = pytc.field(default=jax.nn.hard_silu, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class HardShrink:
    r"""Hard shrink activation function

    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}
    """

    alpha: float = pytc.field(default=0.5)
    func: ActivationType = pytc.field(default=hard_shrink, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x, stop_gradient(self.alpha))


@pytc.treeclass
class HardSigmoid:
    r"""Hard sigmoid activation function

    .. math::
        \mathrm{hard\_sigmoid}(x) = \frac{\mathrm{relu6}(x + 3)}{6}
    """
    func: ActivationType = pytc.field(default=jax.nn.hard_sigmoid, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class HardSwish:
    r"""Hard swish activation function

    .. math::
        \mathrm{hard\_silu}(x) = x \cdot \mathrm{hard\_sigmoid}(x)
    """
    func: ActivationType = pytc.field(default=jax.nn.hard_swish, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class HardTanh:
    r"""Hard tanh activation function

  .. math::
    \mathrm{hard\_tanh}(x) = \begin{cases}
      -1, & x < -1\\
      x, & -1 \le x \le 1\\
      1, & 1 < x
    \end{cases}
    """
    func: ActivationType = pytc.field(default=jax.nn.hard_tanh, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class LogSigmoid:
    r"""Log sigmoid activation function

    .. math::
        \mathrm{log\_sigmoid}(x) = \log(\mathrm{sigmoid}(x)) = -\log(1 + e^{-x})

    """
    func: ActivationType = pytc.field(default=jax.nn.log_sigmoid, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class LogSoftmax:
    r"""Log softmax activation function

    .. math ::
        \mathrm{log\_softmax}(x) = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)}
        \right)
    """
    func: ActivationType = pytc.field(default=jax.nn.log_softmax, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class LeakyReLU:
    r"""Leaky ReLU activation function

    .. math::
        \mathrm{leaky\_relu}(x) = \begin{cases}
        x, & x \ge 0\\
        \alpha x, & x < 0
        \end{cases}
    """
    negative_slope: float = pytc.field(default=0.01)
    func: ActivationType = pytc.field(default=jax.nn.leaky_relu, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x, negative_slope=stop_gradient(self.negative_slope))


@pytc.treeclass
class ReLU:
    r"""ReLU activation function

    .. math::
        \mathrm{relu}(x) = \max(x, 0)
    """
    func: ActivationType = pytc.field(default=jax.nn.relu, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class ReLU6:
    """ReLU activation function"""

    func: ActivationType = pytc.field(default=jax.nn.relu6, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class SeLU:
    r"""Scaled Exponential Linear Unit

    .. math::
        \mathrm{selu}(x) = \lambda \begin{cases}
        x, & x > 0\\
        \alpha e^x - \alpha, & x \le 0
        \end{cases}

    where :math:`\lambda = 1.0507009873554804934193349852946` and
    :math:`\alpha = 1.6732632423543772848170429916717`.
    """
    func: ActivationType = pytc.field(default=jax.nn.selu, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class SILU:
    """SILU activation function"""

    func: ActivationType = pytc.field(default=silu, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class Sigmoid:
    r"""Sigmoid activation function

    .. math::
        \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}
    """
    func: ActivationType = pytc.field(default=jax.nn.sigmoid, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class SoftPlus:
    r"""SoftPlus activation function

    .. math::
        \mathrm{softplus}(x) = \log(1 + e^x)
    """
    func: ActivationType = pytc.field(default=jax.nn.softplus, callbacks=[jtu.Partial])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class SoftSign:
    """SoftSign activation function"""

    func: ActivationType = pytc.field(default=soft_sign, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class SoftShrink:
    """SoftShrink activation function"""

    alpha: float = pytc.field(default=0.5)
    func: ActivationType = pytc.field(default=soft_shrink, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x, stop_gradient(self.alpha))


@pytc.treeclass
class Stan:
    """Stan activation function"""

    beta: float = pytc.field(callbacks=[scalar_like_cb], default=1.0)
    func: ActivationType = pytc.field(default=self_scalable_tanh, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x, self.beta)


@pytc.treeclass
class SquarePlus:
    """SquarePlus activation function"""

    func: ActivationType = pytc.field(default=square_plus, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class Swish:
    """Swish activation function"""

    func: ActivationType = pytc.field(default=jax.nn.swish, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class Tanh:
    """Tanh activation function"""

    func: ActivationType = pytc.field(default=jax.nn.tanh, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class TanhShrink:
    """TanhShrink activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return x - jax.nn.tanh(x)


@pytc.treeclass
class ThresholdedReLU:
    """Thresholded ReLU activation function."""

    theta: float = pytc.field(callbacks=[*non_negative_scalar_cbs])
    func: ActivationType = pytc.field(default=thresholded_relu, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x, stop_gradient(self.theta))


@pytc.treeclass
class Mish:
    """Mish activation function https://arxiv.org/pdf/1908.08681.pdf."""

    func: ActivationType = pytc.field(default=mish, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x)


@pytc.treeclass
class PReLU:
    """Parametric ReLU activation function"""

    a: float = 0.25
    func: ActivationType = pytc.field(default=parametric_relu, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x, self.a)


@pytc.treeclass
class Snake:
    r"""Snake activation function https://arxiv.org/pdf/2006.08195.pdf."""

    a: float = pytc.field(callbacks=[*non_negative_scalar_cbs], default=1.0)
    func: ActivationType = pytc.field(default=snake, **fkwargs)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.func(x, stop_gradient(self.a))
