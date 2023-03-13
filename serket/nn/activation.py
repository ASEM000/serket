from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import pytreeclass as pytc

from serket.nn.callbacks import non_negative_scalar_cbs


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


def soft_shrink(x: jax.Array, alpha: float = 0.5) -> jax.Array:
    return jnp.where(
        x < -alpha,
        x + alpha,
        jnp.where(x > alpha, x - alpha, 0.0),
    )


def soft_sign(x: jax.Array) -> jax.Array:
    return x / (1 + jnp.abs(x))


def thresholded_relu(x: jax.Array, theta: float = 1.0) -> jax.Array:
    return jnp.where(x > theta, x, 0)


def mish(x: jax.Array) -> jax.Array:
    return x * jax.nn.tanh(jax.nn.softplus(x))


def snake(x: jax.Array, frequency: float = 1.0) -> jax.Array:
    return x + (1 - jnp.cos(2 * frequency * x)) / (2 * frequency)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class AdaptiveLeakyReLU:
    """Leaky ReLU activation function with learnable parameters https://arxiv.org/pdf/1906.01170.pdf"""

    a: float = pytc.field(default=1.0, callbacks=[*non_negative_scalar_cbs])
    v: float = pytc.field(
        default=1.0, callbacks=[*non_negative_scalar_cbs, pytc.freeze]
    )

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return adaptive_leaky_relu(x, self.a, self.v)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class AdaptiveReLU:
    """ReLU activation function with learnable parameters https://arxiv.org/pdf/1906.01170.pdf"""

    a: float = pytc.field(default=1.0, callbacks=[*non_negative_scalar_cbs])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return adaptive_relu(x, self.a)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class AdaptiveSigmoid:
    """Sigmoid activation function with learnable parameters https://arxiv.org/pdf/1906.01170.pdf"""

    a: float = pytc.field(default=1.0, callbacks=[*non_negative_scalar_cbs])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return adaptive_sigmoid(x, self.a)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class AdaptiveTanh:
    """Tanh activation function with learnable parameters https://arxiv.org/pdf/1906.01170.pdf"""

    a: float = pytc.field(default=1.0, callbacks=[*non_negative_scalar_cbs])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return adaptive_tanh(x, self.a)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class CeLU:
    """Celu activation function"""

    alpha: float = pytc.field(callbacks=[pytc.freeze], default=1.0)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.celu(x, alpha=self.alpha)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class ELU:
    """Exponential linear unit"""

    alpha: float = pytc.field(callbacks=[pytc.freeze], default=1.0)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.elu(x, alpha=self.alpha)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class GELU:
    approximate: bool = pytc.field(callbacks=[pytc.freeze], default=True)
    """Gaussian error linear unit"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.gelu(x, approximate=self.approximate)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class GLU:
    """Gated linear unit"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.glu(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class HardSILU:
    """Hard SILU activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.hard_silu(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class HardShrink:
    """Hard shrink activation function"""

    alpha: float = pytc.field(callbacks=[pytc.freeze], default=0.5)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return hard_shrink(x, self.alpha)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class HardSigmoid:
    """Hard sigmoid activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.hard_sigmoid(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class HardSwish:
    """Hard swish activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.hard_swish(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class HardTanh:
    """Hard tanh activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.hard_tanh(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class LogSigmoid:
    """Log sigmoid activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.log_sigmoid(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class LogSoftmax:
    """Log softmax activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.log_softmax(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class LeakyReLU:
    """Leaky ReLU activation function"""

    negative_slope: float = pytc.field(callbacks=[pytc.freeze], default=0.01)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.leaky_relu(x, negative_slope=self.negative_slope)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class ReLU:
    """ReLU activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.relu(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class ReLU6:
    """ReLU activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.relu6(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class SeLU:
    """Scaled Exponential Linear Unit"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.selu(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class SILU:
    """SILU activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return x * jax.nn.sigmoid(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class Sigmoid:
    """Sigmoid activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.sigmoid(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class SoftPlus:
    """SoftPlus activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.softplus(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class SoftSign:
    """SoftSign activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return soft_sign(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class SoftShrink:
    """SoftShrink activation function"""

    alpha: float = pytc.field(callbacks=[pytc.freeze], default=0.5)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return soft_shrink(x, self.alpha)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class Swish:
    """Swish activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.swish(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class Tanh:
    """Tanh activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.nn.tanh(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class TanhShrink:
    """TanhShrink activation function"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return x - jax.nn.tanh(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class ThresholdedReLU:
    """Thresholded ReLU activation function"""

    theta: float = pytc.field(callbacks=[*non_negative_scalar_cbs, pytc.freeze])

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return thresholded_relu(x, self.theta)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class Mish:
    """Mish activation function https://arxiv.org/pdf/1908.08681.pdf"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return mish(x)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class PReLU:
    """Parametric ReLU activation function"""

    a: float = 0.25

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return parametric_relu(x, self.a)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class Snake:
    """Snake activation function https://arxiv.org/pdf/2006.08195.pdf"""

    a: float = pytc.field(
        callbacks=[*non_negative_scalar_cbs, pytc.freeze], default=1.0
    )

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return snake(x, self.a)
