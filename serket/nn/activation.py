from __future__ import annotations

import jax
import jax.numpy as jnp
import pytreeclass as pytc

from serket.nn.callbacks import non_negative_scalar_cb

__all__ = (
    "AdaptiveLeakyReLU",
    "AdaptiveReLU",
    "AdaptiveSigmoid",
    "AdaptiveTanh",
    "CeLU",
    "ELU",
    "GELU",
    "GLU",
    "HardSILU",
    "HardShrink",
    "HardSigmoid",
    "HardSwish",
    "HardTanh",
    "LeakyReLU",
    "LogSigmoid",
    "LogSoftmax",
    "Mish",
    "PReLU",
    "ReLU",
    "ReLU6",
    "SILU",
    "SeLU",
    "Sigmoid",
    "SoftPlus",
    "SoftShrink",
    "SoftSign",
    "Snake",
    "Swish",
    "Tanh",
    "TanhShrink",
    "ThresholdedReLU",
)


def adaptive_leaky_relu(x: jnp.ndarray, a: float = 1.0, v: float = 1.0) -> jnp.ndarray:
    return jnp.maximum(0, a * x) - v * jnp.maximum(0, -a * x)


def adaptive_relu(x: jnp.ndarray, a: float = 1.0) -> jnp.ndarray:
    return jnp.maximum(0, a * x)


def adaptive_sigmoid(x: jnp.ndarray, a: float = 1.0) -> jnp.ndarray:
    return 1 / (1 + jnp.exp(-a * x))


def adaptive_tanh(x: jnp.ndarray, a: float = 1.0) -> jnp.ndarray:
    return (jnp.exp(a * x) - jnp.exp(-a * x)) / (jnp.exp(a * x) + jnp.exp(-a * x))


def hard_shrink(x: jnp.ndarray, alpha: float = 0.5) -> jnp.ndarray:
    return jnp.where(x > alpha, x, jnp.where(x < -alpha, x, 0.0))


def parametric_relu(x: jnp.ndarray, a: float = 0.25) -> jnp.ndarray:
    return jnp.where(x >= 0, x, x * a)


def soft_shrink(x: jnp.ndarray, alpha: float = 0.5) -> jnp.ndarray:
    return jnp.where(
        x < -alpha,
        x + alpha,
        jnp.where(x > alpha, x - alpha, 0.0),
    )


def soft_sign(x: jnp.ndarray) -> jnp.ndarray:
    return x / (1 + jnp.abs(x))


def thresholded_relu(x: jnp.ndarray, theta: float = 1.0) -> jnp.ndarray:
    return jnp.where(x > theta, x, 0)


def mish(x: jnp.ndarray) -> jnp.ndarray:
    return x * jax.nn.tanh(jax.nn.softplus(x))


def snake(x: jnp.ndarray, frequency: float = 1.0) -> jnp.ndarray:
    return x + (1 - jnp.cos(2 * frequency * x)) / (2 * frequency)


@pytc.treeclass
class AdaptiveLeakyReLU:
    """Leaky ReLU activation function with learnable parameters https://arxiv.org/pdf/1906.01170.pdf"""

    a: float = pytc.field(default=1.0, callbacks=[non_negative_scalar_cb])
    v: float = pytc.field(default=1.0, callbacks=[non_negative_scalar_cb, pytc.freeze])

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return adaptive_leaky_relu(x, self.a, self.v)


@pytc.treeclass
class AdaptiveReLU:
    """ReLU activation function with learnable parameters https://arxiv.org/pdf/1906.01170.pdf"""

    a: float = pytc.field(default=1.0, callbacks=[non_negative_scalar_cb])

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return adaptive_relu(x, self.a)


@pytc.treeclass
class AdaptiveSigmoid:
    """Sigmoid activation function with learnable parameters https://arxiv.org/pdf/1906.01170.pdf"""

    a: float = pytc.field(default=1.0, callbacks=[non_negative_scalar_cb])

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return adaptive_sigmoid(x, self.a)


@pytc.treeclass
class AdaptiveTanh:
    """Tanh activation function with learnable parameters https://arxiv.org/pdf/1906.01170.pdf"""

    a: float = pytc.field(default=1.0, callbacks=[non_negative_scalar_cb])

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return adaptive_tanh(x, self.a)


@pytc.treeclass
class CeLU:
    """Celu activation function"""

    alpha: float = pytc.field(callbacks=[pytc.freeze], default=1.0)

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.celu(x, alpha=self.alpha)


@pytc.treeclass
class ELU:
    """Exponential linear unit"""

    alpha: float = pytc.field(callbacks=[pytc.freeze], default=1.0)

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.elu(x, alpha=self.alpha)


@pytc.treeclass
class GELU:
    approximate: bool = pytc.field(callbacks=[pytc.freeze], default=True)
    """Gaussian error linear unit"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.gelu(x, approximate=self.approximate)


@pytc.treeclass
class GLU:
    """Gated linear unit"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.glu(x)


@pytc.treeclass
class HardSILU:
    """Hard SILU activation function"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.hard_silu(x)


@pytc.treeclass
class HardShrink:
    """Hard shrink activation function"""

    alpha: float = pytc.field(callbacks=[pytc.freeze], default=0.5)

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return hard_shrink(x, self.alpha)


@pytc.treeclass
class HardSigmoid:
    """Hard sigmoid activation function"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.hard_sigmoid(x)


@pytc.treeclass
class HardSwish:
    """Hard swish activation function"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.hard_swish(x)


@pytc.treeclass
class HardTanh:
    """Hard tanh activation function"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.hard_tanh(x)


@pytc.treeclass
class LogSigmoid:
    """Log sigmoid activation function"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.log_sigmoid(x)


@pytc.treeclass
class LogSoftmax:
    """Log softmax activation function"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.log_softmax(x)


@pytc.treeclass
class LeakyReLU:
    """Leaky ReLU activation function"""

    negative_slope: float = pytc.field(callbacks=[pytc.freeze], default=0.01)

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.leaky_relu(x, negative_slope=self.negative_slope)


@pytc.treeclass
class ReLU:
    """ReLU activation function"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.relu(x)


@pytc.treeclass
class ReLU6:
    """ReLU activation function"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.relu6(x)


@pytc.treeclass
class SeLU:
    """Scaled Exponential Linear Unit"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.selu(x)


@pytc.treeclass
class SILU:
    """SILU activation function"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return x * jax.nn.sigmoid(x)


@pytc.treeclass
class Sigmoid:
    """Sigmoid activation function"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.sigmoid(x)


@pytc.treeclass
class SoftPlus:
    """SoftPlus activation function"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.softplus(x)


@pytc.treeclass
class SoftSign:
    """SoftSign activation function"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return soft_sign(x)


@pytc.treeclass
class SoftShrink:
    """SoftShrink activation function"""

    alpha: float = pytc.field(callbacks=[pytc.freeze], default=0.5)

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return soft_shrink(x, self.alpha)


@pytc.treeclass
class Swish:
    """Swish activation function"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.swish(x)


@pytc.treeclass
class Tanh:
    """Tanh activation function"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return jax.nn.tanh(x)


@pytc.treeclass
class TanhShrink:
    """TanhShrink activation function"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return x - jax.nn.tanh(x)


@pytc.treeclass
class ThresholdedReLU:
    """Thresholded ReLU activation function"""

    theta: float = pytc.field(callbacks=[non_negative_scalar_cb, pytc.freeze])

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return thresholded_relu(x, self.theta)


@pytc.treeclass
class Mish:
    """Mish activation function https://arxiv.org/pdf/1908.08681.pdf"""

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return mish(x)


@pytc.treeclass
class PReLU:
    """Parametric ReLU activation function"""

    a: float = 0.25

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return parametric_relu(x, self.a)


@pytc.treeclass
class Snake:
    """Snake activation function https://arxiv.org/pdf/2006.08195.pdf"""

    a: float = pytc.field(callbacks=[non_negative_scalar_cb, pytc.freeze], default=1.0)

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        return snake(x, self.a)
