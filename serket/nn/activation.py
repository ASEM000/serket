from __future__ import annotations

import jax
import jax.numpy as jnp
import pytreeclass as pytc

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


@pytc.treeclass
class AdaptiveLeakyReLU:
    a: float = 1.0
    v: float = pytc.nondiff_field(default=1.0)

    def __post_init__(self, a: float = 1.0, v: float = 1.0):
        """
        Args:
            a: scaling factor for positive values
            v: scaling factor for negative values

        See:
            https://arxiv.org/pdf/1906.01170.pdf
        """
        if not isinstance(self.a, float) or self.a < 0:
            raise ValueError(f"`a` must be a positive float, got {self.a}")

        if not isinstance(self.v, float) or self.v < 0:
            raise ValueError(f"`v` must be a positive float, got {self.v}")

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jnp.maximum(0, self.a * x) - self.v * jnp.maximum(0, -self.a * x)


@pytc.treeclass
class AdaptiveReLU:
    a: float = 1.0

    def __post_init__(self):
        """
        Args:
            a: scaling factor
        See:
            https://arxiv.org/pdf/1906.01170.pdf"""
        if not isinstance(self.a, float) or self.a < 0:
            raise ValueError(f"`a` must be a positive float, got {self.a}")

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jnp.maximum(0, self.a * x)


@pytc.treeclass
class AdaptiveSigmoid:
    a: float = 1.0

    def __post_init__(self):
        """
        Args:
            a: scaling factor

        See:
            https://arxiv.org/pdf/1906.01170.pdf
        """
        if not isinstance(self.a, float):
            raise TypeError(f"AdaptiveSigmoid: a must be a float, not {type(self.a)}")

        if self.a < 0:
            raise ValueError(f"AdaptiveSigmoid: a must be positive, not {self.a}")

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return 1 / (1 + jnp.exp(-self.a * x))


@pytc.treeclass
class AdaptiveTanh:
    a: float = 1.0

    def __post_init__(self):
        """
        Args:
            a: scaling factor

        See:
            https://arxiv.org/pdf/1906.01170.pdf
        """
        if not isinstance(self.a, float) or self.a < 0:
            raise ValueError(f"`a` must be a positive float, got {self.a}")

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return (jnp.exp(self.a * x) - jnp.exp(-self.a * x)) / (
            jnp.exp(self.a * x) + jnp.exp(-self.a * x)
        )


@pytc.treeclass
class CeLU:
    """Celu activation function"""

    alpha: float = pytc.nondiff_field(default=1.0)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.celu(x, alpha=self.alpha)


@pytc.treeclass
class ELU:
    """Exponential linear unit"""

    alpha: float = pytc.nondiff_field(default=1.0)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.elu(x, alpha=self.alpha)


@pytc.treeclass
class GELU:
    approximate: bool = pytc.nondiff_field(default=True)
    """Gaussian error linear unit"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.gelu(x, approximate=self.approximate)


@pytc.treeclass
class GLU:
    """Gated linear unit"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.glu(x)


@pytc.treeclass
class HardSILU:
    """Hard SILU activation function"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.hard_silu(x)


@pytc.treeclass
class HardShrink:
    """Hard shrink activation function"""

    alpha: float = pytc.nondiff_field(default=0.5)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jnp.where(x > self.alpha, x, jnp.where(x < -self.alpha, x, 0.0))


@pytc.treeclass
class HardSigmoid:
    """Hard sigmoid activation function"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.hard_sigmoid(x)


@pytc.treeclass
class HardSwish:
    """Hard swish activation function"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.hard_swish(x)


@pytc.treeclass
class HardTanh:
    """Hard tanh activation function"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.hard_tanh(x)


@pytc.treeclass
class LogSigmoid:
    """Log sigmoid activation function"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.log_sigmoid(x)


@pytc.treeclass
class LogSoftmax:
    """Log softmax activation function"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.log_softmax(x)


@pytc.treeclass
class LeakyReLU:
    """Leaky ReLU activation function"""

    negative_slope: float = pytc.nondiff_field(default=0.01)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.leaky_relu(x, negative_slope=self.negative_slope)


@pytc.treeclass
class ReLU:
    """ReLU activation function"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.relu(x)


@pytc.treeclass
class ReLU6:
    """ReLU activation function"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.relu6(x)


@pytc.treeclass
class SeLU:
    """Scaled Exponential Linear Unit"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.selu(x)


@pytc.treeclass
class SILU:
    """SILU activation function"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return x * jax.nn.sigmoid(x)


@pytc.treeclass
class Sigmoid:
    """Sigmoid activation function"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.sigmoid(x)


@pytc.treeclass
class SoftPlus:
    """SoftPlus activation function"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.softplus(x)


@pytc.treeclass
class SoftSign:
    """SoftSign activation function"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return x / (1 + jnp.abs(x))


@pytc.treeclass
class SoftShrink:
    """SoftShrink activation function"""

    alpha: float = pytc.nondiff_field(default=0.5)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jnp.where(
            x < -self.alpha,
            x + self.alpha,
            jnp.where(x > self.alpha, x - self.alpha, 0.0),
        )


@pytc.treeclass
class Swish:
    """Swish activation function"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.swish(x)


@pytc.treeclass
class Tanh:
    """Tanh activation function"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jax.nn.tanh(x)


@pytc.treeclass
class TanhShrink:
    """TanhShrink activation function"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return x - jax.nn.tanh(x)


@pytc.treeclass
class ThresholdedReLU:
    theta: float = pytc.nondiff_field()

    def __post_init__(self):
        """
        Args:
            theta: threshold value

        See:
            https://arxiv.org/pdf/1402.3337.pdf
            https://keras.io/api/layers/activation_layers/threshold_relu/
        """

        if not isinstance(self.theta, float) or self.theta < 0:
            raise ValueError(f"`theta` must be a positive float, got {self.theta}")

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jnp.where(x > self.theta, x, 0)


@pytc.treeclass
class Mish:
    """Mish activation function"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return x * jax.nn.tanh(jax.nn.softplus(x))


@pytc.treeclass
class PReLU:
    """Parametric ReLU activation function"""

    a: float = 0.25

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jnp.where(x > 0, x, x * self.a)


@pytc.treeclass
class Snake:
    """Snake activation function
    See: https://arxiv.org/pdf/2006.08195.pdf
    """

    frequency: float = 1.0

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return x + (1 - jnp.cos(2 * self.frequency * x)) / (2 * self.frequency)
