from __future__ import annotations

import jax.numpy as jnp
import pytreeclass as pytc


@pytc.treeclass
class ThresholdedReLU:
    theta: float = pytc.nondiff_field()

    def __init__(self, theta: float):
        """
        Applies f(x) = x for x > theta and f(x) = 0 otherwise`

        Args:
            theta: threshold value

        See:
            https://arxiv.org/pdf/1402.3337.pdf
            https://keras.io/api/layers/activation_layers/threshold_relu/
        """

        self.theta = theta

        if not isinstance(self.theta, float):
            raise TypeError(
                f"ThresholdedReLU: theta must be a float, not {type(self.theta)}"
            )

        if self.theta < 0:
            raise ValueError(
                f"ThresholdedReLU: theta must be positive, not {self.theta}"
            )

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jnp.where(x > self.theta, x, 0)


@pytc.treeclass
class AdaptiveReLU:
    a: float

    def __init__(self, a: float = 1.0):
        """
        Applies f(x) = a * x for x > 0 and f(x) = 0 otherwise`

        Args:
            a: scaling factor

        See:
            https://arxiv.org/pdf/1906.01170.pdf
        """

        self.a = a

        if not isinstance(self.a, float):
            raise TypeError(f"AdaptiveReLU: a must be a float, not {type(self.a)}")

        if self.a < 0:
            raise ValueError(f"AdaptiveReLU: a must be positive, not {self.a}")

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jnp.maximum(0, self.a * x)


@pytc.treeclass
class AdaptiveLeakyReLU:
    a: float
    v: float = pytc.static_field()

    def __init__(self, a: float = 1.0, v: float = 1.0):
        """
        Applies f(x) = a * x for x > 0 and f(x) = v * x otherwise`

        Args:
            a: scaling factor for positive values
            v: scaling factor for negative values

        See:
            https://arxiv.org/pdf/1906.01170.pdf
        """

        self.a = a
        self.v = v

        if not isinstance(self.a, float):
            raise TypeError(f"AdaptiveLeakyReLU: a must be a float, not {type(self.a)}")

        if not isinstance(self.v, float):
            raise TypeError(f"AdaptiveLeakyReLU: v must be a float, not {type(self.v)}")

        if self.a < 0:
            raise ValueError(f"AdaptiveLeakyReLU: a must be positive, not {self.a}")

        if self.v < 0:
            raise ValueError(f"AdaptiveLeakyReLU: v must be positive, not {self.v}")

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jnp.maximum(0, self.a * x) - self.v * jnp.maximum(0, -self.a * x)


@pytc.treeclass
class AdaptiveSigmoid:
    a: float

    def __init__(self, a: float = 1.0):
        """
        Applies f(x) = 1 / (1 + exp(-a * x))

        Args:
            a: scaling factor

        See:
            https://arxiv.org/pdf/1906.01170.pdf
        """

        self.a = a

        if not isinstance(self.a, float):
            raise TypeError(f"AdaptiveSigmoid: a must be a float, not {type(self.a)}")

        if self.a < 0:
            raise ValueError(f"AdaptiveSigmoid: a must be positive, not {self.a}")

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return 1 / (1 + jnp.exp(-self.a * x))


@pytc.treeclass
class AdaptiveTanh:
    a: float

    def __init__(self, a: float = 1.0):
        """
        Applies f(x) = tanh(a * x)

        Args:
            a: scaling factor

        See:
            https://arxiv.org/pdf/1906.01170.pdf
        """

        self.a = a

        if not isinstance(self.a, float):
            raise TypeError(f"AdaptiveTanh: a must be a float, not {type(self.a)}")

        if self.a < 0:
            raise ValueError(f"AdaptiveTanh: a must be positive, not {self.a}")

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return (jnp.exp(self.a * x) - jnp.exp(-self.a * x)) / (
            jnp.exp(self.a * x) + jnp.exp(-self.a * x)
        )
