from __future__ import annotations

import jax.numpy as jnp
import pytreeclass as pytc


@pytc.treeclass
class ThresholdedReLU:
    theta: float = pytc.nondiff_field()

    def __post_init__(self):
        """Applies f(x) = x for x > theta and f(x) = 0 otherwise`

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
class AdaptiveReLU:
    a: float = 1.0

    def __post_init__(self):
        """Applies f(x) = a * x for x > 0 and f(x) = 0 otherwise`

        Args:
            a: scaling factor

        See:
            https://arxiv.org/pdf/1906.01170.pdf
        """
        if not isinstance(self.a, float) or self.a < 0:
            raise ValueError(f"`a` must be a positive float, got {self.a}")

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jnp.maximum(0, self.a * x)


@pytc.treeclass
class AdaptiveLeakyReLU:
    a: float = 1.0
    v: float = pytc.nondiff_field(default=1.0)

    def __post_init__(self, a: float = 1.0, v: float = 1.0):
        """Applies f(x) = a * x for x > 0 and f(x) = v * x otherwise`

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
class AdaptiveSigmoid:
    a: float = 1.0

    def __post_init__(self):
        """Applies f(x) = 1 / (1 + exp(-a * x))

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
        """Applies f(x) = tanh(a * x)

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
