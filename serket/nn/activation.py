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
