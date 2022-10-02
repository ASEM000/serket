from __future__ import annotations

import jax
import jax.numpy as jnp
import pytreeclass as pytc

from .convolution import DepthwiseConv2D


@pytc.treeclass
class AvgBlur2D:
    def __init__(self, in_features: int, kernel_size: int | tuple[int, int]):
        kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )

        # vectorize on channels dimension
        w = jnp.ones([*kernel_size]) / jnp.array(kernel_size).prod()
        w = jnp.repeat(w[None, None], in_features, axis=0)

        self.conv = DepthwiseConv2D(
            in_features=in_features,
            kernel_size=kernel_size,
            padding="same",
            bias_init_func=None,
        )
        self.conv = self.conv.at["weight"].set(w)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 3, "`Input` must be 3D."
        return self.conv(x)


@pytc.treeclass
class GaussianBlur2D:
    in_features: int = pytc.nondiff_field()
    kernel_size: int = pytc.nondiff_field()
    sigma: float = pytc.nondiff_field()

    def __init__(self, in_features: int, kernel_size: int, *, sigma: int = 1.0):
        """Apply Gaussian blur to a channel-first image.

        Args:
            in_features (int): number of input features
            kernel_size (int): kernel size
            sigma (int, optional): sigma. Defaults to 1.

        """
        # type assertions
        assert isinstance(
            in_features, int
        ), f"Expected int for `in_features`, got {in_features}."

        # assert proper values
        assert in_features > 0, "`in_features` must be greater than 0."

        self.in_features = in_features
        self.kernel_size = kernel_size
        self.sigma = sigma

        x = jnp.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size)
        w = jnp.exp(-0.5 * jnp.square(x) * jax.lax.rsqrt(self.sigma))
        w = jnp.outer(w, w)
        w = w / jnp.sum(w)
        w = jnp.repeat(w[None, None], in_features, axis=0)

        self.conv = DepthwiseConv2D(
            in_features=in_features,
            kernel_size=kernel_size,
            padding="same",
            bias_init_func=None,
        )
        self.conv = self.conv.at["weight"].set(w)

    def __call__(self, x, **kwargs):
        assert x.ndim == 3, "`Input` must be 3D."
        return self.conv(x)
