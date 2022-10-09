from __future__ import annotations

import jax
import jax.numpy as jnp
import pytreeclass as pytc

from .convolution import DepthwiseConv2D, _check_and_return_kernel


@pytc.treeclass
class AvgBlur2D:
    def __init__(self, in_features: int, kernel_size: int | tuple[int, int]):
        kernel_size = _check_and_return_kernel(kernel_size, 2)

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

    def __init__(
        self,
        in_features: int,
        kernel_size: int,
        *,
        sigma: int = 1.0,
        separable: bool = False,
    ):
        """Apply Gaussian blur to a channel-first image.

        Args:
            in_features (int): number of input features
            kernel_size (int): kernel size
            sigma (int, optional): sigma. Defaults to 1.
            separable (bool, optional): use separable convolution. Defaults to False.

        """
        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError(
                f"Expected `in_features` to be a positive integer, got {in_features}"
            )
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError(
                f"Expected `kernel_size` to be a positive integer, got {kernel_size}"
            )

        self.in_features = in_features
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.separable = separable

        x = jnp.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size)
        w = jnp.exp(-0.5 * jnp.square(x) * jax.lax.rsqrt(self.sigma))

        if separable is False:
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

        else:
            w = w / jnp.sum(w)
            w = w[:, None]
            w = jnp.repeat(w[None, None], in_features, axis=0)

            conv1 = DepthwiseConv2D(
                in_features=in_features,
                kernel_size=(kernel_size, 1),
                padding="same",
                bias_init_func=None,
            )

            conv2 = DepthwiseConv2D(
                in_features=in_features,
                kernel_size=(1, kernel_size),
                padding="same",
                bias_init_func=None,
            )

            self.conv1 = conv1.at["weight"].set(w)
            self.conv2 = conv2.at["weight"].set(jnp.moveaxis(w, 2, 3))

    def __call__(self, x, **kwargs) -> jnp.ndarray:
        assert x.ndim == 3, "`Input` must be 3D."

        if self.separable is True:
            return self.conv2(self.conv1(x))

        return self.conv(x)
