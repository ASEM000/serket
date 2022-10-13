from __future__ import annotations

import jax
import jax.numpy as jnp

# import kernex as kex
import pytreeclass as pytc

from .convolution import DepthwiseConv2D


@pytc.treeclass
class AvgBlur2D:
    def __init__(self, in_features: int, kernel_size: int | tuple[int, int]):
        """Average blur 2D layer
        Args:
            in_features: number of input channels
            kernel_size: size of the convolving kernel
        """
        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError(
                f"Expected `in_features` to be a positive integer, got {in_features}"
            )
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError(
                f"Expected `kernel_size` to be a positive integer, got {kernel_size}"
            )

        w = jnp.ones(kernel_size)
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

        conv1 = conv1.at["weight"].set(w)
        conv2 = conv2.at["weight"].set(jnp.moveaxis(w, 2, 3))  # transpose
        self._func = lambda x: conv2(conv1(x))

    def __call__(self, x, **kwargs) -> jnp.ndarray:
        assert x.ndim == 3, "`Input` must be 3D."
        return self._func(x)


@pytc.treeclass
class GaussianBlur2D:
    in_features: int = pytc.nondiff_field()
    kernel_size: int = pytc.nondiff_field()
    sigma: float = pytc.nondiff_field()

    def __init__(
        self,
        in_features,
        kernel_size,
        *,
        sigma=1.0,
        # implementation="jax",
    ):
        """Apply Gaussian blur to a channel-first image.

        Args:
            in_features: number of input features
            kernel_size: kernel size
            sigma: sigma. Defaults to 1.
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

        x = jnp.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size)
        w = jnp.exp(-0.5 * jnp.square(x) * jax.lax.rsqrt(self.sigma))

        w = w / jnp.sum(w)
        w = w[:, None]

        # if implementation == "jax":
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

        conv1 = conv1.at["weight"].set(w)
        conv2 = conv2.at["weight"].set(jnp.moveaxis(w, 2, 3))
        self._func = lambda x: conv2(conv1(x))

        # elif implementation == "kernex":
        #     # usually faster than jax for small kernel sizes
        #     # but slower for large kernel sizes

        #     @jax.vmap  # channel
        #     @kex.kmap(kernel_size=(kernel_size, 1), padding="same")
        #     def conv1(x):
        #         return jnp.sum(x * w)

        #     @jax.vmap
        #     @kex.kmap(kernel_size=(1, kernel_size), padding="same")
        #     def conv2(x):
        #         return jnp.sum(x * w.T)

        #     self._func = lambda x: conv2(conv1(x))

        # else:
        #     raise ValueError(f"Unknown implementation {implementation}")

    def __call__(self, x, **kwargs) -> jnp.ndarray:
        assert x.ndim == 3, "`Input` must be 3D."
        return self._func(x)
