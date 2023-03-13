from __future__ import annotations

# import kernex as kex
import functools as ft

import jax
import jax.numpy as jnp
import pytreeclass as pytc

from serket.nn.callbacks import (
    frozen_positive_int_cbs,
    validate_in_features,
    validate_spatial_in_shape,
)
from serket.nn.convolution import DepthwiseConv2D
from serket.nn.fft_convolution import DepthwiseFFTConv2D
from serket.nn.lazy_class import lazy_class

_infer_func = lambda self, *a, **k: (a[0].shape[0],)
_lazy_keywords = ["in_features"]


@ft.partial(lazy_class, lazy_keywords=_lazy_keywords, infer_func=_infer_func)
@pytc.treeclass
class AvgBlur2D:
    in_features: int = pytc.field(callbacks=[*frozen_positive_int_cbs])
    kernel_size: int | tuple[int, int] = pytc.field(
        callbacks=[*frozen_positive_int_cbs]
    )
    conv1: DepthwiseConv2D = pytc.field(callbacks=[pytc.freeze])
    conv2: DepthwiseConv2D = pytc.field(callbacks=[pytc.freeze])

    def __init__(self, in_features: int, kernel_size: int | tuple[int, int]):
        """Average blur 2D layer
        Args:
            in_features: number of input channels
            kernel_size: size of the convolving kernel
        """
        self.in_features = in_features
        self.kernel_size = kernel_size

        w = jnp.ones(kernel_size)
        w = w / jnp.sum(w)
        w = w[:, None]
        w = jnp.repeat(w[None, None], in_features, axis=0)

        self.spatial_ndim = 2
        self.in_features = in_features
        self.conv1 = DepthwiseConv2D(
            in_features=in_features,
            kernel_size=(kernel_size, 1),
            padding="same",
            bias_init_func=None,
        )

        self.conv2 = DepthwiseConv2D(
            in_features=in_features,
            kernel_size=(1, kernel_size),
            padding="same",
            bias_init_func=None,
        )

        self.conv1 = self.conv1.at["weight"].set(w)
        self.conv2 = self.conv2.at["weight"].set(jnp.moveaxis(w, 2, 3))  # transpose

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    @ft.partial(validate_in_features, attribute_name="in_features")
    def __call__(self, x, **k) -> jax.Array:
        return self.conv2(self.conv1(x))


@ft.partial(lazy_class, lazy_keywords=["in_features"], infer_func=_infer_func)
@pytc.treeclass
class GaussianBlur2D:
    in_features: int = pytc.field(callbacks=[*frozen_positive_int_cbs])
    kernel_size: int = pytc.field(callbacks=[*frozen_positive_int_cbs])
    sigma: float = pytc.field(callbacks=[pytc.freeze])
    conv1: DepthwiseConv2D = pytc.field(callbacks=[pytc.freeze])
    conv2: DepthwiseConv2D = pytc.field(callbacks=[pytc.freeze])

    def __init__(
        self,
        in_features,
        kernel_size,
        *,
        sigma=1.0,
    ):
        """Apply Gaussian blur to a channel-first image.

        Args:
            in_features: number of input features
            kernel_size: kernel size
            sigma: sigma. Defaults to 1.
        """

        self.in_features = in_features
        self.kernel_size = kernel_size

        self.in_features = in_features
        self.kernel_size = kernel_size
        self.sigma = sigma

        x = jnp.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size)
        w = jnp.exp(-0.5 * jnp.square(x) * jax.lax.rsqrt(self.sigma))

        w = w / jnp.sum(w)
        w = w[:, None]

        w = jnp.repeat(w[None, None], in_features, axis=0)
        self.conv1 = DepthwiseConv2D(
            in_features=in_features,
            kernel_size=(kernel_size, 1),
            padding="same",
            bias_init_func=None,
        )

        self.conv2 = DepthwiseConv2D(
            in_features=in_features,
            kernel_size=(1, kernel_size),
            padding="same",
            bias_init_func=None,
        )

        self.in_features = in_features
        self.spatial_ndim = 2

        self.conv1 = self.conv1.at["weight"].set(w)
        self.conv2 = self.conv2.at["weight"].set(jnp.moveaxis(w, 2, 3))

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    @ft.partial(validate_in_features, attribute_name="in_features")
    def __call__(self, x, **k) -> jax.Array:
        return self.conv1(self.conv2(x))


@ft.partial(lazy_class, lazy_keywords=["in_features"], infer_func=_infer_func)
@pytc.treeclass
class Filter2D:
    in_features: int = pytc.field(callbacks=[*frozen_positive_int_cbs])
    conv: DepthwiseConv2D = pytc.field(callbacks=[pytc.freeze], repr=False)
    kernel: jax.Array = pytc.field(callbacks=[pytc.freeze])

    def __init__(self, in_features: int, kernel: jax.Array):
        """Apply 2D filter for each channel
        Args:
            in_features: number of input channels
            kernel: kernel array
        """
        if not isinstance(kernel, jax.Array) or kernel.ndim != 2:
            raise ValueError("Expected `kernel` to be a 2D `ndarray` with shape (H, W)")

        self.in_features = in_features
        self.spatial_ndim = 2
        self.kernel = jnp.stack([kernel] * in_features, axis=0)
        self.kernel = self.kernel[:, None]

        self.conv = DepthwiseConv2D(
            in_features=in_features,
            kernel_size=kernel.shape,
            padding="same",
            bias_init_func=None,
        )
        self.conv = self.conv.at["weight"].set(self.kernel)

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    @ft.partial(validate_in_features, attribute_name="in_features")
    def __call__(self, x, **k) -> jax.Array:
        return self.conv(x)


@ft.partial(lazy_class, lazy_keywords=["in_features"], infer_func=_infer_func)
@pytc.treeclass
class FFTFilter2D:
    in_features: int = pytc.field(callbacks=[*frozen_positive_int_cbs])
    kernel: jax.Array = pytc.field(callbacks=[pytc.freeze])
    conv: DepthwiseFFTConv2D = pytc.field(callbacks=[pytc.freeze])

    def __init__(self, in_features: int, kernel: jax.Array):
        """Apply 2D filter for each channel using FFT , faster for large kernels.

        Args:
            in_features: number of input channels
            kernel: kernel array
        """
        if not isinstance(kernel, jax.Array) or kernel.ndim != 2:
            raise ValueError("Expected `kernel` to be a 2D `ndarray` with shape (H, W)")

        self.in_features = in_features
        self.spatial_ndim = 2
        self.kernel = jnp.stack([kernel] * in_features, axis=0)
        self.kernel = self.kernel[:, None]

        self.conv = DepthwiseFFTConv2D(
            in_features=in_features,
            kernel_size=kernel.shape,
            padding="same",
            bias_init_func=None,
        )
        self.conv = self.conv.at["weight"].set(self.kernel)

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    @ft.partial(validate_in_features, attribute_name="in_features")
    def __call__(self, x, **k) -> jax.Array:
        return self.conv(x)
