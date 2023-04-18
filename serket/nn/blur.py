from __future__ import annotations

# import kernex as kex
import functools as ft

import jax
import jax.numpy as jnp
import pytreeclass as pytc
from jax.lax import stop_gradient

from serket.nn.callbacks import (
    positive_int_cb,
    validate_in_features,
    validate_spatial_in_shape,
)
from serket.nn.convolution import DepthwiseConv2D
from serket.nn.fft_convolution import DepthwiseFFTConv2D


class AvgBlur2D(pytc.TreeClass):
    in_features: int = pytc.field(callbacks=[positive_int_cb])
    kernel_size: int | tuple[int, int] = pytc.field(callbacks=[positive_int_cb])
    conv1: DepthwiseConv2D = pytc.field(repr=False)
    conv2: DepthwiseConv2D = pytc.field(repr=False)

    def __init__(self, in_features: int, kernel_size: int | tuple[int, int]):
        """Average blur 2D layer
        Args:
            in_features: number of input channels
            kernel_size: size of the convolving kernel

        Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.nn.AvgBlur2D(in_features=1, kernel_size=3)
        >>> print(layer(jnp.ones((1,5,5))))
        [[[0.44444448 0.6666667  0.6666667  0.6666667  0.44444448]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.44444448 0.6666667  0.6666667  0.6666667  0.44444448]]]

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
        return stop_gradient(self.conv2(self.conv1(x)))


class GaussianBlur2D(pytc.TreeClass):
    in_features: int = pytc.field(callbacks=[positive_int_cb])
    kernel_size: int = pytc.field(callbacks=[positive_int_cb])
    sigma: float
    conv1: DepthwiseConv2D
    conv2: DepthwiseConv2D

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

        Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.nn.GaussianBlur2D(in_features=1, kernel_size=3)
        >>> print(layer(jnp.ones((1,5,5))))
        [[[0.5269764 0.7259314 0.7259314 0.7259314 0.5269764]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.5269764 0.7259314 0.7259314 0.7259314 0.5269764]]]

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

    def __call__(self, x, **k) -> jax.Array:
        return stop_gradient(self.conv1(self.conv2(x)))


class Filter2D(pytc.TreeClass):
    in_features: int = pytc.field(callbacks=[positive_int_cb])
    conv: DepthwiseConv2D
    kernel: jax.Array

    def __init__(self, in_features: int, kernel: jax.Array):
        """Apply 2D filter for each channel
        Args:
            in_features: number of input channels
            kernel: kernel array

        Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.nn.Filter2D(in_features=1, kernel=jnp.ones((3,3)))
        >>> print(layer(jnp.ones((1,5,5))))
        [[[4. 6. 6. 6. 4.]
          [6. 9. 9. 9. 6.]
          [6. 9. 9. 9. 6.]
          [6. 9. 9. 9. 6.]
          [4. 6. 6. 6. 4.]]]

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
        return stop_gradient(self.conv(x))


class FFTFilter2D(pytc.TreeClass):
    in_features: int = pytc.field(callbacks=[positive_int_cb])
    kernel: jax.Array
    conv: DepthwiseFFTConv2D

    def __init__(self, in_features: int, kernel: jax.Array):
        """Apply 2D filter for each channel using FFT
        Args:
            in_features: number of input channels
            kernel: kernel array

        Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.nn.FFTFilter2D(in_features=1, kernel=jnp.ones((3,3)))
        >>> print(layer(jnp.ones((1,5,5))))
        [[[4.0000005 6.0000005 6.000001  6.0000005 4.0000005]
          [6.0000005 9.        9.        9.        6.0000005]
          [6.0000005 9.        9.        9.        6.0000005]
          [6.0000005 9.        9.        9.        6.0000005]
          [4.        6.0000005 6.0000005 6.0000005 4.       ]]]
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
        return stop_gradient(self.conv(x))
