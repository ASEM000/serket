from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp

# import kernex as kex
import pytreeclass as pytc

from serket.nn.convolution import DepthwiseConv2D
from serket.nn.fft_convolution import DepthwiseFFTConv2D
from serket.nn.utils import (
    _check_and_return_positive_int,
    _check_in_features,
    _check_non_tracer,
    _check_spatial_in_shape,
    _instance_cb,
    _range_cb,
)

_frozen_positive_int_cb = [_range_cb(1), _instance_cb(int), pytc.freeze]


@pytc.treeclass
class AvgBlur2D:
    in_features: int = pytc.field(callbacks=[pytc.freeze])
    kernel_size: int | tuple[int, int] = pytc.field(callbacks=[pytc.freeze])

    def __init__(self, in_features: int, kernel_size: int | tuple[int, int]):
        """Average blur 2D layer
        Args:
            in_features: number of input channels
            kernel_size: size of the convolving kernel
        """
        if in_features is None:
            for field_item in pytc.fields(self):
                setattr(self, field_item.name, None)

            self._init = ft.partial(
                AvgBlur2D.__init__,
                self,
                kernel_size=kernel_size,
            )
            return

        if hasattr(self, "_init"):
            delattr(self, "_init")

        self.in_features = _check_and_return_positive_int(in_features, "in_features")
        self.kernel_size = _check_and_return_positive_int(kernel_size, "kernel_size")

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

    def __call__(self, x, **k) -> jnp.ndarray:
        if hasattr(self, "_init"):
            _check_non_tracer(x, self.__class__.__name__)
            getattr(self, "_init")(in_features=x.shape[0])
        _check_spatial_in_shape(x, self.spatial_ndim)
        _check_in_features(x, self.in_features)
        return self.conv2(self.conv1(x))


@pytc.treeclass
class GaussianBlur2D:
    in_features: int = pytc.field(callbacks=[pytc.freeze])
    kernel_size: int = pytc.field(callbacks=[pytc.freeze])
    sigma: float = pytc.field(callbacks=[pytc.freeze])

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
        if in_features is None:
            for field_item in pytc.fields(self):
                setattr(self, field_item.name, None)

            self._init = ft.partial(
                GaussianBlur2D.__init__,
                self=self,
                kernel_size=kernel_size,
                sigma=sigma,
            )
            return

        if hasattr(self, "_init"):
            delattr(self, "_init")

        self.in_features = _check_and_return_positive_int(in_features, "in_features")
        self.kernel_size = _check_and_return_positive_int(kernel_size, "kernel_size")

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

    def __call__(self, x, **k) -> jnp.ndarray:
        if hasattr(self, "_init"):
            _check_non_tracer(x, self.__class__.__name__)
            getattr(self, "_init")(in_features=x.shape[0])
        _check_spatial_in_shape(x, self.spatial_ndim)
        _check_in_features(x, self.in_features)

        return self.conv1(self.conv2(x))


@pytc.treeclass
class Filter2D:
    in_features: int = pytc.field(callbacks=[pytc.freeze])
    conv: DepthwiseConv2D = pytc.field(callbacks=[pytc.freeze], repr=False)
    kernel: jnp.ndarray = pytc.field(callbacks=[pytc.freeze])

    def __init__(self, in_features: int, kernel: jnp.ndarray):
        """Apply 2D filter for each channel
        Args:
            in_features: number of input channels
            kernel: kernel array
        """
        if in_features is None:
            for field_item in pytc.fields(self):
                setattr(self, field_item.name, None)

            self._init = ft.partial(Filter2D.__init__, self=self, kernel=kernel)
            return

        if hasattr(self, "_init"):
            delattr(self, "_init")

        if not isinstance(kernel, jnp.ndarray) or kernel.ndim != 2:
            raise ValueError("Expected `kernel` to be a 2D `ndarray` with shape (H, W)")

        self.in_features = _check_and_return_positive_int(in_features, "in_features")
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

    def __call__(self, x, **k) -> jnp.ndarray:
        if hasattr(self, "_init"):
            _check_non_tracer(x, self.__class__.__name__)
            getattr(self, "_init")(in_features=x.shape[0])
        _check_spatial_in_shape(x, self.spatial_ndim)
        _check_in_features(x, self.in_features)
        return self.conv(x)


@pytc.treeclass
class FFTFilter2D:
    in_features: int = pytc.field(callbacks=[pytc.freeze])
    kernel: jnp.ndarray = pytc.field(callbacks=[pytc.freeze])

    def __init__(self, in_features: int, kernel: jnp.ndarray):
        """Apply 2D filter for each channel using FFT , faster for large kernels.

        Args:
            in_features: number of input channels
            kernel: kernel array
        """
        if in_features is None:
            for field_item in pytc.fields(self):
                setattr(self, field_item.name, None)

            self._init = ft.partial(FFTFilter2D.__init__, self=self, kernel=kernel)
            return

        if hasattr(self, "_init"):
            delattr(self, "_init")

        if not isinstance(kernel, jnp.ndarray) or kernel.ndim != 2:
            raise ValueError("Expected `kernel` to be a 2D `ndarray` with shape (H, W)")

        self.in_features = _check_and_return_positive_int(in_features, "in_features")
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

    def __call__(self, x, **k) -> jnp.ndarray:
        if hasattr(self, "_init"):
            _check_non_tracer(x, self.__class__.__name__)
            getattr(self, "_init")(in_features=x.shape[0])
        _check_spatial_in_shape(x, self.spatial_ndim)
        _check_in_features(x, self.in_features)
        return self.conv(x)
