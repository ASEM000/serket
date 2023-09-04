# Copyright 2023 Serket authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
from typing_extensions import Annotated

import serket as sk
from serket.nn.convolution import fft_conv_general_dilated
from serket.nn.initialization import DType
from serket.utils import (
    generate_conv_dim_numbers,
    maybe_lazy_call,
    maybe_lazy_init,
    positive_int_cb,
    resolve_string_padding,
    validate_axis_shape,
    validate_spatial_ndim,
)


def is_lazy_call(instance, *_, **__) -> bool:
    return getattr(instance, "in_features", False) is None


def is_lazy_init(_, in_features, *__, **___) -> bool:
    return in_features is None


def infer_in_features(_, x, *__, **___) -> int:
    return x.shape[0]


image_updates = dict(in_features=infer_in_features)


def filter_2d(
    array: Annotated[jax.Array, "CHW"],
    weight: Annotated[jax.Array, "OIHW"],
) -> jax.Array:
    """Filtering wrapping ``jax.lax.conv_general_dilated``.

    Args:
        array: input array. shape is (in_features, *spatial).
        weight: convolutional kernel. shape is (out_features, in_features, *kernel).
    """
    assert array.ndim == 3

    ones = (1,) * (array.ndim - 1)
    x = jax.lax.conv_general_dilated(
        lhs=jnp.expand_dims(array, 0),
        rhs=weight,
        window_strides=ones,
        padding="SAME",
        rhs_dilation=ones,
        dimension_numbers=generate_conv_dim_numbers(array.ndim - 1),
        feature_group_count=array.shape[0],  # in_features
    )
    return jax.lax.stop_gradient_p.bind(jnp.squeeze(x, 0))


def fft_filter_2d(
    array: Annotated[jax.Array, "CHW"],
    weight: Annotated[jax.Array, "OIHW"],
) -> jax.Array:
    """Filtering wrapping ``jax.lax.conv_general_dilated``.

    Args:
        array: input array. shape is (in_features, *spatial).
        weight: convolutional kernel. shape is (out_features, in_features, *kernel).
    """
    assert array.ndim == 3

    ones = (1,) * (array.ndim - 1)

    padding = resolve_string_padding(
        in_dim=array.shape[1:],
        padding="SAME",
        kernel_size=weight.shape[2:],
        strides=ones,
    )

    x = fft_conv_general_dilated(
        lhs=jnp.expand_dims(array, 0),
        rhs=weight,
        strides=ones,
        padding=padding,
        dilation=ones,
        groups=array.shape[0],  # in_features
    )
    return jax.lax.stop_gradient_p.bind(jnp.squeeze(x, 0))


class AvgBlur2D(sk.TreeClass):
    """Average blur 2D layer

    .. image:: ../_static/avgblur2d.png

    Args:
        in_features: number of input channels.
        kernel_size: size of the convolving kernel.
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.AvgBlur2D(in_features=1, kernel_size=3)
        >>> print(layer(jnp.ones((1,5,5))))  # doctest: +SKIP
        [[[0.44444448 0.6666667  0.6666667  0.6666667  0.44444448]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.6666667  1.         1.         1.         0.6666667 ]
          [0.44444448 0.6666667  0.6666667  0.6666667  0.44444448]]]

    Note:
        :class:`.AvgBlur2D` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class Blur(sk.TreeClass):
        ...    l1: sk.image.AvgBlur2D = sk.image.AvgBlur2D(None, 3)
        ...    l2: sk.image.AvgBlur2D = sk.image.AvgBlur2D(None, 3)
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_blur = Blur()
        >>> # materialize the layer
        >>> _, materialized_blur = lazy_blur.at["__call__"](jnp.ones((5, 2, 2)))
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int | None,
        kernel_size: int,
        *,
        dtype: DType = jnp.float32,
    ):
        self.in_features = positive_int_cb(in_features)
        kernel_size = positive_int_cb(kernel_size)
        kernel = jnp.ones(kernel_size)
        kernel = kernel / jnp.sum(kernel)
        kernel = kernel[:, None]
        kernel = jnp.repeat(kernel[None, None], in_features, axis=0).astype(dtype)
        self.kernel = kernel

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=image_updates)
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array) -> jax.Array:
        x = filter_2d(x, self.kernel)
        x = filter_2d(x, jnp.moveaxis(self.kernel, 2, 3))
        return x

    @property
    def spatial_ndim(self) -> int:
        return 2


class GaussianBlur2D(sk.TreeClass):
    """Apply Gaussian blur to a channel-first image.

    .. image:: ../_static/gaussianblur2d.png

    Args:
        in_features: number of input features
        kernel_size: kernel size
        sigma: sigma. Defaults to 1.
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.GaussianBlur2D(in_features=1, kernel_size=3)
        >>> print(layer(jnp.ones((1,5,5))))  # doctest: +SKIP
        [[[0.5269764 0.7259314 0.7259314 0.7259314 0.5269764]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.7259314 1.        1.        1.        0.7259314]
          [0.5269764 0.7259314 0.7259314 0.7259314 0.5269764]]]

    Note:
        :class:`.GaussianBlur2D` supports lazy initialization, meaning that the weights and
        biases are not initialized until the first call to the layer. This is
        useful when the input shape is not known at initialization time.

        To use lazy initialization, pass ``None`` as the ``in_features`` argument
        and use the ``.at["calling_method_name"]`` attribute to call the layer
        with an input of known shape.

        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jax
        >>> @sk.autoinit
        ... class Blur(sk.TreeClass):
        ...    l1: sk.image.GaussianBlur2D = sk.image.GaussianBlur2D(None, 3)
        ...    l2: sk.image.GaussianBlur2D = sk.image.GaussianBlur2D(None, 3)
        ...    def __call__(self, x: jax.Array) -> jax.Array:
        ...        return self.l2(jax.nn.relu(self.l1(x)))
        >>> # lazy initialization
        >>> lazy_blur = Blur()
        >>> # materialize the layer
        >>> _, materialized_blur = lazy_blur.at["__call__"](jnp.ones((5, 2, 2)))
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int,
        kernel_size: int,
        *,
        sigma: float = 1.0,
        dtype: DType = jnp.float32,
    ):
        self.in_features = positive_int_cb(in_features)
        self.kernel_size = positive_int_cb(kernel_size)
        self.sigma = sigma

        x = jnp.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size)
        kernel = jnp.exp(-0.5 * jnp.square(x) * jax.lax.rsqrt(self.sigma))
        kernel = kernel / jnp.sum(kernel)
        kernel = kernel[:, None].astype(dtype)
        self.kernel = jnp.repeat(kernel[None, None], in_features, axis=0)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=image_updates)
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array) -> jax.Array:
        x = filter_2d(x, self.kernel)
        x = filter_2d(x, jnp.moveaxis(self.kernel, 2, 3))
        return x

    @property
    def spatial_ndim(self) -> int:
        return 2


class Filter2D(sk.TreeClass):
    """Apply 2D filter for each channel

    .. image:: ../_static/filter2d.png

    Args:
        in_features: number of input channels.
        kernel: kernel array with shape (H, W).
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.Filter2D(in_features=1, kernel=jnp.ones((3,3)))
        >>> print(layer(jnp.ones((1,5,5))))
        [[[4. 6. 6. 6. 4.]
          [6. 9. 9. 9. 6.]
          [6. 9. 9. 9. 6.]
          [6. 9. 9. 9. 6.]
          [4. 6. 6. 6. 4.]]]
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int,
        kernel: jax.Array,
        *,
        dtype: DType = jnp.float32,
    ):
        if not isinstance(kernel, jax.Array) or kernel.ndim != 2:
            raise ValueError("Expected `kernel` to be a 2D `ndarray` with shape (H, W)")

        self.in_features = positive_int_cb(in_features)
        kernel = jnp.stack([kernel] * in_features, axis=0)
        self.kernel = kernel[:, None].astype(dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=image_updates)
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array) -> jax.Array:
        return filter_2d(x, self.kernel)

    @property
    def spatial_ndim(self) -> int:
        return 2


class FFTFilter2D(sk.TreeClass):
    """Apply 2D filter for each channel using FFT

    .. image:: ../_static/filter2d.png

    Args:
        in_features: number of input channels
        kernel: kernel array
        dtype: data type of the layer. Defaults to ``jnp.float32``.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.image.FFTFilter2D(in_features=1, kernel=jnp.ones((3,3)))
        >>> print(layer(jnp.ones((1,5,5))))  # doctest: +SKIP
        [[[4.0000005 6.0000005 6.000001  6.0000005 4.0000005]
          [6.0000005 9.        9.        9.        6.0000005]
          [6.0000005 9.        9.        9.        6.0000005]
          [6.0000005 9.        9.        9.        6.0000005]
          [4.        6.0000005 6.0000005 6.0000005 4.       ]]]
    """

    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
    def __init__(
        self,
        in_features: int,
        kernel: jax.Array,
        *,
        dtype: DType = jnp.float32,
    ):
        if kernel.ndim != 2:
            raise ValueError("Expected `kernel` to be a 2D `ndarray` with shape (H, W)")

        self.in_features = positive_int_cb(in_features)
        kernel = jnp.stack([kernel] * in_features, axis=0)
        self.kernel = kernel[:, None].astype(dtype)

    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=image_updates)
    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    @ft.partial(validate_axis_shape, attribute_name="in_features", axis=0)
    def __call__(self, x: jax.Array) -> jax.Array:
        return fft_filter_2d(x, self.kernel)

    @property
    def spatial_ndim(self) -> int:
        return 2
