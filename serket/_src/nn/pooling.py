# Copyright 2023 serket authors
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

import abc
import functools as ft
from typing import Callable

import jax
import jax.numpy as jnp
from typing_extensions import Annotated

import serket as sk
from serket._src.utils import (
    KernelSizeType,
    PaddingType,
    StridesType,
    canonicalize,
    delayed_canonicalize_padding,
    kernel_map,
    validate_spatial_nd,
)


def pool_nd(
    reducer: Callable[[jax.Array], jax.Array],
    inital_value: float,
    array: Annotated[jax.Array, "I..."],
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    padding: tuple[int, ...],
):
    """ND pooling operation

    Args:
        reducer: reducer function. Takes an array and returns a single value
        array: channeled array of shape (channels, *spatial_dims)
        kernel_size: size of the kernel. accepts tuple of ints for each spatial dimension
        strides: strides of the kernel. accepts tuple of ints for each spatial dimension
        padding: padding of the kernel. accepts tuple of tuples of two ints for
            each spatial dimension for each side of the array
    """

    @jax.vmap
    @ft.partial(
        kernel_map,
        shape=array.shape[1:],
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        padding_kwargs=dict(constant_values=inital_value),
    )
    def reducer_map(x):
        return reducer(x)

    return reducer_map(array)


max_op = jax.custom_jvp(lambda x: jnp.maximum(jnp.max(x), -jnp.inf))


@max_op.defjvp
def _(primals, tangents):
    (x,), (g,) = primals, tangents
    return max_op(x), g.ravel()[x.argmax()]


def max_pool_nd(
    array: jax.Array,
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
) -> jax.Array:
    return pool_nd(max_op, -jnp.inf, array, kernel_size, strides, padding)


def avg_pool_nd(
    array: jax.Array,
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
) -> jax.Array:
    return pool_nd(jnp.mean, 0, array, kernel_size, strides, padding)


def lp_pool_nd(
    array: jax.Array,
    norm_type: float,
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
) -> jax.Array:
    def reducer(x):
        return jnp.sum(x**norm_type) ** (1 / norm_type)

    return pool_nd(reducer, 0, array, kernel_size, strides, padding)


def adaptive_pool_nd(
    reducer: Callable[[jax.Array], jax.Array],
    array: jax.Array,
    outdim: tuple[int, ...],
) -> jax.Array:
    indim = array.shape[1:]
    strides = tuple(i // o for i, o in zip(indim, outdim))
    kernel_size = tuple(i - (o - 1) * s for i, o, s in zip(indim, outdim, strides))

    @jax.vmap
    @ft.partial(
        kernel_map,
        shape=indim,
        kernel_size=kernel_size,
        strides=strides,
        padding=((0, 0),) * len(indim),
    )
    def reducer_map(x):
        return reducer(x)

    return reducer_map(array)


adaptive_avg_pool_nd = ft.partial(adaptive_pool_nd, jnp.mean)
adaptive_max_pool_nd = ft.partial(adaptive_pool_nd, jnp.max)


class MaxPoolND(sk.TreeClass):
    def __init__(
        self,
        kernel_size: KernelSizeType,
        strides: StridesType = 1,
        *,
        padding: PaddingType = "valid",
    ):
        self.kernel_size = canonicalize(
            kernel_size,
            self.spatial_ndim,
            name="kernel_size",
        )
        self.strides = canonicalize(strides, self.spatial_ndim, name="strides")
        self.padding = padding

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x):
        padding = delayed_canonicalize_padding(
            in_dim=x.shape,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )

        return max_pool_nd(x, self.kernel_size, self.strides, padding)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class MaxPool1D(MaxPoolND):
    """1D Max Pooling layer

    Args:
        kernel_size: size of the kernel
        strides: strides of the kernel
        padding: padding of the kernel (valid, same) or tuple of ints

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> layer = sk.nn.MaxPool1D(kernel_size=2, strides=2)
        >>> x = jnp.arange(1, 11).reshape(1, 10).astype(jnp.float32)
        >>> print(layer(x))
        [[ 2.  4.  6.  8. 10.]]
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class MaxPool2D(MaxPoolND):
    """2D Max Pooling layer

    Args:
        kernel_size: size of the kernel
        strides: strides of the kernel
        padding: padding of the kernel (valid, same) or tuple of ints

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> layer = sk.nn.MaxPool2D(kernel_size=2, strides=2)
        >>> x = jnp.arange(1, 17).reshape(1, 4, 4).astype(jnp.float32)
        >>> print(layer(x))
        [[[ 6.  8.]
          [14. 16.]]]
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class MaxPool3D(MaxPoolND):
    """3D Max Pooling layer

    Args:
        kernel_size: size of the kernel
        strides: strides of the kernel
        padding: padding of the kernel (valid, same) or tuple of ints
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class AvgPoolND(sk.TreeClass):
    def __init__(
        self,
        kernel_size: KernelSizeType,
        strides: StridesType = 1,
        *,
        padding: PaddingType = "valid",
    ):
        self.kernel_size = canonicalize(
            kernel_size,
            self.spatial_ndim,
            name="kernel_size",
        )
        self.strides = canonicalize(strides, self.spatial_ndim, name="strides")
        self.padding = padding

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x):
        padding = delayed_canonicalize_padding(
            in_dim=x.shape,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )

        return avg_pool_nd(x, self.kernel_size, self.strides, padding)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class AvgPool1D(AvgPoolND):
    """1D Average Pooling layer

    Args:
        kernel_size: size of the kernel
        strides: strides of the kernel
        padding: padding of the kernel (valid, same) or tuple of ints
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class AvgPool2D(AvgPoolND):
    """2D Average Pooling layer

    Args:
        kernel_size: size of the kernel
        strides: strides of the kernel
        padding: padding of the kernel (valid, same) or tuple of ints
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class AvgPool3D(AvgPoolND):
    """3D Average Pooling layer

    Args:
        kernel_size: size of the kernel
        strides: strides of the kernel
        padding: padding of the kernel (valid, same) or tuple of ints
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class LPPoolND(sk.TreeClass):
    def __init__(
        self,
        norm_type: float,
        kernel_size: KernelSizeType,
        strides: StridesType = 1,
        *,
        padding: PaddingType = "valid",
    ):
        self.norm_type = norm_type

        self.kernel_size = canonicalize(
            kernel_size,
            self.spatial_ndim,
            name="kernel_size",
        )
        self.strides = canonicalize(strides, self.spatial_ndim, name="strides")
        self.padding = padding

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        padding = delayed_canonicalize_padding(
            in_dim=x.shape,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )

        return lp_pool_nd(x, self.norm_type, self.kernel_size, self.strides, padding)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class LPPool1D(LPPoolND):
    """1D Lp pooling to the input.

    Args:
        norm_type: norm type
        kernel_size: size of the kernel
        strides: strides of the kernel
        padding: padding of the kernel
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class LPPool2D(LPPoolND):
    """2D Lp pooling to the input.

    Args:
        norm_type: norm type
        kernel_size: size of the kernel
        strides: strides of the kernel
        padding: padding of the kernel
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class LPPool3D(LPPoolND):
    """3D Lp pooling to the input.

    Args:
        norm_type: norm type
        kernel_size: size of the kernel
        strides: strides of the kernel
        padding: padding of the kernel
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class GlobalAvgPoolND(sk.TreeClass):
    def __init__(self, keepdims: bool = True):
        self.keepdims = keepdims

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        axes = tuple(range(1, self.spatial_ndim + 1))  # reduce spatial dimensions
        return jnp.mean(x, axis=axes, keepdims=self.keepdims)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class GlobalAvgPool1D(GlobalAvgPoolND):
    """1D Global Average Pooling layer

    Args:
        keepdims: whether to keep the dimensions or not
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class GlobalAvgPool2D(GlobalAvgPoolND):
    """2D Global Average Pooling layer

    Args:
        keepdims: whether to keep the dimensions or not
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class GlobalAvgPool3D(GlobalAvgPoolND):
    """3D Global Average Pooling layer

    Args:
        keepdims: whether to keep the dimensions or not
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class GlobalMaxPoolND(sk.TreeClass):
    def __init__(self, keepdims: bool = True):
        self.keepdims = keepdims

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        axes = tuple(range(1, self.spatial_ndim + 1))  # reduce spatial dimensions
        return jnp.max(x, axis=axes, keepdims=self.keepdims)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class GlobalMaxPool1D(GlobalMaxPoolND):
    """1D Global Max Pooling layer

    Args:
        keepdims: whether to keep the dimensions or not
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class GlobalMaxPool2D(GlobalMaxPoolND):
    """2D Global Max Pooling layer

    Args:
        keepdims: whether to keep the dimensions or not
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class GlobalMaxPool3D(GlobalMaxPoolND):
    """3D Global Max Pooling layer

    Args:
        keepdims: whether to keep the dimensions or not
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class AdaptiveAvgPoolND(sk.TreeClass):
    def __init__(self, output_size: tuple[int, ...]):
        self.output_size = canonicalize(
            output_size,
            self.spatial_ndim,
            name="output_size",
        )

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        return adaptive_avg_pool_nd(x, self.output_size)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class AdaptiveAvgPool1D(AdaptiveAvgPoolND):
    """1D Adaptive Average Pooling layer

    Args:
        output_size: size of the output
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class AdaptiveAvgPool2D(AdaptiveAvgPoolND):
    """2D Adaptive Average Pooling layer

    Args:
        output_size: size of the output
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class AdaptiveAvgPool3D(AdaptiveAvgPoolND):
    """3D Adaptive Average Pooling layer

    Args:
        output_size: size of the output
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class AdaptiveMaxPoolND(sk.TreeClass):
    def __init__(self, output_size: tuple[int, ...]):
        self.output_size = canonicalize(
            output_size,
            self.spatial_ndim,
            name="output_size",
        )

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array) -> jax.Array:
        return adaptive_max_pool_nd(x, self.output_size)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class AdaptiveMaxPool1D(AdaptiveMaxPoolND):
    """1D Adaptive Max Pooling layer

    Args:
        output_size: size of the output
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class AdaptiveMaxPool2D(AdaptiveMaxPoolND):
    """2D Adaptive Max Pooling layer

    Args:
        output_size: size of the output
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class AdaptiveMaxPool3D(AdaptiveMaxPoolND):
    """3D Adaptive Max Pooling layer

    Args:
        output_size: size of the output
    """

    @property
    def spatial_ndim(self) -> int:
        return 3
