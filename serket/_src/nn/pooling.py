# Copyright 2024 serket authors
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
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from typing_extensions import Annotated

from serket import TreeClass
from serket._src.utils.convert import canonicalize
from serket._src.utils.mapping import kernel_map
from serket._src.utils.padding import delayed_canonicalize_padding
from serket._src.utils.typing import KernelSizeType, PaddingType, StridesType
from serket._src.utils.validate import validate_spatial_ndim


def pool_nd(
    reducer: Callable[[jax.Array], jax.Array],
    inital_value: float,
    input: Annotated[jax.Array, "I..."],
    kernel_size: Sequence[int],
    strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
):
    """Pooling operation

    Args:
        reducer: reducer function. Takes an input and returns a single value
        input: channeled input of shape (channels, spatial_dims)
        kernel_size: size of the kernel. accepts a sequence of ints for each spatial dimension
        strides: strides of the kernel. accepts a sequence of ints for each spatial dimension
        padding: padding of the kernel. accepts a sequence of tuples of two ints for
            each spatial dimension for each side of the input
    """
    _, *S = input.shape

    @jax.vmap
    @ft.partial(
        kernel_map,
        shape=S,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        padding_mode=inital_value,
    )
    def reducer_map(view):
        return reducer(view)

    return reducer_map(input)


max_op = jax.custom_jvp(lambda x: jnp.maximum(jnp.max(x), -jnp.inf))


@max_op.defjvp
def _(primals, tangents):
    (x,), (g,) = primals, tangents
    return max_op(x), g.ravel()[x.argmax()]


def max_pool_nd(
    input: jax.Array,
    kernel_size: Sequence[int],
    strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
) -> jax.Array:
    """Max pooling operation

    Args:
        input: channeled input of shape (channels, spatial_dims)
        kernel_size: size of the kernel. accepts a sequence of ints for each spatial dimension
        strides: strides of the kernel. accepts a sequence of ints for each spatial dimension
        padding: padding of the kernel. accepts a sequence of tuples of two ints for
            each spatial dimension for each side of the input

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> kernel_size = (3, 3)
        >>> strides = (2, 2)
        >>> input = jnp.ones((2, 25, 25))
        >>> padding = ((1, 1), (1, 1)) # pad 1 on each side of the spatial dimensions
        >>> output = sk.nn.max_pool_nd(input, kernel_size, strides, padding)
        >>> print(output.shape)
        (2, 13, 13)
    """
    return pool_nd(max_op, -jnp.inf, input, kernel_size, strides, padding)


def avg_pool_nd(
    input: jax.Array,
    kernel_size: Sequence[int],
    strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
) -> jax.Array:
    """Average pooling operation

    Args:
        input: channeled input of shape (channels, spatial_dims)
        kernel_size: size of the kernel. accepts tuple of ints for each spatial dimension
        strides: strides of the kernel. accepts tuple of ints for each spatial dimension
        padding: padding of the kernel. accepts tuple of tuples of two ints for
            each spatial dimension for each side of the input

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> kernel_size = (3, 3)
        >>> strides = (2, 2)
        >>> input = jnp.ones((2, 25, 25))
        >>> padding = ((1, 1), (1, 1)) # pad 1 on each side of the spatial dimensions
        >>> output = sk.nn.avg_pool_nd(input, kernel_size, strides, padding)
        >>> print(output.shape)
        (2, 13, 13)
    """
    return pool_nd(jnp.mean, 0, input, kernel_size, strides, padding)


def lp_pool_nd(
    input: jax.Array,
    norm_type: float,
    kernel_size: Sequence[int],
    strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
) -> jax.Array:
    """Lp pooling operation

    Args:
        input: channeled input of shape (channels, spatial_dims)
        norm_type: norm type as a float
        kernel_size: size of the kernel. accepts tuple of ints for each spatial dimension
        strides: strides of the kernel. accepts tuple of ints for each spatial dimension
        padding: padding of the kernel. accepts tuple of tuples of two ints for
            each spatial dimension for each side of the input

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> kernel_size = (3, 3)
        >>> strides = (2, 2)
        >>> input = jnp.ones((2, 25, 25))
        >>> norm_type = 2
        >>> padding = ((1, 1), (1, 1)) # pad 1 on each side of the spatial dimensions
        >>> output = sk.nn.lp_pool_nd(input, norm_type, kernel_size, strides, padding)
        >>> print(output.shape)
        (2, 13, 13)
    """

    def reducer(input: jax.Array) -> jax.Array:
        return jnp.sum(input**norm_type) ** (1 / norm_type)

    return pool_nd(reducer, 0, input, kernel_size, strides, padding)


def adaptive_pool_nd(
    reducer: Callable[[jax.Array], jax.Array],
    input: jax.Array,
    out_dim: Sequence[int],
) -> jax.Array:
    in_dim = input.shape[1:]
    strides = tuple(i // o for i, o in zip(in_dim, out_dim))
    kernel_size = tuple(i - (o - 1) * s for i, o, s in zip(in_dim, out_dim, strides))

    @jax.vmap
    @ft.partial(
        kernel_map,
        shape=in_dim,
        kernel_size=kernel_size,
        strides=strides,
        padding=((0, 0),) * len(in_dim),
    )
    def reducer_map(view: jax.Array) -> jax.Array:
        return reducer(view)

    return reducer_map(input)


def adaptive_avg_pool_nd(input: jax.Array, out_dim: Sequence[int]) -> jax.Array:
    """Adaptive average pooling operation

    Args:
        input: channeled input of shape (channels, spatial_dims)
        out_dim: output dimension. accepts a sequence of ints for each spatial dimension

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> input = jnp.ones((2, 25, 25))
        >>> out_dim = (13, 13)
        >>> output = sk.nn.adaptive_avg_pool_nd(input, out_dim)
        >>> print(output.shape)
        (2, 13, 13)
    """
    return adaptive_pool_nd(jnp.mean, input, out_dim)


def adaptive_max_pool_nd(input: jax.Array, out_dim: Sequence[int]) -> jax.Array:
    """Adaptive max pooling operation

    Args:
        input: channeled input of shape (channels, spatial_dims)
        out_dim: output dimension. accepts a sequence of ints for each spatial dimension

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> input = jnp.ones((2, 25, 25))
        >>> out_dim = (13, 13)
        >>> output = sk.nn.adaptive_max_pool_nd(input, out_dim)
        >>> print(output.shape)
        (2, 13, 13)
    """
    return adaptive_pool_nd(max_op, input, out_dim)


class MaxPoolND(TreeClass):
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

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, input: jax.Array) -> jax.Array:
        padding = delayed_canonicalize_padding(
            in_dim=input.shape,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )

        return max_pool_nd(
            input=input,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=padding,
        )

    spatial_ndim = property(abc.abstractmethod(lambda _: ...))


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

    spatial_ndim: int = 1


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

    spatial_ndim: int = 2


class MaxPool3D(MaxPoolND):
    """3D Max Pooling layer

    Args:
        kernel_size: size of the kernel
        strides: strides of the kernel
        padding: padding of the kernel (valid, same) or tuple of ints
    """

    spatial_ndim: int = 3


class AvgPoolND(TreeClass):
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

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, input: jax.Array) -> jax.Array:
        padding = delayed_canonicalize_padding(
            in_dim=input.shape,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )

        return avg_pool_nd(
            input=input,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=padding,
        )

    spatial_ndim = property(abc.abstractmethod(lambda _: ...))


class AvgPool1D(AvgPoolND):
    """1D Average Pooling layer

    Args:
        kernel_size: size of the kernel
        strides: strides of the kernel
        padding: padding of the kernel (valid, same) or tuple of ints
    """

    spatial_ndim: int = 1


class AvgPool2D(AvgPoolND):
    """2D Average Pooling layer

    Args:
        kernel_size: size of the kernel
        strides: strides of the kernel
        padding: padding of the kernel (valid, same) or tuple of ints
    """

    spatial_ndim: int = 2


class AvgPool3D(AvgPoolND):
    """3D Average Pooling layer

    Args:
        kernel_size: size of the kernel
        strides: strides of the kernel
        padding: padding of the kernel (valid, same) or tuple of ints
    """

    spatial_ndim: int = 3


class LPPoolND(TreeClass):
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

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, input: jax.Array) -> jax.Array:
        padding = delayed_canonicalize_padding(
            in_dim=input.shape,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )

        return lp_pool_nd(
            input=input,
            norm_type=self.norm_type,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=padding,
        )

    spatial_ndim = property(abc.abstractmethod(lambda _: ...))


class LPPool1D(LPPoolND):
    """1D Lp pooling to the input.

    Args:
        norm_type: norm type
        kernel_size: size of the kernel
        strides: strides of the kernel
        padding: padding of the kernel
    """

    spatial_ndim: int = 1


class LPPool2D(LPPoolND):
    """2D Lp pooling to the input.

    Args:
        norm_type: norm type
        kernel_size: size of the kernel
        strides: strides of the kernel
        padding: padding of the kernel
    """

    spatial_ndim: int = 2


class LPPool3D(LPPoolND):
    """3D Lp pooling to the input.

    Args:
        norm_type: norm type
        kernel_size: size of the kernel
        strides: strides of the kernel
        padding: padding of the kernel
    """

    spatial_ndim: int = 3


class GlobalAvgPoolND(TreeClass):
    def __init__(self, keepdims: bool = True):
        self.keepdims = keepdims

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, input: jax.Array) -> jax.Array:
        axes = tuple(range(1, self.spatial_ndim + 1))  # reduce spatial dimensions
        return jnp.mean(input, axis=axes, keepdims=self.keepdims)

    spatial_ndim = property(abc.abstractmethod(lambda _: ...))


class GlobalAvgPool1D(GlobalAvgPoolND):
    """1D Global Average Pooling layer

    Args:
        keepdims: whether to keep the dimensions or not
    """

    spatial_ndim: int = 1


class GlobalAvgPool2D(GlobalAvgPoolND):
    """2D Global Average Pooling layer

    Args:
        keepdims: whether to keep the dimensions or not
    """

    spatial_ndim: int = 2


class GlobalAvgPool3D(GlobalAvgPoolND):
    """3D Global Average Pooling layer

    Args:
        keepdims: whether to keep the dimensions or not
    """

    spatial_ndim: int = 3


class GlobalMaxPoolND(TreeClass):
    def __init__(self, keepdims: bool = True):
        self.keepdims = keepdims

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, input: jax.Array) -> jax.Array:
        axes = tuple(range(1, self.spatial_ndim + 1))  # reduce spatial dimensions
        return jnp.max(input, axis=axes, keepdims=self.keepdims)

    spatial_ndim = property(abc.abstractmethod(lambda _: ...))


class GlobalMaxPool1D(GlobalMaxPoolND):
    """1D Global Max Pooling layer

    Args:
        keepdims: whether to keep the dimensions or not
    """

    spatial_ndim: int = 1


class GlobalMaxPool2D(GlobalMaxPoolND):
    """2D Global Max Pooling layer

    Args:
        keepdims: whether to keep the dimensions or not
    """

    spatial_ndim: int = 2


class GlobalMaxPool3D(GlobalMaxPoolND):
    """3D Global Max Pooling layer

    Args:
        keepdims: whether to keep the dimensions or not
    """

    spatial_ndim: int = 3


class AdaptiveAvgPoolND(TreeClass):
    def __init__(self, output_size: tuple[int, ...]):
        self.output_size = canonicalize(
            output_size,
            self.spatial_ndim,
            name="output_size",
        )

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, input: jax.Array) -> jax.Array:
        return adaptive_avg_pool_nd(input, self.output_size)

    spatial_ndim = property(abc.abstractmethod(lambda _: ...))


class AdaptiveAvgPool1D(AdaptiveAvgPoolND):
    """1D Adaptive Average Pooling layer

    Args:
        output_size: size of the output
    """

    spatial_ndim: int = 1


class AdaptiveAvgPool2D(AdaptiveAvgPoolND):
    """2D Adaptive Average Pooling layer

    Args:
        output_size: size of the output
    """

    spatial_ndim: int = 2


class AdaptiveAvgPool3D(AdaptiveAvgPoolND):
    """3D Adaptive Average Pooling layer

    Args:
        output_size: size of the output
    """

    spatial_ndim: int = 3


class AdaptiveMaxPoolND(TreeClass):
    def __init__(self, output_size: tuple[int, ...]):
        self.output_size = canonicalize(
            output_size,
            self.spatial_ndim,
            name="output_size",
        )

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, input: jax.Array) -> jax.Array:
        return adaptive_max_pool_nd(input, self.output_size)

    spatial_ndim = property(abc.abstractmethod(lambda _: ...))


class AdaptiveMaxPool1D(AdaptiveMaxPoolND):
    """1D Adaptive Max Pooling layer

    Args:
        output_size: size of the output
    """

    spatial_ndim: int = 1


class AdaptiveMaxPool2D(AdaptiveMaxPoolND):
    """2D Adaptive Max Pooling layer

    Args:
        output_size: size of the output
    """

    spatial_ndim: int = 2


class AdaptiveMaxPool3D(AdaptiveMaxPoolND):
    """3D Adaptive Max Pooling layer

    Args:
        output_size: size of the output
    """

    spatial_ndim: int = 3
