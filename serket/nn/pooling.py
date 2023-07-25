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

import abc
import functools as ft
from typing import Callable

import jax
import jax.numpy as jnp
import kernex as kex

import serket as sk
from serket.nn.utils import (
    KernelSizeType,
    PaddingType,
    StridesType,
    canonicalize,
    delayed_canonicalize_padding,
    validate_spatial_ndim,
)

# Based on colab hardware benchmarks `kernex` seems to
# be faster on CPU and on par with JAX on GPU for low number of channels.


class GeneralPoolND(sk.TreeClass):
    def __init__(
        self,
        kernel_size: KernelSizeType,
        strides: StridesType = 1,
        *,
        padding: PaddingType = "valid",
        operation: Callable[[jax.Array], jax.Array],
    ):
        self.kernel_size = canonicalize(
            kernel_size,
            self.spatial_ndim,
            name="kernel_size",
        )
        self.strides = canonicalize(strides, self.spatial_ndim, name="strides")
        self.padding = padding
        self.operation = operation

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x, **k):
        padding = delayed_canonicalize_padding(
            in_dim=x.shape,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )

        @jax.vmap
        @kex.kmap(
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=padding,
        )
        def pool(x):
            return self.operation(x)

        return pool(x)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class MaxPoolND(GeneralPoolND):
    def __init__(
        self,
        kernel_size: int,
        strides: int = 1,
        *,
        padding: str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            operation=jnp.max,
        )


class AvgPoolND(GeneralPoolND):
    def __init__(
        self,
        kernel_size: int,
        strides: int = 1,
        *,
        padding: str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            operation=jnp.mean,
        )


class LPPoolND(GeneralPoolND):
    def __init__(
        self,
        norm_type: float,
        kernel_size: KernelSizeType,
        strides: StridesType | None = None,
        *,
        padding: PaddingType = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides or kernel_size,
            padding=padding,
            operation=lambda x: jnp.sum(x**norm_type) ** (1 / norm_type),
        )


class GlobalPoolND(sk.TreeClass):
    def __init__(self, keepdims: bool = True, operation: Callable = jnp.mean):
        self.keepdims = keepdims
        self.operation = operation

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        axes = tuple(range(1, self.spatial_ndim + 1))  # reduce spatial dimensions
        return self.operation(x, axis=axes, keepdims=self.keepdims)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class GlobalAvgPoolND(GlobalPoolND):
    def __init__(self, keepdims: bool = True):
        super().__init__(keepdims=keepdims, operation=jnp.mean)


class GlobalMaxPoolND(GlobalPoolND):
    def __init__(self, keepdims: bool = True):
        super().__init__(keepdims=keepdims, operation=jnp.max)


class AdaptivePoolND(sk.TreeClass):
    def __init__(self, output_size: tuple[int, ...], *, func: Callable = None):
        self.output_size = canonicalize(
            output_size,
            self.spatial_ndim,
            name="output_size",
        )
        self.func = func

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x, **k):
        indim = x.shape[1:]
        outdim = self.output_size
        strides = tuple(i // o for i, o in zip(indim, outdim))
        kernel_size = tuple(i - (o - 1) * s for i, o, s in zip(indim, outdim, strides))

        @jax.vmap
        @kex.kmap(kernel_size=kernel_size, strides=strides)
        def _adaptive_pool(x):
            return self.func(x)

        return _adaptive_pool(x)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class AdaptiveAvgPoolND(AdaptivePoolND):
    def __init__(self, output_size: tuple[int, ...]):
        super().__init__(output_size=output_size, func=jnp.mean)


class AdaptiveMaxPoolND(AdaptivePoolND):
    def __init__(self, output_size: tuple[int, ...]):
        super().__init__(output_size=output_size, func=jnp.max)


# implementation of pooling layers


class MaxPool1D(MaxPoolND):
    """1D Max Pooling layer

    Args:
        kernel_size: size of the kernel
        strides: strides of the kernel
        padding: padding of the kernel (valid, same) or tuple of ints
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
