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
import pytreeclass as pytc

from serket.nn.utils import (
    KernelSizeType,
    PaddingType,
    StridesType,
    canonicalize,
    validate_spatial_in_shape,
)

# Based on colab hardware benchmarks `kernex` seems to
# be faster on CPU and on par with JAX on GPU.


class GeneralPoolND(pytc.TreeClass):
    func: Callable = pytc.field(repr=False)

    def __init__(
        self,
        kernel_size: KernelSizeType,
        strides: StridesType = 1,
        *,
        padding: PaddingType = "valid",
        func: Callable = None,
        constant_values: float | None = 0,
    ):
        """Apply pooling to the input with function `func` applied to the kernel.

        Args:
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel
            func: function to apply to the kernel
        """
        self.kernel_size = canonicalize(
            kernel_size,
            self.spatial_ndim,
            name="kernel_size",
        )
        self.strides = canonicalize(strides, self.spatial_ndim, name="strides")
        self.padding = padding  # gets canonicalized in kmap

        @jax.vmap
        @kex.kmap(
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )
        def _poolnd(x):
            return func(x)

        self.func = _poolnd

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x, **k):
        return self.func(x)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class LPPoolND(GeneralPoolND):
    def __init__(
        self,
        norm_type: float,
        kernel_size: KernelSizeType,
        strides: StridesType | None = None,
        *,
        padding: PaddingType = "valid",
    ):
        """Apply Lp pooling to the input.

        Args:
            norm_type: norm type
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel
        """

        super().__init__(
            kernel_size=kernel_size,
            strides=strides or kernel_size,
            padding=padding,
            func=lambda x: jnp.sum(x**norm_type) ** (1 / norm_type),
        )


class GlobalPoolND(pytc.TreeClass):
    def __init__(self, keepdims: bool = True, func: Callable = jnp.mean):
        """Apply global pooling to the input with function `func` applied
        to the kernel.

        Args:
            keepdims: keep the spatial dimensions
            func: function to apply to the kernel
        """
        self.keepdims = keepdims
        self.func = func

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        axes = tuple(range(1, self.spatial_ndim + 1))  # reduce spatial dimensions
        return self.func(x, axis=axes, keepdims=self.keepdims)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class AdaptivePoolND(pytc.TreeClass):
    output_size: tuple[int, ...]

    def __init__(self, output_size: tuple[int, ...], *, func: Callable = None):
        """Apply pooling to the input with function `func` applied to the kernel.


        Args:
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel
            func: function to apply to the kernel

        Note:
            The strides and kernel_size are calculated from the output_size as follows:
            * stride_i = (input_size_i//output_size_i)
            * kernel_size_i = input_size_i - (output_size_i-1)*stride_i
            * padding_i = "valid"
        """
        self.output_size = canonicalize(
            output_size, self.spatial_ndim, name="output_size"
        )
        self.func = func

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x, **kwargs):
        input_size = x.shape[1:]
        output_size = self.output_size
        strides = tuple(i // o for i, o in zip(input_size, output_size))
        kernel_size = tuple(
            i - (o - 1) * s for i, o, s in zip(input_size, output_size, strides)
        )

        @jax.vmap
        @kex.kmap(kernel_size=kernel_size, strides=strides)
        def _adaptive_pool(x):
            return self.func(x)

        return _adaptive_pool(x)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class MaxPool1D(GeneralPoolND):
    def __init__(
        self,
        kernel_size: int,
        strides: int = 1,
        *,
        padding: str = "valid",
    ):
        """1D Max Pooling layer
        Args:
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel (valid, same) or tuple of ints
        """
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            func=jnp.max,
            constant_values=-jnp.inf,
        )

    @property
    def spatial_ndim(self) -> int:
        return 1


class MaxPool2D(GeneralPoolND):
    def __init__(
        self,
        kernel_size: int,
        strides: int = 1,
        *,
        padding: str = "valid",
    ):
        """2D Max Pooling layer
        Args:
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel (valid, same) or tuple of ints
        """
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            func=jnp.max,
            constant_values=-jnp.inf,
        )

    @property
    def spatial_ndim(self) -> int:
        return 2


class MaxPool3D(GeneralPoolND):
    def __init__(
        self,
        kernel_size: int,
        strides: int = 1,
        *,
        padding: str = "valid",
    ):
        """3D Max Pooling layer
        Args:
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel (valid, same) or tuple of ints
        """
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            func=jnp.max,
            constant_values=-jnp.inf,
        )

    @property
    def spatial_ndim(self) -> int:
        return 3


class AvgPool1D(GeneralPoolND):
    def __init__(
        self,
        kernel_size: int,
        strides: int = 1,
        *,
        padding: str = "valid",
    ):
        """1D Average Pooling layer
        Args:
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel (valid, same) or tuple of ints
        """
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            func=jnp.mean,
        )

    @property
    def spatial_ndim(self) -> int:
        return 1


class AvgPool2D(GeneralPoolND):
    def __init__(
        self,
        kernel_size: int,
        strides: int = 1,
        *,
        padding: str = "valid",
    ):
        """2D Average Pooling layer
        Args:
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel (valid, same) or tuple of ints
        """
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            func=jnp.mean,
        )

    @property
    def spatial_ndim(self) -> int:
        return 2


class AvgPool3D(GeneralPoolND):
    def __init__(
        self,
        kernel_size: int,
        strides: int = 1,
        *,
        padding: str = "valid",
    ):
        """3D Average Pooling layer
        Args:
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel (valid, same) or tuple of ints
        """
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            func=jnp.mean,
        )

    @property
    def spatial_ndim(self) -> int:
        return 3


class LPPool1D(LPPoolND):
    def __init__(
        self,
        norm_type: float,
        kernel_size: KernelSizeType,
        strides: StridesType | None = None,
        *,
        padding: PaddingType = "valid",
    ):
        """1D Lp pooling to the input.

        Args:
            norm_type: norm type
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel
        """
        super().__init__(
            norm_type=norm_type,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
        )

    @property
    def spatial_ndim(self) -> int:
        return 1


class LPPool2D(LPPoolND):
    def __init__(
        self,
        norm_type: float,
        kernel_size: KernelSizeType,
        strides: StridesType | None = None,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        """2D Lp pooling to the input.

        Args:
            norm_type: norm type
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel
        """
        super().__init__(
            norm_type=norm_type,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
        )

    @property
    def spatial_ndim(self) -> int:
        return 2


class LPPool3D(LPPoolND):
    def __init__(
        self,
        norm_type: float,
        kernel_size: KernelSizeType,
        strides: StridesType = None,
        *,
        padding: PaddingType = "valid",
    ):
        """3D Lp pooling to the input.

        Args:
            norm_type: norm type
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel
        """
        super().__init__(
            norm_type=norm_type,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            spatial_ndim=3,
        )


class GlobalAvgPool1D(GlobalPoolND):
    def __init__(self, keepdims: bool = True):
        """1D Global Average Pooling layer
        Args:
            keepdims: whether to keep the dimensions or not
        """
        super().__init__(func=jnp.mean, keepdims=keepdims)

    @property
    def spatial_ndim(self) -> int:
        return 1


class GlobalAvgPool2D(GlobalPoolND):
    def __init__(self, keepdims: bool = True):
        """2D Global Average Pooling layer
        Args:
            keepdims: whether to keep the dimensions or not
        """
        super().__init__(func=jnp.mean, keepdims=keepdims)

    @property
    def spatial_ndim(self) -> int:
        return 2


class GlobalAvgPool3D(GlobalPoolND):
    def __init__(self, keepdims: bool = True):
        """3D Global Average Pooling layer
        Args:
            keepdims: whether to keep the dimensions or not
        """
        super().__init__(func=jnp.mean, keepdims=keepdims)

    @property
    def spatial_ndim(self) -> int:
        return 3


class GlobalMaxPool1D(GlobalPoolND):
    def __init__(self, keepdims: bool = True):
        """1D Global Max Pooling layer
        Args:
            keepdims: whether to keep the dimensions or not
        """
        super().__init__(func=jnp.max, keepdims=keepdims)

    @property
    def spatial_ndim(self) -> int:
        return 1


class GlobalMaxPool2D(GlobalPoolND):
    def __init__(self, keepdims: bool = True):
        """2D Global Max Pooling layer
        Args:
            keepdims: whether to keep the dimensions or not
        """
        super().__init__(func=jnp.max, keepdims=keepdims)

    @property
    def spatial_ndim(self) -> int:
        return 2


class GlobalMaxPool3D(GlobalPoolND):
    def __init__(self, keepdims: bool = True):
        """3D Global Max Pooling layer
        Args:
            keepdims: whether to keep the dimensions or not
        """
        super().__init__(func=jnp.max, keepdims=keepdims)

    @property
    def spatial_ndim(self) -> int:
        return 3


class AdaptiveAvgPool1D(AdaptivePoolND):
    def __init__(self, output_size: tuple[int, ...]):
        """1D Adaptive Average Pooling layer
        Args:
            output_size: size of the output
        """
        super().__init__(output_size=output_size, func=jnp.mean)

    @property
    def spatial_ndim(self) -> int:
        return 1


class AdaptiveAvgPool2D(AdaptivePoolND):
    def __init__(self, output_size: tuple[int, ...]):
        """2D Adaptive Average Pooling layer
        Args:
            output_size: size of the output
        """
        super().__init__(output_size=output_size, func=jnp.mean)

    @property
    def spatial_ndim(self) -> int:
        return 2


class AdaptiveAvgPool3D(AdaptivePoolND):
    def __init__(self, output_size: tuple[int, ...]):
        """3D Adaptive Average Pooling layer
        Args:
            output_size: size of the output
        """
        super().__init__(output_size=output_size, func=jnp.mean)

    @property
    def spatial_ndim(self) -> int:
        return 3


class AdaptiveMaxPool1D(AdaptivePoolND):
    def __init__(self, output_size: tuple[int, ...]):
        """1D Adaptive Max Pooling layer
        Args:
            output_size: size of the output
        """
        super().__init__(output_size=output_size, func=jnp.max)

    @property
    def spatial_ndim(self) -> int:
        return 1


class AdaptiveMaxPool2D(AdaptivePoolND):
    def __init__(self, output_size: tuple[int, ...]):
        """2D Adaptive Max Pooling layer
        Args:
            output_size: size of the output
        """
        super().__init__(output_size=output_size, func=jnp.max)

    @property
    def spatial_ndim(self) -> int:
        return 2


class AdaptiveMaxPool3D(AdaptivePoolND):
    def __init__(self, output_size: tuple[int, ...]):
        """3D Adaptive Max Pooling layer
        Args:
            output_size: size of the output
        """
        super().__init__(output_size=output_size, func=jnp.max)

    @property
    def spatial_ndim(self) -> int:
        return 3
