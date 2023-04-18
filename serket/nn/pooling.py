from __future__ import annotations

import functools as ft
from typing import Callable

import jax
import jax.numpy as jnp
import kernex as kex
import pytreeclass as pytc

from serket.nn.callbacks import validate_spatial_in_shape
from serket.nn.utils import KernelSizeType, PaddingType, StridesType, canonicalize

# Based on colab hardware benchmarks `kernex` seems to
# be faster on CPU and on par with JAX on GPU.


class GeneralPoolND(pytc.TreeClass):
    kernel_size: KernelSizeType
    strides: StridesType
    padding: PaddingType

    def __init__(
        self,
        kernel_size: KernelSizeType,
        strides: StridesType = 1,
        *,
        padding: PaddingType = "valid",
        spatial_ndim: int = 1,
        func: Callable = None,
    ):
        """Apply pooling to the input with function `func` applied to the kernel.

        Args:
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel
            spatial_ndim: number of dimensions
            func: function to apply to the kernel
        """
        self.kernel_size = canonicalize(kernel_size, spatial_ndim, "kernel_size")
        self.strides = canonicalize(strides, spatial_ndim, "strides")
        self.padding = padding  # gets canonicalized in kmap
        self.spatial_ndim = spatial_ndim

        @jax.jit
        @jax.vmap
        @kex.kmap(self.kernel_size, self.strides, self.padding)
        def _poolnd(x):
            return func(x)

        self.func = _poolnd

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x, **k):
        return self.func(x)


class LPPoolND(GeneralPoolND):
    def __init__(
        self,
        norm_type: float,
        kernel_size: KernelSizeType,
        strides: StridesType | None = None,
        *,
        padding: PaddingType = "valid",
        spatial_ndim: int = 1,
    ):
        """Apply Lp pooling to the input.

        Args:
            norm_type: norm type
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel
            spatial_ndim: number of dimensions
        """

        super().__init__(
            kernel_size=kernel_size,
            strides=strides or kernel_size,
            padding=padding,
            spatial_ndim=spatial_ndim,
            func=lambda x: jnp.sum(x**norm_type) ** (1 / norm_type),
        )


class GlobalPoolND(pytc.TreeClass):
    def __init__(
        self, keepdims: bool = True, spatial_ndim: int = 1, func: Callable = jnp.mean
    ):
        """Apply global pooling to the input with function `func` applied to the kernel.
        Args:
            keepdims: keep the spatial dimensions
            spatial_ndim: number of spatial dimensions
            func: function to apply to the kernel
        """
        self.keepdims = keepdims
        self.spatial_ndim = spatial_ndim
        self.func = func

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        axes = tuple(range(1, self.spatial_ndim + 1))  # reduce spatial dimensions
        return self.func(x, axis=axes, keepdims=self.keepdims)


class AdaptivePoolND(pytc.TreeClass):
    output_size: tuple[int, ...]

    def __init__(
        self,
        output_size: tuple[int, ...],
        *,
        spatial_ndim: int = 1,
        func: Callable = None,
    ):
        """Apply pooling to the input with function `func` applied to the kernel.


        Args:
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel
            spatial_ndim: number of dimensions
            func: function to apply to the kernel

        Note:
            The strides and kernel_size are calculated from the output_size as follows:
            * stride_i = (input_size_i//output_size_i)
            * kernel_size_i = input_size_i - (output_size_i-1)*stride_i
            * padding_i = "valid"
        """
        self.output_size = canonicalize(output_size, spatial_ndim, "output_size")
        self.spatial_ndim = spatial_ndim
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


class AdaptiveConcatPoolND(pytc.TreeClass):
    def __init__(self, output_size: tuple[int, ...], spatial_ndim: int):
        """Concatenate AdaptiveAvgPool1D and AdaptiveMaxPool1D
        See: https://github.com/fastai/fastai/blob/master/fastai/layers.py#L110
        """
        self.spatial_ndim = spatial_ndim
        self.avg_pool = AdaptivePoolND(
            output_size,
            func=jnp.mean,
            spatial_ndim=spatial_ndim,
        )
        self.max_pool = AdaptivePoolND(
            output_size,
            func=jnp.max,
            spatial_ndim=spatial_ndim,
        )

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x, **k):
        return jnp.concatenate([self.max_pool(x), self.avg_pool(x)], axis=0)


class MaxPool1D(GeneralPoolND):
    def __init__(self, kernel_size: int, strides: int = 1, *, padding: str = "valid"):
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
            spatial_ndim=1,
            func=jnp.max,
        )


class MaxPool2D(GeneralPoolND):
    def __init__(self, kernel_size: int, strides: int = 1, *, padding: str = "valid"):
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
            spatial_ndim=2,
            func=jnp.max,
        )


class MaxPool3D(GeneralPoolND):
    def __init__(self, kernel_size: int, strides: int = 1, *, padding: str = "valid"):
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
            spatial_ndim=3,
            func=jnp.max,
        )


class AvgPool1D(GeneralPoolND):
    def __init__(self, kernel_size: int, strides: int = 1, *, padding: str = "valid"):
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
            spatial_ndim=1,
            func=jnp.mean,
        )


class AvgPool2D(GeneralPoolND):
    def __init__(self, kernel_size: int, strides: int = 1, *, padding: str = "valid"):
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
            spatial_ndim=2,
            func=jnp.mean,
        )


class AvgPool3D(GeneralPoolND):
    def __init__(self, kernel_size: int, strides: int = 1, *, padding: str = "valid"):
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
            spatial_ndim=3,
            func=jnp.mean,
        )


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
            spatial_ndim=1,
        )


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
            spatial_ndim=2,
        )


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
        super().__init__(
            spatial_ndim=1,
            func=jnp.mean,
            keepdims=keepdims,
        )


class GlobalAvgPool2D(GlobalPoolND):
    def __init__(self, keepdims: bool = True):
        """2D Global Average Pooling layer
        Args:
            keepdims: whether to keep the dimensions or not
        """
        super().__init__(
            spatial_ndim=2,
            func=jnp.mean,
            keepdims=keepdims,
        )


class GlobalAvgPool3D(GlobalPoolND):
    def __init__(self, keepdims: bool = True):
        """3D Global Average Pooling layer
        Args:
            keepdims: whether to keep the dimensions or not
        """
        super().__init__(
            spatial_ndim=3,
            func=jnp.mean,
            keepdims=keepdims,
        )


class GlobalMaxPool1D(GlobalPoolND):
    def __init__(self, keepdims: bool = True):
        """1D Global Max Pooling layer
        Args:
            keepdims: whether to keep the dimensions or not
        """
        super().__init__(
            spatial_ndim=1,
            func=jnp.max,
            keepdims=keepdims,
        )


class GlobalMaxPool2D(GlobalPoolND):
    def __init__(self, keepdims: bool = True):
        """2D Global Max Pooling layer
        Args:
            keepdims: whether to keep the dimensions or not
        """
        super().__init__(
            spatial_ndim=2,
            func=jnp.max,
            keepdims=keepdims,
        )


class GlobalMaxPool3D(GlobalPoolND):
    def __init__(self, keepdims: bool = True):
        """3D Global Max Pooling layer
        Args:
            keepdims: whether to keep the dimensions or not
        """
        super().__init__(
            spatial_ndim=3,
            func=jnp.max,
            keepdims=keepdims,
        )


class AdaptiveAvgPool1D(AdaptivePoolND):
    def __init__(self, output_size: tuple[int, ...]):
        """1D Adaptive Average Pooling layer
        Args:
            output_size: size of the output
        """
        super().__init__(
            output_size=output_size,
            spatial_ndim=1,
            func=jnp.mean,
        )


class AdaptiveAvgPool2D(AdaptivePoolND):
    def __init__(self, output_size: tuple[int, ...]):
        """2D Adaptive Average Pooling layer
        Args:
            output_size: size of the output
        """
        super().__init__(
            output_size=output_size,
            spatial_ndim=2,
            func=jnp.mean,
        )


class AdaptiveAvgPool3D(AdaptivePoolND):
    def __init__(self, output_size: tuple[int, ...]):
        """3D Adaptive Average Pooling layer
        Args:
            output_size: size of the output
        """
        super().__init__(
            output_size=output_size,
            spatial_ndim=3,
            func=jnp.mean,
        )


class AdaptiveMaxPool1D(AdaptivePoolND):
    def __init__(self, output_size: tuple[int, ...]):
        """1D Adaptive Max Pooling layer
        Args:
            output_size: size of the output
        """
        super().__init__(
            output_size=output_size,
            spatial_ndim=1,
            func=jnp.max,
        )


class AdaptiveMaxPool2D(AdaptivePoolND):
    def __init__(self, output_size: tuple[int, ...]):
        """2D Adaptive Max Pooling layer
        Args:
            output_size: size of the output
        """
        super().__init__(
            output_size=output_size,
            spatial_ndim=2,
            func=jnp.max,
        )


class AdaptiveMaxPool3D(AdaptivePoolND):
    def __init__(self, output_size: tuple[int, ...]):
        """3D Adaptive Max Pooling layer
        Args:
            output_size: size of the output
        """
        super().__init__(
            output_size=output_size,
            spatial_ndim=3,
            func=jnp.max,
        )
