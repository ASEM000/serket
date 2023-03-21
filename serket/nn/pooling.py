from __future__ import annotations

import functools as ft
from typing import Callable

import jax
import jax.numpy as jnp
import kernex as kex
import pytreeclass as pytc

from serket.nn.callbacks import validate_spatial_in_shape
from serket.nn.utils import canonicalize

# Based on colab hardware benchmarks `kernex` seems to
# be faster on CPU and on par with JAX on GPU.


@pytc.treeclass
class GeneralPoolND:
    kernel_size: tuple[int, ...] | int = pytc.field(callbacks=[pytc.freeze])
    strides: tuple[int, ...] | int = pytc.field(callbacks=[pytc.freeze])
    padding: tuple[tuple[int, int], ...] | int | str = pytc.field(
        callbacks=[pytc.freeze]
    )

    def __init__(
        self,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
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
        self.padding = padding
        self.spatial_ndim = spatial_ndim

        @jax.jit
        @jax.vmap
        @kex.kmap(self.kernel_size, self.strides, self.padding)
        def _poolnd(x):
            return func(x)

        self.func = _poolnd

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x, **kwargs):
        return self.func(x)


@pytc.treeclass
class LPPoolND(GeneralPoolND):
    def __init__(
        self,
        norm_type: float,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = None,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
        spatial_ndim: int = 1,
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides or kernel_size,
            padding=padding,
            spatial_ndim=spatial_ndim,
            func=lambda x: jnp.sum(x**norm_type) ** (1 / norm_type),
        )


@pytc.treeclass
class GlobalPoolND:
    def __init__(
        self, keepdims: bool = True, spatial_ndim: int = 1, func: Callable = jnp.mean
    ):
        self.keepdims = keepdims
        self.spatial_ndim = spatial_ndim
        self.func = func

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        axes = tuple(range(1, self.spatial_ndim + 1))  # reduce spatial dimensions
        return self.func(x, axis=axes, keepdims=self.keepdims)


@pytc.treeclass
class AdaptivePoolND:
    output_size: tuple[int, ...] = pytc.field(callbacks=[pytc.freeze])

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
            this is different from the PyTorch implementation
            the strides and kernel_size are calculated from the output_size as follows:
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


@pytc.treeclass
class AdaptiveConcatPoolND:
    def __init__(self, output_size: tuple[int, ...], spatial_ndim: int):
        """Concatenate AdaptiveAvgPool1D and AdaptiveMaxPool1D
        See: https://github.com/fastai/fastai/blob/master/fastai/layers.py#L110
        """
        self.spatial_ndim = spatial_ndim
        self.avg_pool = AdaptivePoolND(
            output_size, func=jnp.mean, spatial_ndim=spatial_ndim
        )
        self.max_pool = AdaptivePoolND(
            output_size, func=jnp.max, spatial_ndim=spatial_ndim
        )

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x, **k):
        return jnp.concatenate([self.max_pool(x), self.avg_pool(x)], axis=0)


@pytc.treeclass
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


@pytc.treeclass
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


@pytc.treeclass
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


@pytc.treeclass
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


@pytc.treeclass
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


@pytc.treeclass
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


@pytc.treeclass
class LPPool1D(LPPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=1)


@pytc.treeclass
class LPPool2D(LPPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=2)


@pytc.treeclass
class LPPool3D(LPPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=3)


@pytc.treeclass
class GlobalAvgPool1D(GlobalPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=1, func=jnp.mean)


@pytc.treeclass
class GlobalAvgPool2D(GlobalPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=2, func=jnp.mean)


@pytc.treeclass
class GlobalAvgPool3D(GlobalPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=3, func=jnp.mean)


@pytc.treeclass
class GlobalMaxPool1D(GlobalPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=1, func=jnp.max)


@pytc.treeclass
class GlobalMaxPool2D(GlobalPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=2, func=jnp.max)


@pytc.treeclass
class GlobalMaxPool3D(GlobalPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=3, func=jnp.max)


@pytc.treeclass
class AdaptiveAvgPool1D(AdaptivePoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=1, func=jnp.mean)


@pytc.treeclass
class AdaptiveAvgPool2D(AdaptivePoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=2, func=jnp.mean)


@pytc.treeclass
class AdaptiveAvgPool3D(AdaptivePoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=3, func=jnp.mean)


@pytc.treeclass
class AdaptiveMaxPool1D(AdaptivePoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=1, func=jnp.max)


@pytc.treeclass
class AdaptiveMaxPool2D(AdaptivePoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=2, func=jnp.max)


@pytc.treeclass
class AdaptiveMaxPool3D(AdaptivePoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=3, func=jnp.max)


@pytc.treeclass
class AdaptiveConcatPool1D(AdaptiveConcatPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=1)


@pytc.treeclass
class AdaptiveConcatPool2D(AdaptiveConcatPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=2)


@pytc.treeclass
class AdaptiveConcatPool3D(AdaptiveConcatPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=3)
