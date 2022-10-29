from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import kernex as kex
import pytreeclass as pytc

from serket.nn.utils import (
    _check_and_return,
    _check_and_return_kernel,
    _check_and_return_strides,
    _check_spatial_in_shape,
)

# Based on colab hardware benchmarks `kernex` seems to
# be faster on CPU and on par with JAX on GPU.


@pytc.treeclass
class GeneralPoolND:

    kernel_size: tuple[int, ...] | int = pytc.nondiff_field()
    strides: tuple[int, ...] | int = pytc.nondiff_field()
    padding: tuple[tuple[int, int], ...] | int | str = pytc.nondiff_field()

    def __init__(
        self,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
        ndim: int = 1,
        func: Callable = None,
    ):
        """Apply pooling to the input with function `func` applied to the kernel.

        Args:
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel
            ndim: number of dimensions
            func: function to apply to the kernel
        """
        self.kernel_size = _check_and_return_kernel(kernel_size, ndim)
        self.strides = _check_and_return_strides(strides, ndim)
        self.padding = padding
        self.ndim = ndim

        @jax.jit
        @jax.vmap
        @kex.kmap(self.kernel_size, self.strides, self.padding)
        def _maxpoolnd(x):
            return func(x)

        self._func = _maxpoolnd

    @_check_spatial_in_shape
    def __call__(self, x, **kwargs):
        return self._func(x)


@pytc.treeclass
class LPPoolND(GeneralPoolND):
    def __init__(
        self,
        norm_type: float,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = None,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
        ndim: int = 1,
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides or kernel_size,
            padding=padding,
            ndim=ndim,
            func=lambda x: jnp.sum(x**norm_type) ** (1 / norm_type),
        )


@pytc.treeclass
class GlobalPoolND:
    def __init__(self, keepdims: bool = True, ndim: int = 1, func: Callable = jnp.mean):
        self.keepdims = keepdims
        self.ndim = ndim
        self.func = func

    @_check_spatial_in_shape
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        axes = tuple(range(1, self.ndim + 1))
        return self.func(x, axis=axes, keepdims=self.keepdims)


@pytc.treeclass
class AdaptivePoolND:

    output_size: tuple[int, ...] = pytc.nondiff_field()

    def __init__(
        self,
        output_size: tuple[int, ...],
        *,
        ndim: int = 1,
        func: Callable = None,
    ):
        """Apply pooling to the input with function `func` applied to the kernel.


        Args:
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel
            ndim: number of dimensions
            func: function to apply to the kernel

        Note:
            this is different from the PyTorch implementation
            the strides and kernel_size are calculated from the output_size as follows:
            * stride_i = (input_size_i//output_size_i)
            * kernel_size_i = input_size_i - (output_size_i-1)*stride_i
            * padding_i = "valid"
        """
        self.output_size = _check_and_return(output_size, ndim, "output_size")
        self.ndim = ndim
        self.func = func

    @_check_spatial_in_shape
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
    def __init__(self, output_size: tuple[int, ...], ndim: int):
        """Concatenate AdaptiveAvgPool1D and AdaptiveMaxPool1D
        See: https://github.com/fastai/fastai/blob/master/fastai/layers.py#L110
        """
        self.ndim = ndim
        self.avg_pool = AdaptivePoolND(output_size, func=jnp.mean, ndim=ndim)
        self.max_pool = AdaptivePoolND(output_size, func=jnp.max, ndim=ndim)

    @_check_spatial_in_shape
    def __call__(self, x, **kwargs):
        return jnp.concatenate([self.max_pool(x), self.avg_pool(x)], axis=0)


@pytc.treeclass
class MaxPool1D(GeneralPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=1, func=jnp.max)


# write all the other pooling layers here
@pytc.treeclass
class MaxPool2D(GeneralPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=2, func=jnp.max)


@pytc.treeclass
class MaxPool3D(GeneralPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=3, func=jnp.max)


@pytc.treeclass
class AvgPool1D(GeneralPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=1, func=jnp.mean)


@pytc.treeclass
class AvgPool2D(GeneralPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=2, func=jnp.mean)


@pytc.treeclass
class AvgPool3D(GeneralPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=3, func=jnp.mean)


@pytc.treeclass
class LPPool1D(LPPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=1)


@pytc.treeclass
class LPPool2D(LPPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=2)


@pytc.treeclass
class LPPool3D(LPPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=3)


@pytc.treeclass
class GlobalAvgPool1D(GlobalPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=1, func=jnp.mean)


@pytc.treeclass
class GlobalAvgPool2D(GlobalPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=2, func=jnp.mean)


@pytc.treeclass
class GlobalAvgPool3D(GlobalPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=3, func=jnp.mean)


@pytc.treeclass
class GlobalMaxPool1D(GlobalPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=1, func=jnp.max)


@pytc.treeclass
class GlobalMaxPool2D(GlobalPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=2, func=jnp.max)


@pytc.treeclass
class GlobalMaxPool3D(GlobalPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=3, func=jnp.max)


@pytc.treeclass
class AdaptiveAvgPool1D(AdaptivePoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=1, func=jnp.mean)


@pytc.treeclass
class AdaptiveAvgPool2D(AdaptivePoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=2, func=jnp.mean)


@pytc.treeclass
class AdaptiveAvgPool3D(AdaptivePoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=3, func=jnp.mean)


@pytc.treeclass
class AdaptiveMaxPool1D(AdaptivePoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=1, func=jnp.max)


@pytc.treeclass
class AdaptiveMaxPool2D(AdaptivePoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=2, func=jnp.max)


@pytc.treeclass
class AdaptiveMaxPool3D(AdaptivePoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=3, func=jnp.max)


@pytc.treeclass
class AdaptiveConcatPool1D(AdaptiveConcatPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=1)


@pytc.treeclass
class AdaptiveConcatPool2D(AdaptiveConcatPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=2)


@pytc.treeclass
class AdaptiveConcatPool3D(AdaptiveConcatPoolND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=3)
