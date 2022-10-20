from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import kernex as kex
import pytreeclass as pytc

from serket.nn.utils import _check_and_return_kernel

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
        self.strides = strides
        self.padding = padding

        @jax.jit
        @jax.vmap
        @kex.kmap(self.kernel_size, self.strides, self.padding)
        def _maxpoolnd(x):
            return func(x)

        self._func = _maxpoolnd

    def __call__(self, x, **kwargs):
        return self._func(x)


@pytc.treeclass
class MaxPool1D(GeneralPoolND):
    def __init__(
        self,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = None,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides or kernel_size,
            padding=padding,
            ndim=1,
            func=jnp.max,
        )


@pytc.treeclass
class MaxPool2D(GeneralPoolND):
    def __init__(
        self,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = None,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides or kernel_size,
            padding=padding,
            ndim=2,
            func=jnp.max,
        )


@pytc.treeclass
class MaxPool3D(GeneralPoolND):
    def __init__(
        self,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = None,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides or kernel_size,
            padding=padding,
            ndim=3,
            func=jnp.max,
        )


@pytc.treeclass
class AvgPool1D(GeneralPoolND):
    def __init__(
        self,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = None,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides or kernel_size,
            padding=padding,
            ndim=1,
            func=jnp.mean,
        )


@pytc.treeclass
class AvgPool2D(GeneralPoolND):
    def __init__(
        self,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = None,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides or kernel_size,
            padding=padding,
            ndim=2,
            func=jnp.mean,
        )


@pytc.treeclass
class AvgPool3D(GeneralPoolND):
    def __init__(
        self,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = None,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides or kernel_size,
            padding=padding,
            ndim=3,
            func=jnp.mean,
        )


@pytc.treeclass
class LPPool1D(GeneralPoolND):
    def __init__(
        self,
        norm_type: float,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = None,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides or kernel_size,
            padding=padding,
            ndim=1,
            func=lambda x: jnp.sum(x**norm_type) ** (1 / norm_type),
        )


@pytc.treeclass
class LPPool2D(GeneralPoolND):
    def __init__(
        self,
        norm_type: float,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = None,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides or kernel_size,
            padding=padding,
            ndim=2,
            func=lambda x: jnp.sum(x**norm_type) ** (1 / norm_type),
        )


@pytc.treeclass
class LPPool3D(GeneralPoolND):
    def __init__(
        self,
        norm_type: float,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = None,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            norm_type=norm_type,
            kernel_size=kernel_size,
            strides=strides or kernel_size,
            padding=padding,
            ndim=3,
            func=lambda x: jnp.sum(x**norm_type) ** (1 / norm_type),
        )


@pytc.treeclass
class GlobalAvgPool1D:
    keepdims: bool = pytc.nondiff_field(default=True)
    """ Average last channels """

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 2, "Input must be 2D."
        return jnp.mean(x, axis=1, keepdims=self.keepdims)


@pytc.treeclass
class GlobalAvgPool2D:
    keepdims: bool = pytc.nondiff_field(default=True)
    """ Average last channels """

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 3, "Input must be 3D."
        return jnp.mean(x, axis=(1, 2), keepdims=self.keepdims)


@pytc.treeclass
class GlobalAvgPool3D:
    keepdims: bool = pytc.nondiff_field(default=True)
    """ Average last channels """

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 4, "Input must be 4D."
        return jnp.mean(x, axis=(1, 2, 3), keepdims=self.keepdims)


@pytc.treeclass
class GlobalMaxPool1D:
    keepdims: bool = pytc.nondiff_field(default=True)
    """ Average last channels """

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 2, "Input must be 2D."
        return jnp.max(x, axis=1, keepdims=self.keepdims)


@pytc.treeclass
class GlobalMaxPool2D:
    keepdims: bool = pytc.nondiff_field(default=True)
    """ Average last channels """

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 3, "Input must be 3D."
        return jnp.max(x, axis=(1, 2), keepdims=self.keepdims)


@pytc.treeclass
class GlobalMaxPool3D:
    keepdims: bool = pytc.nondiff_field(default=True)
    """ Get maximum last channels """

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 4, "Input must be 4D."
        return jnp.max(x, axis=(1, 2, 3), keepdims=self.keepdims)
