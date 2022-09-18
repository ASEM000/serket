from __future__ import annotations

import jax
import jax.numpy as jnp
import kernex as kex
import pytreeclass as pytc


@pytc.treeclass
class MaxPoolND:

    kernel_size: tuple[int, ...] | int = pytc.nondiff_field()
    strides: tuple[int, ...] | int = pytc.nondiff_field()
    padding: tuple[tuple[int, int], ...] | int | str = pytc.nondiff_field()

    def __init__(
        self,
        *,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        padding: tuple[tuple[int, int], ...] | str = "valid",
        ndim: int = 1,
    ):

        self.kernel_size = (
            (kernel_size,) * ndim if isinstance(kernel_size, int) else kernel_size
        )
        self.strides = strides
        self.padding = padding

    def __call__(self, x):

        # `vmap` on channel dimension
        @jax.vmap
        @kex.kmap(self.kernel_size, self.strides, self.padding)
        def _maxpoolnd(x):
            return jnp.max(x)

        return _maxpoolnd(x)


@pytc.treeclass
class MaxPool1D(MaxPoolND):
    def __init__(
        self,
        *,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size, strides=strides, padding=padding, ndim=1
        )


@pytc.treeclass
class MaxPool2D(MaxPoolND):
    def __init__(
        self,
        *,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size, strides=strides, padding=padding, ndim=2
        )


@pytc.treeclass
class MaxPool3D(MaxPoolND):
    def __init__(
        self,
        *,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size, strides=strides, padding=padding, ndim=3
        )


@pytc.treeclass
class AvgPoolND:

    kernel_size: tuple[int, ...] | int = pytc.nondiff_field()
    strides: tuple[int, ...] | int = pytc.nondiff_field()
    padding: tuple[tuple[int, int], ...] | int | str = pytc.nondiff_field()

    def __init__(
        self,
        *,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        padding: tuple[tuple[int, int], ...] | str = "valid",
        ndim: int = 1,
    ):

        self.kernel_size = (
            (kernel_size,) * ndim if isinstance(kernel_size, int) else kernel_size
        )
        self.strides = strides
        self.padding = padding

    def __call__(self, x):

        # `vmap` on channel dimension
        @jax.vmap
        @kex.kmap(self.kernel_size, self.strides, self.padding)
        def _avgpoolnd(x):
            return jnp.mean(x)

        return _avgpoolnd(x)


@pytc.treeclass
class AvgPool1D(AvgPoolND):
    def __init__(
        self,
        *,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size, strides=strides, padding=padding, ndim=1
        )


@pytc.treeclass
class AvgPool2D(AvgPoolND):
    def __init__(
        self,
        *,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size, strides=strides, padding=padding, ndim=2
        )


@pytc.treeclass
class AvgPool3D(AvgPoolND):
    def __init__(
        self,
        *,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size, strides=strides, padding=padding, ndim=3
        )
