from __future__ import annotations

import jax
import jax.numpy as jnp
import kernex as kex
import pytreeclass as pytc

from serket.nn.utils import _check_and_return_kernel

# Based on colab hardware benchmarks `kernex` seems to
# be faster on CPU and on par with JAX on GPU.


@pytc.treeclass
class MaxPoolND:

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
    ):
        """Apply max pooling to the second dimension of the input.

        Args:
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel
            ndim: number of dimensions
        """
        self.kernel_size = _check_and_return_kernel(kernel_size, ndim)
        self.strides = strides
        self.padding = padding

        @jax.jit
        @jax.vmap
        @kex.kmap(self.kernel_size, self.strides, self.padding)
        def _maxpoolnd(x):
            return jnp.max(x)

        self._func = _maxpoolnd

    def __call__(self, x, **kwargs):
        return self._func(x)


@pytc.treeclass
class MaxPool1D(MaxPoolND):
    def __init__(
        self,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size, strides=strides, padding=padding, ndim=1
        )


@pytc.treeclass
class MaxPool2D(MaxPoolND):
    def __init__(
        self,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size, strides=strides, padding=padding, ndim=2
        )


@pytc.treeclass
class MaxPool3D(MaxPoolND):
    def __init__(
        self,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        *,
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
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
        ndim: int = 1,
    ):
        """Apply average pooling to the second dimension of the input.

        Args:
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel
            ndim: number of dimensions
        """
        self.kernel_size = _check_and_return_kernel(kernel_size, ndim)
        self.strides = strides
        self.padding = padding

        @jax.jit
        @jax.vmap
        @kex.kmap(self.kernel_size, self.strides, self.padding)
        def _avgpoolnd(x):
            return jnp.mean(x)

        self._func = _avgpoolnd

    def __call__(self, x, **kwargs):
        return self._func(x)


@pytc.treeclass
class AvgPool1D(AvgPoolND):
    def __init__(
        self,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size, strides=strides, padding=padding, ndim=1
        )


@pytc.treeclass
class AvgPool2D(AvgPoolND):
    def __init__(
        self,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size, strides=strides, padding=padding, ndim=2
        )


@pytc.treeclass
class AvgPool3D(AvgPoolND):
    def __init__(
        self,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            kernel_size=kernel_size, strides=strides, padding=padding, ndim=3
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


@pytc.treeclass
class LPPoolND:
    norm_type: float = pytc.nondiff_field()
    kernel_size: tuple[int, ...] | int = pytc.nondiff_field()
    padding: tuple[tuple[int, int], ...] | int | str = pytc.nondiff_field()

    def __init__(
        self,
        norm_type: float,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
        ndim: int = 1,
    ):
        """Apply  Lp pooling

        See: https://pytorch.org/docs/stable/generated/torch.nn.LPPool1d.html#torch.nn.LPPool1d

        Args:
            kernel_size: size of the kernel
            strides: strides of the kernel
            padding: padding of the kernel
            ndim: number of dimensions
        """
        if not isinstance(norm_type, float):
            raise TypeError("norm_type must be a float")

        self.kernel_size = _check_and_return_kernel(kernel_size, ndim)
        self.strides = self.kernel_size
        self.padding = padding

        @jax.jit
        @jax.vmap
        @kex.kmap(self.kernel_size, self.strides, self.padding)
        def _lppolnd(x):
            return jnp.sum(x**norm_type) ** (1 / norm_type)

        self._func = _lppolnd

    def __call__(self, x, **kwargs):
        return self._func(x)


@pytc.treeclass
class LPPool1D(LPPoolND):
    def __init__(
        self,
        norm_type: float,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            norm_type=norm_type,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            ndim=1,
        )


@pytc.treeclass
class LPPool2D(LPPoolND):
    def __init__(
        self,
        norm_type: float,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            norm_type=norm_type,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            ndim=2,
        )


@pytc.treeclass
class LPPool3D(LPPoolND):
    def __init__(
        self,
        norm_type: float,
        kernel_size: tuple[int, ...] | int,
        strides: tuple[int, ...] | int = 1,
        *,
        padding: tuple[tuple[int, int], ...] | str = "valid",
    ):
        super().__init__(
            norm_type=norm_type,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            ndim=3,
        )
