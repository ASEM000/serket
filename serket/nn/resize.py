from __future__ import annotations

import jax
import jax.numpy as jnp
import kernex as kex
import pytreeclass as pytc

from serket.nn.utils import _check_and_return


@pytc.treeclass
class Repeat1D:
    """repeats input along axis 1"""

    scale: int = pytc.nondiff_field(default=1)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        @kex.kmap(kernel_size=(-1, -1), strides=(1, 1), padding="valid")
        def _repeat(x):
            return x.repeat(self.scale, axis=1)

        return jnp.squeeze(_repeat(x), axis=(1, 2))


@pytc.treeclass
class Repeat2D:
    """repeats input along axes 1,2"""

    scale: int = pytc.nondiff_field(default=1)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        @kex.kmap(kernel_size=(-1, -1, -1), strides=(1, 1, 1), padding="valid")
        def _repeat(x):
            return x.repeat(self.scale, axis=2).repeat(self.scale, axis=1)

        return jnp.squeeze(_repeat(x), axis=(1, 2, 3))


@pytc.treeclass
class Repeat3D:
    """repeats input along axes 1,2,3"""

    scale: int = pytc.nondiff_field(default=1)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        @kex.kmap(kernel_size=(-1, -1, -1, -1), strides=(1, 1, 1, 1), padding="valid")
        def _repeat(x):
            return (
                x.repeat(self.scale, axis=3)
                .repeat(self.scale, axis=2)
                .repeat(self.scale, axis=1)
            )

        return jnp.squeeze(_repeat(x), axis=(1, 2, 3, 4))


@pytc.treeclass
class ResizeND:
    size: int | tuple[int, ...] = pytc.nondiff_field()
    method: str = pytc.nondiff_field()
    antialias: bool = pytc.nondiff_field()

    """
    Resize an image to a given size using a given interpolation method.
    
    Args:
        size (int | tuple[int, int], optional): the size of the output.
        method (str, optional): the method of interpolation. Defaults to "nearest".
        antialias (bool, optional): whether to use antialiasing. Defaults to True.

    Note:
        - if size is None, the output size is calculated as input size * scale
        - interpolation methods
            "nearest" :
                Nearest neighbor interpolation. The values of antialias and precision are ignored.

            "linear", "bilinear", "trilinear", "triangle" :
                Linear interpolation. If antialias is True, uses a triangular filter when downsampling.

            "cubic", "bicubic", "tricubic" :
                Cubic interpolation, using the Keys cubic kernel.

            "lanczos3" :
                Lanczos resampling, using a kernel of radius 3.

            "lanczos5"
                Lanczos resampling, using a kernel of radius 5.
    """

    def __init__(self, size, method="nearest", antialias=True, ndim=1):
        self.size = _check_and_return(size, ndim, "size")
        self.method = method
        self.antialias = antialias
        self.ndim = ndim

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == self.ndim + 1, f"input must be {self.ndim}D"

        return jax.image.resize(
            x,
            shape=(x.shape[0], *self.size),
            method=self.method,
            antialias=self.antialias,
        )


@pytc.treeclass
class Resize1D(ResizeND):
    def __init__(self, size, method="nearest", antialias=True):
        super().__init__(size, method, antialias, ndim=1)


@pytc.treeclass
class Resize2D(ResizeND):
    def __init__(self, size, method="nearest", antialias=True):
        super().__init__(size, method, antialias, ndim=2)


@pytc.treeclass
class Resize3D(ResizeND):
    def __init__(self, size, method="nearest", antialias=True):
        super().__init__(size, method, antialias, ndim=3)


@pytc.treeclass
class UpsampleND:
    scale: int | tuple[int, ...] = pytc.nondiff_field(default=1)
    method: str = pytc.nondiff_field(default="nearest")

    def __init__(
        self, scale: int | tuple[int, ...], method: str = "nearest", ndim: int = 1
    ):
        # the difference between this and ResizeND is that UpsamplingND
        # use scale instead of size
        # assert types
        self.scale = _check_and_return(scale, ndim, "scale")
        self.method = method
        self.ndim = ndim

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        msg = f"Input must have {self.ndim+1} dimensions, got {x.ndim}."
        assert x.ndim == self.ndim + 1, msg

        resized_shape = tuple(s * x.shape[i + 1] for i, s in enumerate(self.scale))
        return jax.image.resize(
            x,
            shape=(x.shape[0], *resized_shape),
            method=self.method,
        )


@pytc.treeclass
class Upsample1D(UpsampleND):
    def __init__(self, scale: int | tuple[int, ...], method: str = "nearest"):
        super().__init__(scale, method, ndim=1)


@pytc.treeclass
class Upsample2D(UpsampleND):
    def __init__(self, scale: int | tuple[int, ...], method: str = "nearest"):
        super().__init__(scale, method, ndim=2)


@pytc.treeclass
class Upsample3D(UpsampleND):
    def __init__(self, scale: int | tuple[int, ...], method: str = "nearest"):
        super().__init__(scale, method, ndim=3)
