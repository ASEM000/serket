from __future__ import annotations

import jax
import jax.numpy as jnp
import kernex as kex
import pytreeclass as pytc


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
    size: int | tuple[int, int]
    method: str = pytc.nondiff_field(default="nearest")
    antialias: bool = pytc.nondiff_field(default=True)
    ndim: int = pytc.nondiff_field(default=1, repr=False)

    """
    Resize an image to a given size using a given interpolation method.
    
    Args:
        size (int | tuple[int, int], optional): the size of the output. Defaults to None.
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

    def __post_init__(self):
        if isinstance(self.size, (int)):
            self.size = (self.size,) * self.ndim
        elif isinstance(self.size, tuple):
            assert (
                len(self.size) == self.ndim
            ), f"size must be a tuple of length {self.ndim} or an int."

        else:
            raise ValueError("size must be a tuple or an int")

        assert self.method.lower() in [
            "nearest",
            "linear",
            "bilinear",
            "trilinear",
            "triangle",
            "cubic",
            "bicubic",
            "tricubic",
            "lanczos3",
            "lanczos5",
        ], (
            "method must be one of 'nearest', 'linear', 'bilinear', 'trilinear', 'triangle',"
            f"'cubic', 'bicubic', 'tricubic', 'lanczos3', 'lanczos5'. Found {self.method}"
        )


@pytc.treeclass
class Resize1D(ResizeND):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ndim=1)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 2, "Input must be 2D [Channel, Spatial]."
        return jax.image.resize(
            x, (x.shape[0], *self.size), method=self.method, antialias=self.antialias
        )


@pytc.treeclass
class Resize2D(ResizeND):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ndim=2)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 3, "Input must be 3D [Channel, Height, Width]."
        return jax.image.resize(
            x, (x.shape[0], *self.size), method=self.method, antialias=self.antialias
        )


@pytc.treeclass
class Resize3D(ResizeND):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ndim=3)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 4, "Input must be 4D [Channel, Depth, Height, Width]."
        return jax.image.resize(
            x, (x.shape[0], *self.size), method=self.method, antialias=self.antialias
        )
