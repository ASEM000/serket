from __future__ import annotations

import jax
import jax.numpy as jnp
import kernex as kex
import pytreeclass as pytc

from serket.nn.utils import (
    _canonicalize,
    _canonicalize_positive_int,
    _check_spatial_in_shape,
)


def _recursive_repeat(x, scale, axis):
    if axis == 1:
        return x.repeat(scale, axis=axis)
    return _recursive_repeat(x.repeat(scale, axis=axis), scale, axis - 1)


@pytc.treeclass
class RepeatND:
    def __init__(self, scale: int = 1, spatial_ndim: int = 1):
        """repeats input along axes 1,2,3"""
        self.scale = _canonicalize_positive_int(scale, "scale")
        self.spatial_ndim = spatial_ndim

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        _check_spatial_in_shape(x, self.spatial_ndim)

        kernel_size = (-1,) * (self.spatial_ndim + 1)
        strides = (1,) * (self.spatial_ndim + 1)

        @kex.kmap(kernel_size=kernel_size, strides=strides, padding="valid")
        def _repeat(x):
            return _recursive_repeat(x, self.scale, self.spatial_ndim)

        return jnp.squeeze(_repeat(x), axis=tuple(range(1, self.spatial_ndim + 2)))


@pytc.treeclass
class ResizeND:
    size: int | tuple[int, ...] = pytc.field(callbacks=[pytc.freeze])
    method: str = pytc.field(callbacks=[pytc.freeze])
    antialias: bool = pytc.field(callbacks=[pytc.freeze])

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

    def __init__(self, size, method="nearest", antialias=True, spatial_ndim=1):
        self.size = _canonicalize(size, spatial_ndim, "size")
        self.method = method
        self.antialias = antialias
        self.spatial_ndim = spatial_ndim

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        _check_spatial_in_shape(x, self.spatial_ndim)
        assert x.ndim == self.spatial_ndim + 1, f"input must be {self.spatial_ndim}D"

        return jax.image.resize(
            x,
            shape=(x.shape[0], *self.size),
            method=self.method,
            antialias=self.antialias,
        )


@pytc.treeclass
class UpsampleND:
    scale: int | tuple[int, ...] = pytc.field(callbacks=[pytc.freeze], default=1)
    method: str = pytc.field(callbacks=[pytc.freeze], default="nearest")

    def __init__(
        self,
        scale: int | tuple[int, ...],
        method: str = "nearest",
        spatial_ndim: int = 1,
    ):
        # the difference between this and ResizeND is that UpsamplingND
        # use scale instead of size
        # assert types
        self.scale = _canonicalize(scale, spatial_ndim, "scale")
        self.method = method
        self.spatial_ndim = spatial_ndim

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        _check_spatial_in_shape(x, self.spatial_ndim)
        msg = f"Input must have {self.spatial_ndim+1} dimensions, got {x.ndim}."
        assert x.ndim == self.spatial_ndim + 1, msg

        resized_shape = tuple(s * x.shape[i + 1] for i, s in enumerate(self.scale))
        return jax.image.resize(
            x,
            shape=(x.shape[0], *resized_shape),
            method=self.method,
        )


@pytc.treeclass
class Repeat1D(RepeatND):
    def __init__(self, *a, **k):
        super().__init__(*a, spatial_ndim=1, **k)


@pytc.treeclass
class Repeat2D(RepeatND):
    def __init__(self, *a, **k):
        super().__init__(*a, spatial_ndim=2, **k)


@pytc.treeclass
class Repeat3D(RepeatND):
    def __init__(self, *a, **k):
        super().__init__(*a, spatial_ndim=3, **k)


@pytc.treeclass
class Resize1D(ResizeND):
    def __init__(self, *a, **k):
        super().__init__(*a, spatial_ndim=1, **k)


@pytc.treeclass
class Resize2D(ResizeND):
    def __init__(self, *a, **k):
        super().__init__(*a, spatial_ndim=2, **k)


@pytc.treeclass
class Resize3D(ResizeND):
    def __init__(self, *a, **k):
        super().__init__(*a, spatial_ndim=3, **k)


@pytc.treeclass
class Upsample1D(UpsampleND):
    def __init__(self, *a, **k):
        super().__init__(*a, spatial_ndim=1, **k)


@pytc.treeclass
class Upsample2D(UpsampleND):
    def __init__(self, *a, **k):
        super().__init__(*a, spatial_ndim=2, **k)


@pytc.treeclass
class Upsample3D(UpsampleND):
    def __init__(self, *a, **k):
        super().__init__(*a, spatial_ndim=3, **k)
