from __future__ import annotations

import functools as ft
from typing import Literal

import jax
import pytreeclass as pytc

from serket.nn.callbacks import isinstance_factory, validate_spatial_in_shape
from serket.nn.utils import canonicalize

MethodKind = Literal["nearest", "linear", "cubic", "lanczos3", "lanczos5"]


class ResizeND(pytc.TreeClass):
    size: int | tuple[int, ...]
    method: MethodKind
    antialias: bool

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
        self.size = canonicalize(size, spatial_ndim, "size")
        self.method = method
        self.antialias = antialias
        self.spatial_ndim = spatial_ndim

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.image.resize(
            x,
            shape=(x.shape[0], *self.size),
            method=self.method,
            antialias=self.antialias,
        )


class UpsampleND(pytc.TreeClass):
    scale: int | tuple[int, ...] = pytc.field(
        callbacks=[isinstance_factory((int, tuple))]
    )
    method: MethodKind

    def __init__(
        self,
        scale: int | tuple[int, ...] = 1,
        method: str = "nearest",
        spatial_ndim: int = 1,
    ):
        # the difference between this and ResizeND is that UpsamplingND
        # use scale instead of size
        # assert types
        self.scale = canonicalize(scale, spatial_ndim, "scale")
        self.method = method
        self.spatial_ndim = spatial_ndim

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        resized_shape = tuple(s * x.shape[i + 1] for i, s in enumerate(self.scale))
        return jax.image.resize(
            x,
            shape=(x.shape[0], *resized_shape),
            method=self.method,
        )


class Resize1D(ResizeND):
    def __init__(
        self,
        size: int | tuple[int, ...],
        method: MethodKind = "nearest",
        antialias=True,
    ):
        """Resize a 1D input to a given size using a given interpolation method.

        Args:
            size: the size of the output.
            method: the method of interpolation. Defaults to "nearest".
            antialias: whether to use antialiasing. Defaults to True.

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
        super().__init__(size=size, method=method, antialias=antialias, spatial_ndim=1)


class Resize2D(ResizeND):
    def __init__(
        self,
        size: int | tuple[int, ...],
        method: MethodKind = "nearest",
        antialias=True,
    ):
        """Resize a 2D input to a given size using a given interpolation method.

        Args:
            size: the size of the output.
            method: the method of interpolation. Defaults to "nearest".
            antialias: whether to use antialiasing. Defaults to True.

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
        super().__init__(size=size, method=method, antialias=antialias, spatial_ndim=2)


class Resize3D(ResizeND):
    def __init__(
        self,
        size: int | tuple[int, ...],
        method: MethodKind = "nearest",
        antialias=True,
    ):
        """Resize a 3D input to a given size using a given interpolation method.

        Args:
            size: the size of the output.
            method: the method of interpolation. Defaults to "nearest".
            antialias: whether to use antialiasing. Defaults to True.

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
        super().__init__(size=size, method=method, antialias=antialias, spatial_ndim=3)


class Upsample1D(UpsampleND):
    def __init__(self, scale: int, method: str = "nearest"):
        """Upsample a 1D input to a given size using a given interpolation method.

        Args:
            scale: the scale of the output.
            method: the method of interpolation. Defaults to "nearest".
        """
        super().__init__(scale=scale, method=method, spatial_ndim=1)


class Upsample2D(UpsampleND):
    def __init__(self, scale: int | tuple[int, int], method: str = "nearest"):
        """Upsample a 2D input to a given size using a given interpolation method.

        Args:
            scale: the scale of the output.
            method: the method of interpolation. Defaults to "nearest".
        """
        super().__init__(scale=scale, method=method, spatial_ndim=2)


class Upsample3D(UpsampleND):
    def __init__(self, scale: int | tuple[int, int, int], method: str = "nearest"):
        """Upsample a 1D input to a given size using a given interpolation method.

        Args:
            scale: the scale of the output.
            method: the method of interpolation. Defaults to "nearest".
        """
        super().__init__(scale=scale, method=method, spatial_ndim=3)
