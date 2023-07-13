# Copyright 2023 Serket authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import abc
import functools as ft
from typing import Literal

import jax

import serket as sk
from serket.nn.utils import canonicalize, validate_spatial_ndim

MethodKind = Literal["nearest", "linear", "cubic", "lanczos3", "lanczos5"]


class ResizeND(sk.TreeClass):
    """
    Resize an image to a given size using a given interpolation method.

    Args:
        size: the size of the output. if size is None, the output size is
            calculated as input size * scale
        method: the method of interpolation. Defaults to "nearest". choices are:
        - "nearest": Nearest neighbor interpolation. The values of antialias
            and precision are ignored.
        - "linear", "bilinear", "trilinear", "triangle": Linear interpolation.
            If antialias is True, uses a triangular filter when downsampling.
        - "cubic", "bicubic", "tricubic": Cubic interpolation, using the Keys
            cubic kernel.
        - "lanczos3": Lanczos resampling, using a kernel of radius 3.
        - "lanczos5": Lanczos resampling, using a kernel of radius 5.
        antialias: whether to use antialiasing. Defaults to True.
        antialias: whether to use antialiasing. Defaults to True.
    """

    def __init__(
        self,
        size: int | tuple[int, ...],
        method: MethodKind = "nearest",
        antialias: bool = True,
    ):
        self.size = canonicalize(size, self.spatial_ndim, name="size")
        self.method = method
        self.antialias = antialias

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return jax.image.resize(
            x,
            shape=(x.shape[0], *self.size),
            method=self.method,
            antialias=self.antialias,
        )

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        ...


class UpsampleND(sk.TreeClass):
    def __init__(
        self,
        scale: int | tuple[int, ...] = 1,
        method: MethodKind = "nearest",
    ):
        # the difference between this and ResizeND is that UpsamplingND
        # use scale instead of size
        # assert types
        self.scale = canonicalize(scale, self.spatial_ndim, name="scale")
        self.method = method

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        resized_shape = tuple(s * x.shape[i + 1] for i, s in enumerate(self.scale))
        return jax.image.resize(
            x,
            shape=(x.shape[0], *resized_shape),
            method=self.method,
        )

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        ...


class Resize1D(ResizeND):
    """Resize a 1D input to a given size using a given interpolation method.

    Args:
        size: the size of the output. if size is None, the output size is
            calculated as input size * scale
        method: the method of interpolation. Defaults to "nearest". choices are:
        - "nearest": Nearest neighbor interpolation. The values of antialias
            and precision are ignored.
        - "linear", "bilinear", "trilinear", "triangle": Linear interpolation.
            If antialias is True, uses a triangular filter when downsampling.
        - "cubic", "bicubic", "tricubic": Cubic interpolation, using the Keys
            cubic kernel.
        - "lanczos3": Lanczos resampling, using a kernel of radius 3.
        - "lanczos5": Lanczos resampling, using a kernel of radius 5.
        antialias: whether to use antialiasing. Defaults to True.
    """

    def __init__(
        self,
        size: int | tuple[int, ...],
        method: MethodKind = "nearest",
        antialias=True,
    ):
        super().__init__(size=size, method=method, antialias=antialias)

    @property
    def spatial_ndim(self) -> int:
        return 1


class Resize2D(ResizeND):
    """Resize a 2D input to a given size using a given interpolation method.

    Args:
        size: the size of the output. if size is None, the output size is
            calculated as input size * scale
        method: the method of interpolation. Defaults to "nearest". choices are:
        - "nearest": Nearest neighbor interpolation. The values of antialias
            and precision are ignored.
        - "linear", "bilinear", "trilinear", "triangle": Linear interpolation.
            If antialias is True, uses a triangular filter when downsampling.
        - "cubic", "bicubic", "tricubic": Cubic interpolation, using the Keys
            cubic kernel.
        - "lanczos3": Lanczos resampling, using a kernel of radius 3.
        - "lanczos5": Lanczos resampling, using a kernel of radius 5.
        antialias: whether to use antialiasing. Defaults to True.
    """

    def __init__(
        self,
        size: int | tuple[int, ...],
        method: MethodKind = "nearest",
        antialias=True,
    ):
        super().__init__(size=size, method=method, antialias=antialias)

    @property
    def spatial_ndim(self) -> int:
        return 2


class Resize3D(ResizeND):
    """Resize a 3D input to a given size using a given interpolation method.

    Args:
        size: the size of the output. if size is None, the output size is
            calculated as input size * scale
        method: the method of interpolation. Defaults to "nearest". choices are:
        - "nearest": Nearest neighbor interpolation. The values of antialias
            and precision are ignored.
        - "linear", "bilinear", "trilinear", "triangle": Linear interpolation.
            If antialias is True, uses a triangular filter when downsampling.
        - "cubic", "bicubic", "tricubic": Cubic interpolation, using the Keys
            cubic kernel.
        - "lanczos3": Lanczos resampling, using a kernel of radius 3.
        - "lanczos5": Lanczos resampling, using a kernel of radius 5.
        antialias: whether to use antialiasing. Defaults to True.
    """

    def __init__(
        self,
        size: int | tuple[int, ...],
        method: MethodKind = "nearest",
        antialias=True,
    ):
        super().__init__(size=size, method=method, antialias=antialias)

    @property
    def spatial_ndim(self) -> int:
        return 3


class Upsample1D(UpsampleND):
    """Upsample a 1D input to a given size using a given interpolation method.

    Args:
        scale: the scale of the output.
        method: the method of interpolation. Defaults to "nearest".
    """

    def __init__(self, scale: int, method: str = "nearest"):
        super().__init__(scale=scale, method=method)

    @property
    def spatial_ndim(self) -> int:
        return 1


class Upsample2D(UpsampleND):
    """Upsample a 2D input to a given size using a given interpolation method.

    Args:
        scale: the scale of the output.
        method: the method of interpolation. Defaults to "nearest".
    """

    def __init__(self, scale: int | tuple[int, int], method: str = "nearest"):
        super().__init__(scale=scale, method=method)

    @property
    def spatial_ndim(self) -> int:
        return 2


class Upsample3D(UpsampleND):
    """Upsample a 1D input to a given size using a given interpolation method.

    Args:
        scale: the scale of the output.
        method: the method of interpolation. Defaults to "nearest".
    """

    def __init__(self, scale: int | tuple[int, int, int], method: str = "nearest"):
        super().__init__(scale=scale, method=method)

    @property
    def spatial_ndim(self) -> int:
        return 3
