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

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc
from jax import lax

from serket.nn.utils import validate_spatial_in_shape


def adjust_contrast_nd(x: jax.Array, contrast_factor: float):
    """Adjusts the contrast of an image by scaling the pixel values by a factor."""
    μ = jnp.mean(x, axis=tuple(range(1, x.ndim)), keepdims=True)
    return (contrast_factor * (x - μ) + μ).astype(x.dtype)


def random_contrast_nd(
    x: jax.Array,
    contrast_range: tuple[float, float],
    key: jr.KeyArray = jr.PRNGKey(0),
) -> jax.Array:
    """Randomly adjusts the contrast of an image by scaling the pixel values by a factor."""
    minval, maxval = contrast_range
    contrast_factor = jr.uniform(key=key, shape=(), minval=minval, maxval=maxval)
    return adjust_contrast_nd(x, contrast_factor)


class AdjustContrastND(pytc.TreeClass):
    """Adjusts the contrast of an NDimage by scaling the pixel values by a factor.

    See:
        https://www.tensorflow.org/api_docs/python/tf/image/adjust_contrast
        https://github.com/deepmind/dm_pix/blob/master/dm_pix/_src/augment.py
    """

    contrast_factor: float = 1.0

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return lax.stop_gradient(adjust_contrast_nd(x, self.contrast_factor))

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        ...


class AdjustContrast2D(AdjustContrastND):
    def __init__(self, contrast_factor=1.0):
        """Adjusts the contrast of an image by scaling the pixel values by a factor.

        Args:
            contrast_factor: contrast factor to adjust the image by.
        """
        super().__init__(contrast_factor=contrast_factor)

    @property
    def spatial_ndim(self) -> int:
        return 2


class RandomContrastND(pytc.TreeClass):
    contrast_range: tuple

    def __init__(self, contrast_range=(0.5, 1)):
        """Randomly adjusts the contrast of an image by scaling the pixel
        values by a factor.

        Args:
            contrast_range: range of contrast factors to randomly sample from.
        """
        if not (
            isinstance(contrast_range, tuple)
            and len(contrast_range) == 2
            and contrast_range[0] <= contrast_range[1]
        ):
            msg = "contrast_range must be a tuple of two floats, "
            msg += "with the first one smaller than the second one."
            raise ValueError(msg)

        self.contrast_range = contrast_range

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(
        self,
        x: jax.Array,
        *,
        key: jr.KeyArray = jr.PRNGKey(0),
        **k,
    ) -> jax.Array:
        return random_contrast_nd(
            x,
            lax.stop_gradient(self.contrast_range),
            key=key,
        )

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        ...


class RandomContrast2D(RandomContrastND):
    def __init__(self, contrast_range=(0.5, 1)):
        """Randomly adjusts the contrast of an image by scaling the pixel
        values by a factor.

        Args:
            contrast_range: range of contrast factors to randomly sample from.
        """
        super().__init__(contrast_range=contrast_range)

    @property
    def spatial_ndim(self) -> int:
        return 2
