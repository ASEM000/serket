from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from serket.nn.callbacks import validate_spatial_in_shape


def adjust_contrast_nd(x: jax.Array, contrast_factor: float, spatial_ndim: int):
    """Adjusts the contrast of an image by scaling the pixel values by a factor."""
    μ = jnp.mean(x, axis=tuple(range(1, x.ndim)), keepdims=True)
    return (contrast_factor * (x - μ) + μ).astype(x.dtype)


def random_contrast_nd(
    x: jax.Array,
    contrast_range: tuple[float, float],
    spatial_ndim: int,
    key: jr.PRNGKey = jr.PRNGKey(0),
) -> jax.Array:
    """Randomly adjusts the contrast of an image by scaling the pixel values by a factor."""
    minval, maxval = contrast_range
    contrast_factor = jr.uniform(key=key, shape=(), minval=minval, maxval=maxval)
    return adjust_contrast_nd(x, contrast_factor, spatial_ndim)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class AdjustContrastND:
    """Adjusts the contrast of an NDimage by scaling the pixel values by a factor.

    See:
        https://www.tensorflow.org/api_docs/python/tf/image/adjust_contrast
        https://github.com/deepmind/dm_pix/blob/master/dm_pix/_src/augment.py
    """

    contrast_factor: float = pytc.field(callbacks=[pytc.freeze])

    def __init__(self, contrast_factor=1.0, spatial_ndim=1):
        """
        Args:
            contrast_factor: contrast factor to adjust the image by.
        """

        self.spatial_ndim = spatial_ndim
        self.contrast_factor = contrast_factor

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return adjust_contrast_nd(x, self.contrast_factor, self.spatial_ndim)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class AdjustContrast2D(AdjustContrastND):
    def __init__(self, contrast_factor=1.0):
        """Adjusts the contrast of an image by scaling the pixel values by a factor.
        Args:
            contrast_factor: contrast factor to adjust the image by.
        """
        super().__init__(contrast_factor=contrast_factor, spatial_ndim=2)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class RandomContrastND:
    contrast_range: tuple = pytc.field(callbacks=[pytc.freeze])

    def __init__(self, contrast_range=(0.5, 1), spatial_ndim=1):
        """Randomly adjusts the contrast of an image by scaling the pixel values by a factor.
        Args:
            contrast_range: range of contrast factors to randomly sample from.
        """
        if not (
            isinstance(contrast_range, tuple)
            and len(contrast_range) == 2
            and contrast_range[0] <= contrast_range[1]
        ):
            msg = "contrast_range must be a tuple of two floats, with the first one smaller than the second one."
            raise ValueError(msg)

        self.contrast_range = contrast_range
        self.spatial_ndim = spatial_ndim

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(
        self, x: jax.Array, *, key: jr.PRNGKey = jr.PRNGKey(0), **k
    ) -> jax.Array:
        return random_contrast_nd(x, self.contrast_range, self.spatial_ndim, key=key)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class RandomContrast2D(RandomContrastND):
    def __init__(self, contrast_range=(0.5, 1)):
        """Randomly adjusts the contrast of an image by scaling the pixel values by a factor.
        Args:
            contrast_range: range of contrast factors to randomly sample from.
        """
        super().__init__(contrast_range=contrast_range, spatial_ndim=2)
