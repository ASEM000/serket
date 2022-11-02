from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from serket.nn.utils import _check_spatial_in_shape


@pytc.treeclass
class AdjustContrastND:
    contrast_factor: float = pytc.field(nondiff=True)

    def __init__(self, contrast_factor=1.0, spatial_ndim=1):
        """Adjusts the contrast of an image by scaling the pixel values by a factor.

        Args:
            contrast_factor: factor to scale the pixel values by.

        See:
            https://www.tensorflow.org/api_docs/python/tf/image/adjust_contrast
            https://github.com/deepmind/dm_pix/blob/master/dm_pix/_src/augment.py
        """
        self.spatial_ndim = spatial_ndim
        self.contrast_factor = contrast_factor

    def __call__(self, x: jnp.ndarray, **k) -> jnp.ndarray:
        _check_spatial_in_shape(x, self.spatial_ndim)
        μ = jnp.mean(x, axis=tuple(range(1, x.ndim)), keepdims=True)
        return (self.contrast_factor * (x - μ) + μ).astype(x.dtype)


@pytc.treeclass
class AdjustContrast2D(AdjustContrastND):
    def __init__(self, contrast_factor=1.0):
        super().__init__(contrast_factor, spatial_ndim=2)


@pytc.treeclass
class RandomContrastND:
    contrast_range: tuple = pytc.field(nondiff=True)

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

    def __call__(
        self, x: jnp.ndarray, key: jr.PRNGKey = jr.PRNGKey(0), **k
    ) -> jnp.ndarray:
        _check_spatial_in_shape(x, self.spatial_ndim)
        contrast_factor = jr.uniform(
            key=key,
            shape=(),
            minval=self.contrast_range[0],
            maxval=self.contrast_range[1],
        )
        return AdjustContrastND(
            contrast_factor=contrast_factor, spatial_ndim=self.spatial_ndim
        )(x)


@pytc.treeclass
class RandomContrast2D(RandomContrastND):
    def __init__(self, contrast_range=(0.5, 1)):
        super().__init__(contrast_range, spatial_ndim=2)
