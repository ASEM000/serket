from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc


@pytc.treeclass
class AdjustContrastND:
    contrast_factor: float = pytc.nondiff_field()

    def __init__(self, contrast_factor=1.0, ndim=1):
        """Adjusts the contrast of an image by scaling the pixel values by a factor.

        Args:
            contrast_factor: factor to scale the pixel values by.

        See:
            https://www.tensorflow.org/api_docs/python/tf/image/adjust_contrast
            https://github.com/deepmind/dm_pix/blob/master/dm_pix/_src/augment.py
        """
        self.ndim = ndim
        self.contrast_factor = contrast_factor

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        msg = f"Input must have {self.ndim + 1} dimensions, got {x.ndim}."
        assert x.ndim == self.ndim + 1, msg
        μ = jnp.mean(x, axis=tuple(range(1, x.ndim)), keepdims=True)
        return (self.contrast_factor * (x - μ) + μ).astype(x.dtype)


@pytc.treeclass
class AdjustContrast2D(AdjustContrastND):
    def __init__(self, contrast_factor=1.0):
        super().__init__(contrast_factor, ndim=2)


@pytc.treeclass
class RandomContrastND:
    contrast_range: tuple = pytc.nondiff_field()

    def __init__(self, contrast_range=(0.5, 1), ndim=1):
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
        self.ndim = ndim

    def __call__(
        self, x: jnp.ndarray, key: jr.PRNGKey = jr.PRNGKey(0), **kwargs
    ) -> jnp.ndarray:
        msg = f"Input must have {self.ndim + 1} dimensions, got {x.ndim}."
        assert x.ndim == self.ndim + 1, msg
        contrast_factor = jr.uniform(
            key=key,
            shape=(),
            minval=self.contrast_range[0],
            maxval=self.contrast_range[1],
        )
        return AdjustContrastND(contrast_factor=contrast_factor, ndim=self.ndim)(x)


@pytc.treeclass
class RandomContrast2D(RandomContrastND):
    def __init__(self, contrast_range=(0.5, 1)):
        super().__init__(contrast_range, ndim=2)
