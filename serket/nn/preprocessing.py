from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import pytreeclass as pytc


@ft.partial(jax.jit, static_argnums=(1,))
def _histeq(x, bins_count: int = 256):
    hist, bins = jnp.histogram(x.flatten(), bins_count, density=True)
    cdf = hist.cumsum()
    cdf = (bins_count - 1) * cdf / cdf[-1]
    return jnp.interp(x.flatten(), bins[:-1], cdf).reshape(x.shape)


@pytc.treeclass
class HistogramEqualization2D:
    bins: int = pytc.nondiff_field()

    def __init__(self, bins: int = 256):
        """Apply histogram equalization to 2D spatial array channel wise
        Args:
            bins (int, optional): number of bins. Defaults to 256.

        Note:
            See:
            https://en.wikipedia.org/wiki/Histogram_equalization
            http://www.janeriksolem.net/histogram-equalization-with-python-and.html
            https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_hist
            https://stackoverflow.com/questions/28518684/histogram-equalization-of-grayscale-images-with-numpy
        """
        self.bins = bins

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 3, "x must be 3D"
        return _histeq(x, self.bins)
