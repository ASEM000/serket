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
            bins: number of bins. Defaults to 256.

        Note:
            See:
            https://en.wikipedia.org/wiki/Histogram_equalization
            http://www.janeriksolem.net/histogram-equalization-with-python-and.html
            https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_hist
            https://stackoverflow.com/questions/28518684/histogram-equalization-of-grayscale-images-with-numpy
        """
        self.bins = bins

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        msg = f"Input must have 3 dimensions, got {x.ndim}."
        assert x.ndim == 3, msg
        return _histeq(x, self.bins)


@pytc.treeclass
class PixelShuffle:
    def __init__(self, upscale_factor: int | tuple[int, int] = 1):

        if isinstance(upscale_factor, int):
            if upscale_factor < 1:
                raise ValueError("upscale_factor must be >= 1")

            self.upscale_factor = (upscale_factor, upscale_factor)

        elif isinstance(upscale_factor, tuple):
            if len(upscale_factor) != 2:
                raise ValueError("upscale_factor must be a tuple of length 2")
            if upscale_factor[0] < 1 or upscale_factor[1] < 1:
                raise ValueError("upscale_factor must be >= 1")
            self.upscale_factor = upscale_factor

        else:
            raise ValueError("upscale_factor must be an integer or tuple of length 2")

    def __call__(self, x: jnp.ndarray, **kwargs):
        msg = f"Input must have 3 dimensions, got {x.ndim}."
        assert x.ndim == 3, msg
        channels = x.shape[0]

        sr, sw = self.upscale_factor
        oc = channels // (sr * sw)

        msg = f"Input channels must be divisible by {sr*sw}, got {channels}."
        assert channels % (sr * sw) == 0, msg

        ih, iw = x.shape[1], x.shape[2]
        x = jnp.reshape(x, (sr, sw, oc, ih, iw))
        x = jnp.transpose(x, (2, 3, 0, 4, 1))
        x = jnp.reshape(x, (oc, ih * sr, iw * sw))
        return x
