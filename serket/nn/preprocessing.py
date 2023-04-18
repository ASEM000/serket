from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import pytreeclass as pytc

from serket.nn.callbacks import positive_int_cb, validate_spatial_in_shape


@ft.partial(jax.jit, static_argnums=(1,))
def histeq(x, bins_count: int = 256):
    hist, bins = jnp.histogram(x.flatten(), bins_count, density=True)
    cdf = hist.cumsum()
    cdf = (bins_count - 1) * cdf / cdf[-1]
    return jnp.interp(x.flatten(), bins[:-1], cdf).reshape(x.shape)


class HistogramEqualization2D(pytc.TreeClass):
    bins: int = pytc.field(callbacks=[positive_int_cb])

    def __init__(self, bins: int = 256):
        """Apply histogram equalization to 2D spatial array channel wise
        Args:
            bins: number of bins. Defaults to 256.

        Note:
            https://en.wikipedia.org/wiki/Histogram_equalization
            http://www.janeriksolem.net/histogram-equalization-with-python-and.html
            https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_hist
            https://stackoverflow.com/questions/28518684/histogram-equalization-of-grayscale-images-with-numpy
        """
        self.spatial_ndim = 2
        self.bins = bins

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return histeq(x, self.bins)


class PixelShuffle2D(pytc.TreeClass):
    def __init__(self, upscale_factor: int | tuple[int, int] = 1):
        self.spatial_ndim = 2
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

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k):
        channels = x.shape[0]

        sr, sw = self.upscale_factor
        oc = channels // (sr * sw)

        if not (channels % (sr * sw)) == 0:
            msg = f"Input channels must be divisible by {sr*sw}, got {channels}."
            raise ValueError(msg)

        ih, iw = x.shape[1], x.shape[2]
        x = jnp.reshape(x, (sr, sw, oc, ih, iw))
        x = jnp.transpose(x, (2, 3, 0, 4, 1))
        x = jnp.reshape(x, (oc, ih * sr, iw * sw))
        return x
