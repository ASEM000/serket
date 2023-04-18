from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import pytreeclass as pytc

from serket.nn.callbacks import validate_spatial_in_shape
from serket.nn.utils import delayed_canonicalize_padding


class PadND(pytc.TreeClass):
    def __init__(
        self,
        padding: int | tuple[int, int],
        value: float = 0.0,
        spatial_ndim=1,
    ):
        """
        Args:
            padding: padding to apply to each side of the input.
            value: value to pad with. Defaults to 0.0.
            spatial_ndim: number of spatial dimensions. Defaults to 1.

        Note:
            https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding1D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding3D
        """
        self.spatial_ndim = spatial_ndim

        self.padding = delayed_canonicalize_padding(
            in_dim=None,
            padding=padding,
            kernel_size=((1,),) * self.spatial_ndim,
            strides=None,
        )
        self.value = value

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        # do not pad the channel axis
        return jnp.pad(x, ((0, 0), *self.padding), constant_values=self.value)


class Pad1D(PadND):
    def __init__(self, padding: int | tuple[int, int], value: float = 0.0):
        """
        Pad a 1D tensor.

        Args:
            padding: padding to apply to each side of the input.
            value: value to pad with. Defaults to 0.0.

        Note:
            https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding1D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding3D
        """
        super().__init__(
            padding=padding,
            value=value,
            spatial_ndim=1,
        )


class Pad2D(PadND):
    def __init__(
        self,
        padding: int | tuple[int, int],
        value: float = 0.0,
    ):
        """
        Pad a 2D tensor.

        Args:
            padding: padding to apply to each side of the input.
            value: value to pad with. Defaults to 0.0.

        Note:
            https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding1D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding3D
        """
        super().__init__(
            padding=padding,
            value=value,
            spatial_ndim=2,
        )


class Pad3D(PadND):
    def __init__(
        self,
        padding: int | tuple[int, int],
        value: float = 0.0,
    ):
        """
        Pad a 3D tensor.

        Args:
            padding: padding to apply to each side of the input.
            value: value to pad with. Defaults to 0.0.

        Note:
            https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding1D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding3D
        """
        super().__init__(
            padding=padding,
            value=value,
            spatial_ndim=3,
        )
