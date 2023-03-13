from __future__ import annotations

import jax
import jax.numpy as jnp
import pytreeclass as pytc

from serket.nn.utils import _canonicalize_padding, _check_spatial_in_shape


@pytc.treeclass
class PadND:
    def __init__(
        self, padding: int | tuple[int, int], value: float = 0.0, spatial_ndim=1
    ):
        """
        Args:
            padding: padding to apply to each side of the input.
            value: value to pad with. Defaults to 0.0.
            spatial_ndim: number of spatial dimensions. Defaults to 1.

        see:
            https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding1D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding3D
        """
        self.spatial_ndim = spatial_ndim
        self.padding = _canonicalize_padding(padding, ((1,),) * self.spatial_ndim)
        self.value = value

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        _check_spatial_in_shape(x, self.spatial_ndim)
        # do not pad the channel axis
        return jnp.pad(x, ((0, 0), *self.padding), constant_values=self.value)


@pytc.treeclass
class Pad1D(PadND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=1)


@pytc.treeclass
class Pad2D(PadND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=2)


@pytc.treeclass
class Pad3D(PadND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, spatial_ndim=3)
