from __future__ import annotations

import jax.numpy as jnp
import pytreeclass as pytc

from serket.nn.utils import _check_and_return_padding, _check_spatial_in_shape


@pytc.treeclass
class PadND:
    def __init__(self, padding: int | tuple[int, int], value: float = 0.0, ndim=1):
        """
        Args:
            padding: padding to apply to each side of the input.
            value: value to pad with. Defaults to 0.0.
            ndim: number of spatial dimensions. Defaults to 1.

        see:
            https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding1D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding3D
        """
        self.ndim = ndim
        self.padding = _check_and_return_padding(padding, ((1,),) * self.ndim)
        self.value = value

    @_check_spatial_in_shape
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        # do not pad the channel axis
        return jnp.pad(x, ((0, 0), *self.padding), constant_values=self.value)


@pytc.treeclass
class Pad1D(PadND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=1)


@pytc.treeclass
class Pad2D(PadND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=2)


@pytc.treeclass
class Pad3D(PadND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=3)
