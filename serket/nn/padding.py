from __future__ import annotations

import jax.numpy as jnp
import pytreeclass as pytc

from serket.nn.utils import _check_and_return_padding


@pytc.treeclass
class PadND:
    padding: int | tuple[int, int] = pytc.nondiff_field(default=0)
    value: float = pytc.nondiff_field(default=0.0)
    ndim: int = pytc.nondiff_field(default=1, repr=False)

    def __post_init__(self):
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
        self.padding = _check_and_return_padding(self.padding, ((1,),) * self.ndim)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        msg = f"Input must have {self.ndim + 1} dimensions, got {x.ndim}."
        assert x.ndim == self.ndim + 1, msg
        # do not pad the channel axis
        return jnp.pad(x, ((0, 0), *self.padding), constant_values=self.value)


@pytc.treeclass
class Pad1D(PadND):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ndim=1)


@pytc.treeclass
class Pad2D(PadND):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ndim=2)


@pytc.treeclass
class Pad3D(PadND):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ndim=3)
