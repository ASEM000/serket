from __future__ import annotations

import jax.numpy as jnp
import pytreeclass as pytc


@pytc.treeclass
class PaddingND:
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
        if isinstance(self.padding, int):
            self.padding = ((self.padding, self.padding),) * self.ndim

        elif isinstance(self.padding, tuple):
            assert (
                len(self.padding) == self.ndim), f"padding must be a tuple of length {self.ndim}"  # fmt: skip

            for i, x in enumerate(self.padding):
                if isinstance(x, int):
                    self.padding[i] = (x, x)

                elif isinstance(x, tuple):
                    assert len(x) == 2, f"padding must be a tuple of length 2, got {x}"
                    assert all(isinstance(y, int) for y in x), f"padding must be a tuple of ints, got {x}"  # fmt: skip

                else:
                    raise TypeError(f"padding must be an int or tuple, got {x}")

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == self.ndim + 1, f"Input must be {self.ndim + 1}."
        return jnp.pad(x, ((0, 0), *self.padding), constant_values=self.value)


@pytc.treeclass
class Padding1D(PaddingND):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ndim=1)


@pytc.treeclass
class Padding2D(PaddingND):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ndim=2)


@pytc.treeclass
class Padding3D(PaddingND):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ndim=3)
