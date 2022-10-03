from __future__ import annotations

import jax
import jax.numpy as jnp
import pytreeclass as pytc


@pytc.treeclass
class Crop1D:
    length: int = pytc.nondiff_field()
    start: int = pytc.nondiff_field()

    def __init__(
        self,
        length: int,
        start: int = 0,
    ):

        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            length (int): length of the cropped image
            start (int, optional): start coordinate of the crop box. Defaults to 0.
        """
        self.length = length
        self.start = start

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 2, f"Expected 2D array, got {x.ndim}D image."
        return jax.lax.dynamic_slice_in_dim(x, self.start, self.length, axis=1)


@pytc.treeclass
class Crop2D:
    height: int = pytc.nondiff_field()
    width: int = pytc.nondiff_field()
    top: int = pytc.nondiff_field()
    left: int = pytc.nondiff_field()

    def __init__(
        self,
        height: int,
        width: int,
        top: int = 0,
        left: int = 0,
    ):

        """
        Args:
            height (int): height of the cropped image
            width (int): width of the cropped image
            top (int, optional): vertical coordinate of the top left corner of the crop box. Defaults to 0.
            left (int, optional): horizontal coordinate of the top left corner of the crop box. Defaults to 0.
            pad_if_needed (bool, optional): if True, pad the image if the crop box is outside the image.
            padding_mode (str, optional): padding mode if pad_if_needed is True. Defaults to "constant".
        """
        self.height = height
        self.width = width
        self.top = top
        self.left = left

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert (x.ndim == 3), f"Expected 2D array of shape [channel,height,width], got {x.ndim}D array."  # fmt: skip
        start_indices = (0, self.top, self.left)
        slice_sizes = (x.shape[0], self.height, self.width)
        return jax.lax.dynamic_slice(x, start_indices, slice_sizes)
