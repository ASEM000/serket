from __future__ import annotations

import jax
import jax.numpy as jnp
import pytreeclass as pytc


@pytc.treeclass
class Crop1D:
    size: int = pytc.nondiff_field()
    start: int = pytc.nondiff_field()

    def __init__(
        self,
        size: int,
        start: int = 0,
    ):

        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            size (int): size of the cropped image
            start (int, optional): start coordinate of the crop box. Defaults to 0.
        """
        assert isinstance(size, int), f"Expected size to be an int, got {type(size)}."
        assert isinstance(
            start, int
        ), f"Expected start to be an int, got {type(start)}."
        self.size = size
        self.start = start

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 2, f"Expected 2D array, got {x.ndim}D image."
        return jax.lax.dynamic_slice_in_dim(x, self.start, self.size, axis=1)


@pytc.treeclass
class Crop2D:
    size: tuple[int, int] = pytc.nondiff_field()
    start: tuple[int, int] = pytc.nondiff_field()

    def __init__(
        self,
        size: tuple[int, int],
        start: tuple[int, int] = (0, 0),
    ):

        """
        Args:
            size (tuple[int, int]): size of the cropped image
            start (tuple[int, int], optional): top left coordinate of the crop box. Defaults to (0, 0).
            pad_if_needed (bool, optional): if True, pad the image if the crop box is outside the image.
            padding_mode (str, optional): padding mode if pad_if_needed is True. Defaults to "constant".
        """
        assert (
            len(size) == 2
        ), f"Expected size to be a tuple of size 2, got {len(size)}."
        assert (
            len(start) == 2
        ), f"Expected start to be a tuple of size 2, got {len(start)}."

        self.size = size
        self.start = start

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert (x.ndim == 3), f"Expected 2D array of shape [channel,height,width], got {x.ndim}D array."  # fmt: skip
        start_indices = (0, *self.start)
        slice_sizes = (x.shape[0], *self.size)
        return jax.lax.dynamic_slice(x, start_indices, slice_sizes)
