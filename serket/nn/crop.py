from __future__ import annotations

import jax
import jax.numpy as jnp
import pytreeclass as pytc


@pytc.treeclass
class Crop1D:
    size: int = pytc.nondiff_field()
    start: int = pytc.nondiff_field(default=0)

    def __post_init__(self):

        """Applies jax.lax.dynamic_slice_in_dim to the second dimension of the input.

        Args:
            size (int): size of the cropped image
            start (int, optional): start coordinate of the crop box. Defaults to 0.
        """
        assert isinstance(
            self.size, int
        ), f"Expected size to be an int, got {type(self.size)}."
        assert isinstance(
            self.start, int
        ), f"Expected start to be an int, got {type(self.start)}."

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 2, f"Expected 2D array, got {x.ndim}D image."
        return jax.lax.dynamic_slice_in_dim(x, self.start, self.size, axis=1)


@pytc.treeclass
class Crop2D:
    size: tuple[int, int] = pytc.nondiff_field()
    start: tuple[int, int] = pytc.nondiff_field(default=(0, 0))

    def __post_init__(self):

        """
        Args:
            size (tuple[int, int]): size of the cropped image
            start (tuple[int, int], optional): top left coordinate of the crop box. Defaults to (0, 0).
            pad_if_needed (bool, optional): if True, pad the image if the crop box is outside the image.
            padding_mode (str, optional): padding mode if pad_if_needed is True. Defaults to "constant".
        """
        assert (
            len(self.size) == 2
        ), f"Expected size to be a tuple of size 2, got {len(self.size)}."
        assert (
            len(self.start) == 2
        ), f"Expected start to be a tuple of size 2, got {len(self.start)}."

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert (x.ndim == 3), f"Expected 2D array of shape [channel,height,width], got {x.ndim}D array."  # fmt: skip
        start_indices = (0, *self.start)
        slice_sizes = (x.shape[0], *self.size)
        return jax.lax.dynamic_slice(x, start_indices, slice_sizes)
