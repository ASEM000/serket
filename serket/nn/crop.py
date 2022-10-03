from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc


def _crop_1d(
    x: jnp.ndarray,
    length: int,
    start: int = 0,
    *,
    pad_if_needed: bool = False,
    padding_mode: str = "constant",
) -> jnp.ndarray:
    """Crop a 1D tensor to a given size.

    Args:
        x (jnp.ndarray): image to crop
        length (int): length of the cropped image
        start (int, optional): start coordinate of the crop box. Defaults to 0.
        pad_if_needed (bool, optional): if True, pad the image if the crop box is outside the image. Defaults to False.
        padding_mode (str, optional): padding mode if pad_if_needed is True. Defaults to "constant".
    """

    # assertions for type and shape
    assert x.ndim == 2, f"Expected 2D array, got {x.ndim}D array."
    assert (x.shape[1] >= length), f"Length {length} is larger than image length {x.shape[1]}."  # fmt: skip

    # crop
    assert start >= 0, f"Start {start} must be non-negative."
    assert length > 0, f"Length {length} must be positive."

    if pad_if_needed and (start + length) > x.shape[1]:
        x = jnp.pad(x, ((0, 0), (0, start + length - x.shape[1])), mode=padding_mode)

    return x[:, start : start + length]


def _crop_2d(
    x: jnp.ndarray,
    height: int,
    width: int,
    top: int = 0,
    left: int = 0,
    *,
    pad_if_needed: bool = False,
    padding_mode: str = "constant",
) -> jnp.ndarray:
    """Crop a 3D tensor to a given size.

    Args:
        x (jnp.ndarray): image to crop
        height (int): height of the cropped image
        width (int): width of the cropped image
        top (int, optional): vertical coordinate of the top left corner of the crop box. Defaults to 0.
        left (int, optional): horizontal coordinate of the top left corner of the crop box. Defaults to 0.
        pad_if_needed (bool, optional): if True, pad the image if the crop box is outside the image. Defaults to False.
        padding_mode (str, optional): padding mode if pad_if_needed is True. Defaults to "constant".
    """

    # assertions for type and shape
    assert (x.ndim == 3), f"Expected 2D array of shape [channel,height,width], got {x.ndim}D array."  # fmt: skip
    assert (x.shape[1] >= height), f"Height {height} is larger than image height {x.shape[1]}."  # fmt: skip
    assert (x.shape[2] >= width), f"Width {width} is larger than image width {x.shape[2]}."  # fmt: skip

    # crop
    assert top >= 0, f"Top {top} must be non-negative."
    assert left >= 0, f"Left {left} must be non-negative."
    assert height > 0, f"Height {height} must be positive."
    assert width > 0, f"Width {width} must be positive."

    if pad_if_needed and (top + height) > x.shape[1] and (left + width) > x.shape[2]:
        row_padding = (0, top + height - x.shape[1])
        col_padding = (0, left + width - x.shape[2])
        x = jnp.pad(x, ((0, 0), row_padding, col_padding), mode=padding_mode)
    elif pad_if_needed and (top + height) > x.shape[1]:
        row_padding = (0, top + height - x.shape[1])
        x = jnp.pad(x, ((0, 0), row_padding, (0, 0)), mode=padding_mode)
    elif pad_if_needed and (left + width) > x.shape[2]:
        col_padding = (0, left + width - x.shape[2])
        x = jnp.pad(x, ((0, 0), (0, 0), col_padding), mode=padding_mode)

    return x[:, top : top + height, left : left + width]


@pytc.treeclass
class Crop1D:
    def __init__(
        self,
        length: int,
        start: int = 0,
        *,
        pad_if_needed: bool = False,
        padding_mode: str = "constant",
    ):

        """Crop a 1D tensor to a given size.

        Args:
            length (int): length of the cropped image
            start (int, optional): start coordinate of the crop box. Defaults to 0.
            pad_if_needed (bool, optional): if True, pad the image if the crop box is outside the image.
            padding_mode (str, optional): padding mode if pad_if_needed is True. Defaults to "constant".
        """

        self.length = length
        self.start = start
        self.pad_if_needed = pad_if_needed
        self.padding_mode = padding_mode

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D image."

        return _crop_1d(
            x,
            self.length,
            self.start,
            pad_if_needed=self.pad_if_needed,
            padding_mode=self.padding_mode,
        )


@pytc.treeclass
class Crop2D:
    def __init__(
        self,
        height: int,
        width: int,
        top: int = 0,
        left: int = 0,
        *,
        pad_if_needed: bool = False,
        padding_mode: str = "constant",
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
        self.pad_if_needed = pad_if_needed
        self.padding_mode = padding_mode

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}D image."

        return _crop_2d(
            x,
            self.height,
            self.width,
            self.top,
            self.left,
            pad_if_needed=self.pad_if_needed,
            padding_mode=self.padding_mode,
        )


@pytc.treeclass
class RandomCrop1D:
    def __init__(
        self,
        length: int,
        *,
        pad_if_needed: bool = False,
        padding_mode: str = "constant",
    ):

        """Crop a 1D tensor to a given size.

        Args:
            length (int): length of the cropped image
            pad_if_needed (bool, optional): if True, pad the image if the crop box is outside the image.
            padding_mode (str, optional): padding mode if pad_if_needed is True. Defaults to "constant".
        """

        self.length = length
        self.pad_if_needed = pad_if_needed
        self.padding_mode = padding_mode

    def __call__(self, x: jnp.ndarray, key: jr.PRNGKey = jr.PRNGKey(0)) -> jnp.ndarray:
        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D image."

        start = jr.randint(key, shape=(), minval=0, maxval=x.shape[1] - self.length)

        return _crop_1d(
            x,
            self.length,
            start,
            pad_if_needed=self.pad_if_needed,
            padding_mode=self.padding_mode,
        )


@pytc.treeclass
class RandomCrop2D:
    def __init__(
        self,
        height: int,
        width: int,
        *,
        pad_if_needed: bool = False,
        padding_mode: str = "constant",
    ):

        """
        Args:
            height (int): height of the cropped image
            width (int): width of the cropped image
            pad_if_needed (bool, optional): if True, pad the image if the crop box is outside the image.
            padding_mode (str, optional): padding mode if pad_if_needed is True. Defaults to "constant".
        """

        self.height = height
        self.width = width
        self.pad_if_needed = pad_if_needed
        self.padding_mode = padding_mode

    def __call__(self, x: jnp.ndarray, key: jr.PRNGKey = jr.PRNGKey(0)) -> jnp.ndarray:
        assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}D image."

        top = jr.randint(key, shape=(), minval=0, maxval=x.shape[1] - self.height)
        left = jr.randint(key, shape=(), minval=0, maxval=x.shape[2] - self.width)

        return _crop_2d(
            x,
            self.height,
            self.width,
            top,
            left,
            pad_if_needed=self.pad_if_needed,
            padding_mode=self.padding_mode,
        )
