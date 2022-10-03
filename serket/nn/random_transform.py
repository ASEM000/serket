from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc
from pytreeclass._src.tree_util import is_treeclass

from .crop import Crop1D, Crop2D


@pytc.treeclass
class RandomCrop1D:
    length: int = pytc.nondiff_field()

    def __init__(
        self,
        length: int,
    ):

        """Crop a 1D tensor to a given size.

        Args:
            length (int): length of the cropped image
            pad_if_needed (bool, optional): if True, pad the image if the crop box is outside the image.
            padding_mode (str, optional): padding mode if pad_if_needed is True. Defaults to "constant".
        """

        self.length = length

    def __call__(self, x: jnp.ndarray, key: jr.PRNGKey = jr.PRNGKey(0)) -> jnp.ndarray:
        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D image."
        start = jr.randint(key, shape=(), minval=0, maxval=x.shape[1] - self.length)
        return Crop1D(length=self.length, start=start)(x)


@pytc.treeclass
class RandomCrop2D:
    height: int = pytc.nondiff_field()
    width: int = pytc.nondiff_field()
    pad_if_needed: bool = pytc.nondiff_field()

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

    def __call__(self, x: jnp.ndarray, key: jr.PRNGKey = jr.PRNGKey(0)) -> jnp.ndarray:
        assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}D image."

        top = jr.randint(key, shape=(), minval=0, maxval=x.shape[1] - self.height)
        left = jr.randint(key, shape=(), minval=0, maxval=x.shape[2] - self.width)

        return Crop2D(height=self.height, width=self.width, top=top, left=left)(x)


@pytc.treeclass
class RandomApply:
    layer: int
    p: float = pytc.nondiff_field(default=1.0)
    eval: bool | None

    def __init__(self, layer, p: float = 0.5, ndim: int = 1, eval: bool | None = None):
        """
        Randomly applies a layer with probability p.

        Args:
            p: probability of applying the layer

        Example:
            >>> layer = RandomApply(sk.nn.MaxPool2D(kernel_size=2, strides=2), p=0.0)
            >>> layer(jnp.ones((1, 10, 10))).shape
            (1, 10, 10)

            >>> layer = RandomApply(sk.nn.MaxPool2D(kernel_size=2, strides=2), p=1.0)
            >>> layer(jnp.ones((1, 10, 10))).shape
            (1, 5, 5)

        Note:
            See: https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomApply
            Use sk.nn.Sequential to apply multiple layers.
        """

        if p < 0 or p > 1:
            raise ValueError(f"p must be between 0 and 1, got {p}")

        if isinstance(eval, bool) or eval is None:
            self.eval = eval
        else:
            raise ValueError(f"eval must be a boolean or None, got {eval}")

        self.p = p

        if not is_treeclass(layer):
            raise ValueError("Layer must be a `treeclass`.")
        self.layer = layer

    def __call__(self, x: jnp.ndarray, key: jr.PRNGKey = jr.PRNGKey(0)):

        if self.eval is True or not jr.bernoulli(key, (self.p)):
            return x

        return self.layer(x)


@pytc.treeclass
class RandomCutout1D:
    shape: tuple[int] = pytc.nondiff_field()
    cutout_count: int = pytc.nondiff_field()
    fill_value: float = pytc.nondiff_field()

    def __init__(
        self,
        shape: tuple[int, ...],
        cutout_count: int = 1,
        fill_value: int | float = 0,
    ):
        """Random Cutouts for spatial 1D array.

        Args:
            shape (tuple[int, ...]): shape of the cutout
            cutout_count (int, optional): number of holes. Defaults to 1.
            fill_value (float, optional): fill_value to fill. Defaults to 0.

        See:
            https://arxiv.org/abs/1708.04552
            https://keras.io/api/keras_cv/layers/preprocessing/random_cutout/
        """
        assert len(shape) == 1, "shape must be 1D"
        self.shape = shape
        self.cutout_count = cutout_count
        self.fill_value = fill_value

    def __call__(
        self, x: jnp.ndarray, *, key: jr.PRNGKey = jr.PRNGKey(0)
    ) -> jnp.ndarray:
        length = self.shape[0]
        row_arange = jnp.arange(x.shape[1])

        keys = jr.split(key, self.cutout_count)

        def scan_step(x, key):
            start = jr.randint(key, shape=(), minval=0, maxval=x.shape[1]).astype(jnp.int32)  # fmt: skip
            row_mask = (row_arange >= start) & (row_arange < start + length)
            x = x * ~row_mask[None, :]
            return x, None

        res = jax.lax.scan(scan_step, x, keys)[0]

        if self.fill_value != 0:
            return jnp.where(res == 0, self.fill_value, res)

        return res


@pytc.treeclass
class RandomCutout2D:
    shape: tuple[int, int] = pytc.nondiff_field()
    cutout_count: int = pytc.nondiff_field()
    fill_value: float = pytc.nondiff_field()

    def __init__(
        self,
        shape: tuple[int, ...],
        cutout_count: int = 1,
        fill_value: int | float = 0,
    ):
        """Random Cutouts for spatial 2D array

        Args:
            shape (tuple[int, ...]): shape of the cutout
            cutout_count (int, optional): number of holes. Defaults to 1.
            fill_value (float, optional): fill_value to fill. Defaults to 0.

        See:
            https://arxiv.org/abs/1708.04552
            https://keras.io/api/keras_cv/layers/preprocessing/random_cutout/
        """
        assert len(shape) == 2, "shape must be 2D"
        self.shape = shape
        self.cutout_count = cutout_count
        self.fill_value = fill_value

    def __call__(
        self, x: jnp.ndarray, *, key: jr.PRNGKey = jr.PRNGKey(0)
    ) -> jnp.ndarray:
        height, width = self.shape
        row_arange = jnp.arange(x.shape[1])
        col_arange = jnp.arange(x.shape[2])

        keys = jr.split(key, self.cutout_count)

        def scan_step(x, key):
            ktop, kleft = jr.split(key, 2)
            top = jr.randint(ktop, shape=(), minval=0, maxval=x.shape[1]).astype(jnp.int32)  # fmt: skip
            left = jr.randint(kleft, shape=(), minval=0, maxval=x.shape[2]).astype(jnp.int32)  # fmt: skip
            row_mask = (row_arange >= top) & (row_arange < top + height)
            col_mask = (col_arange >= left) & (col_arange < left + width)
            x = x * (~jnp.outer(row_mask, col_mask))
            return x, None

        res = jax.lax.scan(scan_step, x, keys)[0]

        if self.fill_value != 0:
            return jnp.where(res == 0, self.fill_value, res)

        return res
