from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from .crop import Crop1D, Crop2D
from .padding import Padding2D
from .resize import Resize2D


@pytc.treeclass
class RandomCrop1D:
    size: int = pytc.nondiff_field()

    def __init__(
        self,
        size: tuple[int],
    ):

        """Crop a 1D tensor to a given size.

        Args:
            size (int): size of the cropped image
            pad_if_needed (bool, optional): if True, pad the image if the crop box is outside the image.
            padding_mode (str, optional): padding mode if pad_if_needed is True. Defaults to "constant".
        """
        assert len(size) == 1, f"Expected size to be a 1-tuple, got {size}."
        self.size = size

    def __call__(self, x: jnp.ndarray, key: jr.PRNGKey = jr.PRNGKey(0)) -> jnp.ndarray:
        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D image."
        start = jr.randint(key, shape=(), minval=0, maxval=x.shape[1] - self.size)
        return Crop1D(self.size, start)(x)


@pytc.treeclass
class RandomCrop2D:
    size: tuple[int, int] = pytc.nondiff_field()
    pad_if_needed: bool = pytc.nondiff_field()

    def __init__(
        self,
        size: tuple[int, int],
        *,
        pad_if_needed: bool = False,
        padding_mode: str = "constant",
    ):

        """
        Args:
            size (tuple[int, int]): size of the cropped image
            pad_if_needed (bool, optional): if True, pad the image if the crop box is outside the image.
            padding_mode (str, optional): padding mode if pad_if_needed is True. Defaults to "constant".
        """
        assert (
            len(size) == 2
        ), f"Expected size to be a tuple of size 2, got {len(size)}."
        self.size = size

    def __call__(self, x: jnp.ndarray, key: jr.PRNGKey = jr.PRNGKey(0)) -> jnp.ndarray:
        assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}D image."

        top = jr.randint(key, shape=(), minval=0, maxval=x.shape[1] - self.size[0])
        left = jr.randint(key, shape=(), minval=0, maxval=x.shape[2] - self.size[1])

        return Crop2D(self.size, (top, left))(x)


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

        if not pytc.is_treeclass(layer):
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
        size = self.shape[0]
        row_arange = jnp.arange(x.shape[1])

        keys = jr.split(key, self.cutout_count)

        def scan_step(x, key):
            start = jr.randint(key, shape=(), minval=0, maxval=x.shape[1]).astype(jnp.int32)  # fmt: skip
            row_mask = (row_arange >= start) & (row_arange < start + size)
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


@pytc.treeclass
class RandomZoom2D:
    def __init__(
        self,
        height_factor: tuple[float, float] = (0, 1),
        width_factor: tuple[float, float] = (0, 1),
    ):
        """
        Args:
            height_factor: (min, max)
            width_factor: (min, max)

        Note:
            See: https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomZoom
            Positive values are zoom in, negative values are zoom out.
        """
        assert (
            isinstance(height_factor, tuple) and len(height_factor) == 2
        ), "height_factor must be a tuple of length 2"

        assert (
            isinstance(width_factor, tuple) and len(width_factor) == 2
        ), "width_factor must be a tuple of length 2"

        self.height_factor = height_factor
        self.width_factor = width_factor

    def __call__(self, x: jnp.ndarray, key: jr.PRNGKey = jr.PRNGKey(0)) -> jnp.ndarray:

        keys = jr.split(key, 3)

        height_factor = jr.uniform(
            keys[0],
            shape=(),
            minval=self.height_factor[0],
            maxval=self.height_factor[1],
        )
        width_factor = jr.uniform(
            keys[1], shape=(), minval=self.width_factor[0], maxval=self.width_factor[1]
        )

        resized_rows = int(x.shape[1] * (1 + height_factor))
        resized_cols = int(x.shape[2] * (1 + width_factor))

        if height_factor >= 0 and width_factor >= 0:
            # zoom in rows and cols
            crop_transform = RandomCrop2D(x.shape[1:])
            resize_transform = Resize2D((resized_rows, resized_cols))
            x = resize_transform(x)
            x = crop_transform(x, key=keys[2])
            return x

        elif height_factor <= 0 and width_factor <= 0:
            # zoom out rows and cols
            resize_transform = Resize2D((resized_rows, resized_cols))
            padding = (x.shape[1] - resized_rows, 0), (x.shape[2] - resized_cols, 0)
            padding_transform = Padding2D(padding)
            x = resize_transform(x)
            x = padding_transform(x)
            return x

        elif height_factor <= 0 and width_factor >= 0:
            # zoom out rows and zoom in cols
            resize_transform = Resize2D((x.shape[1], resized_cols))
            padding = ((x.shape[1] - resized_rows, 0), (0, 0))
            padding_transform = Padding2D(padding)
            crop_transform = RandomCrop2D(x.shape[1:])
            x = resize_transform(x)
            x = crop_transform(x, key=keys[2])
            x = padding_transform(x)
            return x

        elif height_factor >= 0 and width_factor <= 0:
            # zoom in rows and zoom out cols
            resize_transform = Resize2D((resized_rows, x.shape[2]))
            padding = ((0, 0), (x.shape[2] - resized_cols, 0))
            padding_transform = Padding2D(padding)
            crop_transform = RandomCrop2D(x.shape[1:])
            x = resize_transform(x)
            x = crop_transform(x, key=keys[2])
            x = padding_transform(x)
            return x
