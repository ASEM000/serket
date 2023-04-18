from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc
from jax.lax import stop_gradient

from serket.nn.callbacks import positive_int_cb, validate_spatial_in_shape
from serket.nn.utils import canonicalize


def random_cutout_1d(
    x: jax.Array,
    shape: int,
    cutout_count: int = 1,
    fill_value: int = 0,
    key: jr.KeyArray = jr.PRNGKey(0),
) -> jax.Array:
    """Random Cutouts for spatial 1D array.

    Args:
        x: input array
        shape: shape of the cutout
        cutout_count: number of holes. Defaults to 1.
        fill_value: fill_value to fill. Defaults to 0.
    """
    size = shape[0]
    row_arange = jnp.arange(x.shape[1])

    # split the key into subkeys, in essence, one for each cutout
    keys = jr.split(key, cutout_count)

    def scan_step(x, key):
        # define the start and end of the cutout region
        minval, maxval = 0, x.shape[1] - size
        # sample the start of the cutout region
        start = jnp.int32(jr.randint(key, shape=(), minval=minval, maxval=maxval))
        # define the mask for the cutout region
        row_mask = (row_arange >= start) & (row_arange < start + size)
        # apply the mask
        x = x * ~row_mask[None, :]
        # return the updated array as carry, skip the scan output
        return x, None

    x, _ = jax.lax.scan(scan_step, x, keys)

    if fill_value != 0:
        # avoid repeating filling the cutout region if the fill value is zero
        return jnp.where(x == 0, fill_value, x)

    return x


def random_cutout_2d(
    x: jax.Array,
    shape: tuple[int, int],
    cutout_count: int = 1,
    fill_value: int = 0,
    key: jr.KeyArray = jr.PRNGKey(0),
) -> jax.Array:
    height, width = shape
    row_arange = jnp.arange(x.shape[1])
    col_arange = jnp.arange(x.shape[2])

    # split the key into `cutout_count` keys, in essence, one for each cutout
    keys = jr.split(key, cutout_count)

    def scan_step(x, key):
        # define a subkey for each dimension
        ktop, kleft = jr.split(key, 2)

        # for top define the start and end of the cutout region
        minval, maxval = 0, x.shape[1] - shape[0]
        # sample the start of the cutout region
        top = jnp.int32(jr.randint(ktop, shape=(), minval=minval, maxval=maxval))

        # for left define the start and end of the cutout region
        minval, maxval = 0, x.shape[2] - shape[1]
        left = jnp.int32(jr.randint(kleft, shape=(), minval=minval, maxval=maxval))

        # define the mask for the cutout region
        row_mask = (row_arange >= top) & (row_arange < top + height)
        col_mask = (col_arange >= left) & (col_arange < left + width)

        x = x * (~jnp.outer(row_mask, col_mask))
        return x, None

    x, _ = jax.lax.scan(scan_step, x, keys)

    if fill_value != 0:
        # avoid repeating filling the cutout region if the fill value is zero
        return jnp.where(x == 0, fill_value, x)

    return x


class RandomCutout1D(pytc.TreeClass):
    shape: int | tuple[int]
    cutout_count: int = pytc.field(callbacks=[positive_int_cb])
    fill_value: float

    def __init__(
        self,
        shape: tuple[int],
        cutout_count: int = 1,
        fill_value: int | float = 0,
    ):
        """Random Cutouts for spatial 1D array.

        Args:
            shape: shape of the cutout
            cutout_count: number of holes. Defaults to 1.
            fill_value: fill_value to fill. Defaults to 0.

        See:
            https://arxiv.org/abs/1708.04552
            https://keras.io/api/keras_cv/layers/preprocessing/random_cutout/

        Examples:
            >>> RandomCutout1D(5)(jnp.ones((1, 10))*100)
            [[100., 100., 100., 100.,   0.,   0.,   0.,   0.,   0., 100.]]
        """
        self.shape = canonicalize(shape, ndim=1, name="shape")
        self.cutout_count = cutout_count
        self.fill_value = fill_value
        self.spatial_ndim = 1

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, *, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        return stop_gradient(
            random_cutout_1d(
                x,
                self.shape,
                self.cutout_count,
                self.fill_value,
                key,
            )
        )


class RandomCutout2D(pytc.TreeClass):
    shape: int | tuple[int, int]
    cutout_count: int = pytc.field(callbacks=[positive_int_cb])
    fill_value: float

    def __init__(
        self,
        shape: int | tuple[int, int],
        cutout_count: int = 1,
        fill_value: int | float = 0,
    ):
        """Random Cutouts for spatial 2D array

        Args:
            shape: shape of the cutout
            cutout_count: number of holes. Defaults to 1.
            fill_value: fill_value to fill. Defaults to 0.

        See:
            https://arxiv.org/abs/1708.04552
            https://keras.io/api/keras_cv/layers/preprocessing/random_cutout/
        """
        self.shape = canonicalize(shape, 2, "shape")
        self.cutout_count = cutout_count
        self.fill_value = fill_value
        self.spatial_ndim = 2

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, *, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        return stop_gradient(
            random_cutout_2d(
                x,
                self.shape,
                self.cutout_count,
                self.fill_value,
                key,
            )
        )
