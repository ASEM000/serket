# Copyright 2023 Serket authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax

import serket as sk
from serket.nn.utils import canonicalize, positive_int_cb, validate_spatial_ndim


def random_cutout_1d(
    x: jax.Array,
    shape: tuple[int] | int,
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
    size = shape[0] if isinstance(shape, tuple) else shape
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


class RandomCutout1D(sk.TreeClass):
    """Random Cutouts for spatial 1D array.

    Args:
        shape: shape of the cutout. accepts an int or a tuple of int.
        cutout_count: number of holes. Defaults to 1.
        fill_value: fill_value to fill. Defaults to 0.

    Note:
        https://arxiv.org/abs/1708.04552
        https://keras.io/api/keras_cv/layers/preprocessing/random_cutout/

    Examples:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> print(sk.nn.RandomCutout1D(5)(jnp.ones((1, 10)) * 100))
        [[100. 100. 100. 100.   0.   0.   0.   0.   0. 100.]]
    """

    def __init__(
        self,
        shape: int | tuple[int],
        cutout_count: int = 1,
        fill_value: int | float = 0,
    ):
        self.shape = canonicalize(shape, ndim=1, name="shape")
        self.cutout_count = positive_int_cb(cutout_count)
        self.fill_value = fill_value

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, *, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        out = random_cutout_1d(x, self.shape, self.cutout_count, self.fill_value, key)
        return lax.stop_gradient(out)

    @property
    def spatial_ndim(self) -> int:
        return 1


class RandomCutout2D(sk.TreeClass):
    """Random Cutouts for spatial 2D array

    Args:
        shape: shape of the cutout. accepts int or a two element tuple.
        cutout_count: number of holes. Defaults to 1.
        fill_value: fill_value to fill. Defaults to 0.

    Note:
        https://arxiv.org/abs/1708.04552
        https://keras.io/api/keras_cv/layers/preprocessing/random_cutout/
    """

    def __init__(
        self,
        shape: int | tuple[int, int],
        cutout_count: int = 1,
        fill_value: int | float = 0,
    ):
        self.shape = canonicalize(shape, 2, name="shape")
        self.cutout_count = positive_int_cb(cutout_count)
        self.fill_value = fill_value

    @ft.partial(validate_spatial_ndim, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, *, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        out = random_cutout_2d(x, self.shape, self.cutout_count, self.fill_value, key)
        return lax.stop_gradient(out)

    @property
    def spatial_ndim(self) -> int:
        return 2
