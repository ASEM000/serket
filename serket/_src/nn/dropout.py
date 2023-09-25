# Copyright 2023 serket authors
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

import abc
import functools as ft
from itertools import chain
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import kernex as kex

import serket as sk
from serket._src.custom_transform import tree_eval
from serket._src.utils import (
    IsInstance,
    Range,
    canonicalize,
    positive_int_cb,
    validate_spatial_nd,
)


def dropout_nd(
    key: jr.KeyArray,
    x: jax.Array,
    drop_rate,
    drop_axes: Sequence[int] | None = None,
) -> jax.Array:
    """Drop some elements of the input array."""
    # drop_axes = None means dropout is applied to all axes
    shape = (
        x.shape
        if drop_axes is None
        else (x.shape[i] if i in drop_axes else 1 for i in range(x.ndim))
    )

    return jnp.where(
        (keep_prop := (1 - drop_rate)) == 0.0,
        jnp.zeros_like(x),
        jnp.where(jr.bernoulli(key, keep_prop, shape=shape), x / keep_prop, 0),
    )


def random_cutout_nd(
    key: jax.Array,
    array: jax.Array,
    shape: tuple[int, ...],
    cutout_count: int,
    fill_value: int | float,
):
    """Random Cutouts for spatial ND array.

    Args:
        key: random number generator key
        array: input array
        shape: shape of the cutout
        cutout_count: number of holes. Defaults to 1.
        fill_value: fill_value to fill. Defaults to 0.
    """
    start_indices = [0] * len(shape)
    slice_sizes = [di - (di % ki) for di, ki in zip(array.shape, shape)]
    valid_array = jax.lax.dynamic_slice(array, start_indices, slice_sizes)

    # get non-overlapping patches
    patches = kex.kmap(kernel_size=shape, strides=shape)(lambda x: x)(array)
    patches_shape = patches.shape

    # patches_shape = (patch_0, ..., patch_n, k0, ..., kn)
    patches = patches.reshape(-1, *shape)
    indices = jr.choice(key, patches.shape[0], shape=(cutout_count,), replace=False)
    patches = patches.at[indices].set(fill_value).reshape(patches_shape)
    # patches_shape = (patch_0, k0, ..., patch_n, kn)
    patch_axes = range(len(shape))
    kernel_axes = range(len(shape), len(shape) * 2)
    transpose_axes = list(chain.from_iterable(zip(patch_axes, kernel_axes)))
    depatched = patches.transpose(transpose_axes).reshape(valid_array.shape)
    return jax.lax.dynamic_update_slice(array, depatched, start_indices)


@sk.autoinit
class Dropout(sk.TreeClass):
    """Drop some elements of the input array.

    Randomly zeroes some of the elements of the input array with
    probability ``drop_rate`` using samples from a Bernoulli distribution.

    Args:
        drop_rate: probability of an element to be zeroed. Default: 0.5
        drop_axes: axes to apply dropout. Default: None to apply to all axes.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> layer = sk.nn.Dropout(0.5)
        >>> print(layer(jnp.ones([10]), key=jr.PRNGKey(0)))
        [2. 0. 2. 2. 2. 2. 2. 2. 0. 0.]

    Note:
        Use :func:`.tree_eval` to turn off dropout during evaluation.

        >>> import serket as sk
        >>> import jax.random as jr
        >>> linear = sk.nn.Linear(10, 10, key=jr.PRNGKey(0))
        >>> dropout = sk.nn.Dropout(0.5)
        >>> layers = sk.nn.Sequential(dropout, linear)
        >>> sk.tree_eval(layers)
        Sequential(
          layers=(
            Identity(),
            Linear(
              in_features=(10),
              out_features=10,
              weight_init=glorot_uniform,
              bias_init=zeros,
              weight=f32[10,10](μ=0.01, σ=0.31, ∈[-0.54,0.54]),
              bias=f32[10](μ=0.00, σ=0.00, ∈[0.00,0.00])
            )
          )
        )
    """

    drop_rate: float = sk.field(
        default=0.5,
        on_setattr=[IsInstance(float), Range(0, 1)],
        on_getattr=[jax.lax.stop_gradient_p.bind],
    )
    drop_axes: tuple[int, ...] | None = None

    def __call__(self, x, *, key: jr.KeyArray):
        """Drop some elements of the input array.

        Args:
            x: input array
            key: random number generator key
        """
        return dropout_nd(key, x, self.drop_rate, self.drop_axes)


@sk.autoinit
class DropoutND(sk.TreeClass):
    drop_rate: float = sk.field(
        default=0.5,
        on_setattr=[IsInstance(float), Range(0, 1)],
        on_getattr=[jax.lax.stop_gradient_p.bind],
    )

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x, *, key):
        """Drop some elements of the input array.

        Args:
            x: input array
            key: random number generator key
        """
        return dropout_nd(key, x, self.drop_rate, [0])

    @property
    @abc.abstractmethod
    def spatial_ndim(self):
        ...


class Dropout1D(DropoutND):
    """Drops full feature maps along the channel axis.

    Args:
        drop_rate: fraction of an elements to be zeroed out.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> layer = sk.nn.Dropout1D(0.5)
        >>> print(layer(jnp.ones((1, 10)), key=jr.PRNGKey(0)))
        [[2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]]

    Note:
        Use :func:`.tree_eval` to turn off dropout during evaluation.

        >>> import serket as sk
        >>> import jax.random as jr
        >>> linear = sk.nn.Linear(10, 10, key=jr.PRNGKey(0))
        >>> dropout = sk.nn.Dropout1D(0.5)
        >>> layers = sk.nn.Sequential(dropout, linear)
        >>> sk.tree_eval(layers)
        Sequential(
          layers=(
            Identity(),
            Linear(
              in_features=(10),
              out_features=10,
              weight_init=glorot_uniform,
              bias_init=zeros,
              weight=f32[10,10](μ=0.01, σ=0.31, ∈[-0.54,0.54]),
              bias=f32[10](μ=0.00, σ=0.00, ∈[0.00,0.00])
            )
          )
        )

    Reference:
        - https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
        - https://arxiv.org/abs/1411.4280
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class Dropout2D(DropoutND):
    """Drops full feature maps along the channel axis.

    Args:
        drop_rate: fraction of an elements to be zeroed out.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> layer = sk.nn.Dropout2D(0.5)
        >>> print(layer(jnp.ones((1, 5, 5)), key=jr.PRNGKey(0)))
        [[[2. 2. 2. 2. 2.]
          [2. 2. 2. 2. 2.]
          [2. 2. 2. 2. 2.]
          [2. 2. 2. 2. 2.]
          [2. 2. 2. 2. 2.]]]

    Note:
        Use :func:`.tree_eval` to turn off dropout during evaluation.

        >>> import serket as sk
        >>> import jax.random as jr
        >>> linear = sk.nn.Linear(10, 10, key=jr.PRNGKey(0))
        >>> dropout = sk.nn.Dropout2D(0.5)
        >>> layers = sk.nn.Sequential(dropout, linear)
        >>> sk.tree_eval(layers)
        Sequential(
          layers=(
            Identity(),
            Linear(
              in_features=(10),
              out_features=10,
              weight_init=glorot_uniform,
              bias_init=zeros,
              weight=f32[10,10](μ=0.01, σ=0.31, ∈[-0.54,0.54]),
              bias=f32[10](μ=0.00, σ=0.00, ∈[0.00,0.00])
            )
          )
        )

    Reference:
        - https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
        - https://arxiv.org/abs/1411.4280
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class Dropout3D(DropoutND):
    """Drops full feature maps along the channel axis.

    Args:
        drop_rate: fraction of an elements to be zeroed out.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> layer = sk.nn.Dropout3D(0.5)
        >>> print(layer(jnp.ones((1, 2, 2, 2)), key=jr.PRNGKey(0)))  # doctest: +NORMALIZE_WHITESPACE
        [[[[2. 2.]
        [2. 2.]]
        <BLANKLINE>
        [[2. 2.]
        [2. 2.]]]]

    Note:
        Use :func:`.tree_eval` to turn off dropout during evaluation.

        >>> import serket as sk
        >>> import jax.random as jr
        >>> linear = sk.nn.Linear(10, 10, key=jr.PRNGKey(0))
        >>> dropout = sk.nn.Dropout3D(0.5)
        >>> layers = sk.nn.Sequential(dropout, linear)
        >>> sk.tree_eval(layers)
        Sequential(
          layers=(
            Identity(),
            Linear(
              in_features=(10),
              out_features=10,
              weight_init=glorot_uniform,
              bias_init=zeros,
              weight=f32[10,10](μ=0.01, σ=0.31, ∈[-0.54,0.54]),
              bias=f32[10](μ=0.00, σ=0.00, ∈[0.00,0.00])
            )
          )
        )

    Reference:
        - https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
        - https://arxiv.org/abs/1411.4280
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


class RandomCutoutND(sk.TreeClass):
    def __init__(
        self,
        shape: int | tuple[int],
        cutout_count: int = 1,
        fill_value: int | float = 0,
    ):
        self.shape = canonicalize(shape, ndim=self.spatial_ndim, name="shape")
        self.cutout_count = positive_int_cb(cutout_count)
        self.fill_value = fill_value

    @ft.partial(validate_spatial_nd, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, *, key: jr.KeyArray) -> jax.Array:
        """Drop some elements of the input array.

        Args:
            x: input array
            key: random number generator key
        """
        fill_value = jax.lax.stop_gradient(self.fill_value)

        def cutout(x):
            return random_cutout_nd(key, x, self.shape, self.cutout_count, fill_value)

        return jax.vmap(cutout)(x)

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        ...


class RandomCutout1D(RandomCutoutND):
    """Random Cutouts for spatial 1D array.

    Args:
        shape: shape of the cutout. accepts an int or a tuple of int.
        cutout_count: number of holes. Defaults to 1.
        fill_value: ``fill_value`` to fill the cutout region. Defaults to 0.

    Note:
        Use :func:`.tree_eval` to turn off the cutout during evaluation.

    Examples:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> import jax.random as jr
        >>> print(sk.nn.RandomCutout1D(5)(jnp.ones((1, 10)) * 100, key=jr.PRNGKey(0)))
        [[100. 100. 100. 100. 100.   0.   0.   0.   0.   0.]]

    Reference:
        - https://arxiv.org/abs/1708.04552
        - https://keras.io/api/keras_cv/layers/preprocessing/random_cutout/
    """

    @property
    def spatial_ndim(self) -> int:
        return 1


class RandomCutout2D(RandomCutoutND):
    """Random Cutouts for spatial 2D array

    .. image:: ../_static/randomcutout2d.png

    Args:
        shape: shape of the cutout. accepts int or a two element tuple.
        cutout_count: number of holes. Defaults to 1.
        fill_value: ``fill_value`` to fill the cutout region. Defaults to 0.

    Note:
        Use :func:`.tree_eval` to turn off the cutout during evaluation.


    Examples:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> x = jnp.arange(1,101).reshape(1, 10, 10)
        >>> key = jr.PRNGKey(0)
        >>> print(sk.nn.RandomCutout2D(shape=(3,2), cutout_count=2, fill_value=0)(x,key=key))
        [[[  1   2   3   4   5   6   7   8   9  10]
          [ 11  12  13  14  15  16  17  18  19  20]
          [ 21  22  23  24  25  26  27  28  29  30]
          [ 31  32  33  34   0   0  37  38  39  40]
          [ 41  42  43  44   0   0  47  48  49  50]
          [ 51  52  53  54   0   0  57  58  59  60]
          [ 61  62   0   0  65  66  67  68  69  70]
          [ 71  72   0   0  75  76  77  78  79  80]
          [ 81  82   0   0  85  86  87  88  89  90]
          [ 91  92  93  94  95  96  97  98  99 100]]]

    Reference:
        - https://arxiv.org/abs/1708.04552
        - https://keras.io/api/keras_cv/layers/preprocessing/random_cutout/
    """

    @property
    def spatial_ndim(self) -> int:
        return 2


class RandomCutout3D(RandomCutoutND):
    """Random Cutouts for spatial 2D array

    Args:
        shape: shape of the cutout. accepts int or a three element tuple.
        cutout_count: number of holes. Defaults to 1.
        fill_value: ``fill_value`` to fill the cutout region. Defaults to 0.

    Note:
        Use :func:`.tree_eval` to turn off the cutout during evaluation.


    Examples:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> x = jnp.arange(1, 2 * 5 * 5 + 1).reshape(1, 2, 5, 5)
        >>> key = jr.PRNGKey(0)
        >>> print(sk.nn.RandomCutout3D(shape=(2, 2, 2), cutout_count=2, fill_value=0)(x, key=key))
        [[[[ 1  2  0  0  5]
           [ 6  7  0  0 10]
           [ 0  0 13 14 15]
           [ 0  0 18 19 20]
           [21 22 23 24 25]]
        <BLANKLINE>
         [[26 27  0  0 30]
          [31 32  0  0 35]
          [ 0  0 38 39 40]
          [ 0  0 43 44 45]
          [46 47 48 49 50]]]]

    Reference:
        - https://arxiv.org/abs/1708.04552
        - https://keras.io/api/keras_cv/layers/preprocessing/random_cutout/
    """

    @property
    def spatial_ndim(self) -> int:
        return 3


@tree_eval.def_eval(RandomCutoutND)
@tree_eval.def_eval(DropoutND)
@tree_eval.def_eval(Dropout)
def _(_) -> sk.nn.Identity:
    return sk.nn.Identity()
