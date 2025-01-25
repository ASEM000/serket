# Copyright 2024 serket authors
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

import jax
import jax.numpy as jnp
import jax.random as jr

from serket import TreeClass, autoinit, field
from serket._src.custom_transform import tree_eval
from serket._src.nn.linear import Identity
from serket._src.utils.convert import canonicalize
from serket._src.utils.mapping import kernel_map
from serket._src.utils.validate import (
    IsInstance,
    Range,
    validate_pos_int,
    validate_spatial_ndim,
)


def dropout_nd(
    key: jax.Array,
    input: jax.Array,
    drop_rate: float,
    drop_axes: tuple[int, ...] | None = None,
) -> jax.Array:
    """Drop some elements of the input array.

    Args:
        key: random number generator key
        input: input array
        drop_rate: probability of an element to be zeroed.
        drop_axes: axes to apply dropout. Default: None to apply to all axes.
    """
    shape = (
        input.shape
        if drop_axes is None
        else (input.shape[i] if i in drop_axes else 1 for i in range(input.ndim))
    )

    return jnp.where(
        (keep_prop := (1 - drop_rate)) == 0.0,
        jnp.zeros_like(input),
        jnp.where(jr.bernoulli(key, keep_prop, shape=shape), input / keep_prop, 0),
    )


def random_cutout_nd(
    key: jax.Array,
    input: jax.Array,
    shape: tuple[int, ...],
    cutout_count: int,
    fill_value: int | float,
):
    """Randomly cutout a region of the input array.

    Args:
        key: random number generator key
        input: input array
        shape: shape of the cutout region. acecepts a tuple of int.
        cutout_count: number of holes.
        fill_value: fill_value to fill.
    """
    start_indices = [0] * len(shape)
    slice_sizes = [di - (di % ki) for di, ki in zip(input.shape, shape)]
    valid_array = jax.lax.dynamic_slice(input, start_indices, slice_sizes)

    @ft.partial(
        kernel_map,
        shape=input.shape,
        kernel_size=shape,
        strides=shape,
        padding=((0, 0),) * len(shape),
    )
    def generate_patches(input):
        return input

    # get non-overlapping patches
    patches = generate_patches(input)
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
    return jax.lax.dynamic_update_slice(input, depatched, start_indices)


@autoinit
class Dropout(TreeClass):
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
        >>> input = jnp.ones(10)
        >>> key = jr.key(0)
        >>> output = layer(input, key=key)

    Note:
        Use :func:`.tree_eval` to turn off dropout during evaluation by converting
        dropout to :class:`.Identity`.

        >>> import serket as sk
        >>> import jax.random as jr
        >>> class Model(sk.TreeClass):
        ...     def __init__(self, rate):
        ...         self.dropout = sk.nn.Dropout(rate)
        >>> model = Model(rate=0.5)
        >>> sk.tree_eval(model)
        Model(dropout=Identity())
    """

    drop_rate: float = field(
        default=0.5,
        on_setattr=[IsInstance(float), Range(0, 1)],
        on_getattr=[jax.lax.stop_gradient_p.bind],
    )
    drop_axes: tuple[int, ...] | None = None

    def __call__(self, input: jax.Array, *, key: jax.Array | None = None) -> jax.Array:
        """Drop some elements of the input array.

        Args:
            x: input array
            key: random number generator key
        """
        return dropout_nd(key, input, self.drop_rate, self.drop_axes)


@autoinit
class DropoutND(TreeClass):
    drop_rate: float = field(
        default=0.5,
        on_setattr=[IsInstance(float), Range(0, 1)],
        on_getattr=[jax.lax.stop_gradient],
    )

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, input: jax.Array, *, key: jax.Array | None = None) -> jax.Array:
        """Drop some elements of the input array.

        Args:
            input: input array
            key: random number generator key
        """
        return dropout_nd(key, input, self.drop_rate, (0,))

    @property
    @abc.abstractmethod
    def spatial_ndim(self): ...


class Dropout1D(DropoutND):
    """Drops full feature maps along the channel axis.

    Args:
        drop_rate: fraction of an elements to be zeroed out.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> layer = sk.nn.Dropout1D(0.5)
        >>> input = jnp.ones((1, 10))
        >>> key = jr.key(0)
        >>> output = layer(input, key=key)

    Note:
        Use :func:`.tree_eval` to turn off dropout during evaluation by converting
        dropout to :class:`.Identity`.

        >>> import serket as sk
        >>> class Model(sk.TreeClass):
        ...     def __init__(self, rate):
        ...         self.dropout = sk.nn.Dropout1D(rate)
        >>> model = Model(rate=0.5)
        >>> sk.tree_eval(model)
        Model(dropout=Identity())

    Reference:
        - https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
        - https://arxiv.org/abs/1411.4280
    """

    spatial_ndim: int = 1


class Dropout2D(DropoutND):
    """Drops full feature maps along the channel axis.

    Args:
        drop_rate: fraction of an elements to be zeroed out.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> layer = sk.nn.Dropout2D(0.5)
        >>> input = jnp.ones((1, 5, 5))
        >>> key = jr.key(0)
        >>> output = layer(input, key=key)

    Note:
        Use :func:`.tree_eval` to turn off dropout during evaluation by converting
        dropout to :class:`.Identity`.

        >>> import serket as sk
        >>> class Model(sk.TreeClass):
        ...     def __init__(self, rate):
        ...         self.dropout = sk.nn.Dropout2D(rate)
        >>> model = Model(rate=0.5)
        >>> sk.tree_eval(model)
        Model(dropout=Identity())

    Reference:
        - https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
        - https://arxiv.org/abs/1411.4280
    """

    spatial_ndim: int = 2


class Dropout3D(DropoutND):
    """Drops full feature maps along the channel axis.

    Args:
        drop_rate: fraction of an elements to be zeroed out.

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> layer = sk.nn.Dropout3D(0.5)
        >>> input = jnp.ones((1, 2, 2, 2))
        >>> key = jr.key(0)
        >>> output = layer(input, key=key)

    Note:
        Use :func:`.tree_eval` to turn off dropout during evaluation by converting
        dropout to :class:`.Identity`.

        >>> import serket as sk
        >>> class Model(sk.TreeClass):
        ...     def __init__(self, rate):
        ...         self.dropout = sk.nn.Dropout3D(rate)
        >>> model = Model(rate=0.5)
        >>> sk.tree_eval(model)
        Model(dropout=Identity())

    Reference:
        - https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
        - https://arxiv.org/abs/1411.4280
    """

    spatial_ndim: int = 3


class RandomCutoutND(TreeClass):
    def __init__(
        self,
        shape: int | tuple[int],
        cutout_count: int = 1,
        fill_value: int | float = 0,
    ):
        self.shape = canonicalize(shape, ndim=self.spatial_ndim, name="shape")
        self.cutout_count = validate_pos_int(cutout_count)
        self.fill_value = fill_value

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, input: jax.Array, *, key: jax.Array | None = None) -> jax.Array:
        """Drop some elements of the input array.

        Args:
            x: input array
            key: random number generator key
        """
        fill_value = jax.lax.stop_gradient(self.fill_value)
        in_axes = (None, 0, None, None, None)
        args = (key, input, self.shape, self.cutout_count, fill_value)
        return jax.vmap(random_cutout_nd, in_axes=in_axes)(*args)

    spatial_ndim = property(abc.abstractmethod(lambda _: ...))


class RandomCutout1D(RandomCutoutND):
    """Random Cutouts for spatial 1D input.

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
        >>> layer = sk.nn.RandomCutout1D(5)
        >>> input = jnp.ones((1, 10)) * 100
        >>> key = jr.key(0)
        >>> output = layer(input, key=key)

    Reference:
        - https://arxiv.org/abs/1708.04552
        - https://keras.io/api/keras_cv/layers/preprocessing/random_cutout/
    """

    spatial_ndim: int = 1


class RandomCutout2D(RandomCutoutND):
    """Random Cutouts for spatial 2D input

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
        >>> layer = sk.nn.RandomCutout2D(shape=(3,2), cutout_count=2, fill_value=0)
        >>> input = jnp.arange(1,101).reshape(1, 10, 10)
        >>> key = jr.key(0)
        >>> output = layer(input, key=key)

    Reference:
        - https://arxiv.org/abs/1708.04552
        - https://keras.io/api/keras_cv/layers/preprocessing/random_cutout/
    """

    spatial_ndim: int = 2


class RandomCutout3D(RandomCutoutND):
    """Random Cutouts for spatial 2D input

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
        >>> layer = sk.nn.RandomCutout3D(shape=(2, 2, 2), cutout_count=2, fill_value=0)
        >>> input = jnp.arange(1, 2 * 5 * 5 + 1).reshape(1, 2, 5, 5)
        >>> key = jr.key(0)
        >>> output = layer(input, key=key)

    Reference:
        - https://arxiv.org/abs/1708.04552
        - https://keras.io/api/keras_cv/layers/preprocessing/random_cutout/
    """

    spatial_ndim: int = 3


@tree_eval.def_eval(RandomCutoutND)
@tree_eval.def_eval(DropoutND)
@tree_eval.def_eval(Dropout)
def _(_) -> Identity:
    return Identity()
