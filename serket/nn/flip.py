from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import pytreeclass as pytc

from serket.nn.callbacks import validate_spatial_in_shape


class FlipLeftRight2D(pytc.TreeClass):
    def __init__(self):
        """Flip channels left to right.

        Note:
            See: https://github.com/deepmind/dm_pix/blob/master/dm_pix/_src/augment.py

        Examples:
            >>> x = jnp.arange(1,10).reshape(1,3, 3)
            >>> x
            [[[1 2 3]
            [4 5 6]
            [7 8 9]]]

            >>> FlipLeftRight2D()(x)
            [[[3 2 1]
            [6 5 4]
            [9 8 7]]]
        """
        self.spatial_ndim = 2

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        flip = lambda x: jnp.flip(x, axis=1)
        return jax.vmap(flip)(x)


class FlipUpDown2D(pytc.TreeClass):
    def __init__(self):
        """Flip channels up to down.

        Note:
            See: https://github.com/deepmind/dm_pix/blob/master/dm_pix/_src/augment.py

        Examples:
            >>> x = jnp.arange(1,10).reshape(1,3, 3)
            >>> x
            [[[1 2 3]
            [4 5 6]
            [7 8 9]]]

            >>> FlipUpDown2D()(x)
            [[[7 8 9]
            [4 5 6]
            [1 2 3]]]
        """
        self.spatial_ndim = 2

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        flip = lambda x: jnp.flip(x, axis=0)
        return jax.vmap(flip)(x)
