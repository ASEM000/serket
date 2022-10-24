from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from serket.nn.utils import _check_and_return


@pytc.treeclass
class RandomCutout1D:
    shape: int = pytc.nondiff_field()
    cutout_count: int = pytc.nondiff_field()
    fill_value: float = pytc.nondiff_field()

    def __init__(
        self,
        shape: tuple[int, ...],
        cutout_count: int = 1,
        fill_value: int | float = 0,
        ndim=1,
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
        self.shape = _check_and_return(shape, ndim=ndim, name="shape")
        self.cutout_count = cutout_count
        self.fill_value = fill_value

    def __call__(
        self, x: jnp.ndarray, *, key: jr.PRNGKey = jr.PRNGKey(0)
    ) -> jnp.ndarray:
        msg = f"Input must have 2 dimensions, got {x.ndim}."
        assert x.ndim == 2, msg

        size = self.shape[0]
        row_arange = jnp.arange(x.shape[1])

        keys = jr.split(key, self.cutout_count)

        def scan_step(x, key):
            start = jr.randint(
                key, shape=(), minval=0, maxval=x.shape[1] - size
            ).astype(jnp.int32)
            row_mask = (row_arange >= start) & (row_arange < start + size)
            x = x * ~row_mask[None, :]
            return x, None

        x, _ = jax.lax.scan(scan_step, x, keys)

        if self.fill_value != 0:
            return jnp.where(x == 0, self.fill_value, x)

        return x


@pytc.treeclass
class RandomCutout2D:
    shape: tuple[int, int] = pytc.nondiff_field()
    cutout_count: int = pytc.nondiff_field()
    fill_value: float = pytc.nondiff_field()

    def __init__(
        self,
        shape: int | tuple[int, ...],
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
        self.shape = _check_and_return(shape, 2, "shape")
        self.cutout_count = cutout_count
        self.fill_value = fill_value

    def __call__(
        self, x: jnp.ndarray, *, key: jr.PRNGKey = jr.PRNGKey(0)
    ) -> jnp.ndarray:
        msg = f"Input must have 3 dimensions, got {x.ndim}."
        assert x.ndim == 3, msg
        height, width = self.shape
        row_arange = jnp.arange(x.shape[1])
        col_arange = jnp.arange(x.shape[2])

        keys = jr.split(key, self.cutout_count)

        def scan_step(x, key):
            ktop, kleft = jr.split(key, 2)
            top = jr.randint(
                ktop, shape=(), minval=0, maxval=x.shape[1] - self.shape[0]
            ).astype(jnp.int32)
            left = jr.randint(
                kleft, shape=(), minval=0, maxval=x.shape[2] - self.shape[1]
            ).astype(jnp.int32)
            row_mask = (row_arange >= top) & (row_arange < top + height)
            col_mask = (col_arange >= left) & (col_arange < left + width)
            x = x * (~jnp.outer(row_mask, col_mask))
            return x, None

        x, _ = jax.lax.scan(scan_step, x, keys)

        if self.fill_value != 0:
            return jnp.where(x == 0, self.fill_value, x)

        return x
