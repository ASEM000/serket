import jax
import jax.numpy as jnp
import pytreeclass as pytc

from serket.nn.utils import _check_spatial_in_shape


@pytc.treeclass
class FlipLeftRight2D:
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
        self.ndim = 2

    @_check_spatial_in_shape
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        flip = lambda x: jnp.flip(x, axis=1)
        return jax.vmap(flip)(x)


@pytc.treeclass
class FlipUpDown2D:
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
        self.ndim = 2

    @_check_spatial_in_shape
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        flip = lambda x: jnp.flip(x, axis=0)
        return jax.vmap(flip)(x)
