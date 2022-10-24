import jax
import jax.numpy as jnp
import pytreeclass as pytc


@pytc.treeclass
class FlipLeftRight2D:
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

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        msg = f"Input must have 3 dimensions, got {x.ndim}."
        assert x.ndim == 3, msg
        flip = lambda x: jnp.flip(x, axis=1)
        return jax.vmap(flip)(x)


@pytc.treeclass
class FlipUpDown2D:
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

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        msg = f"Input must have 3 dimensions, got {x.ndim}."
        assert x.ndim == 3, msg
        flip = lambda x: jnp.flip(x, axis=0)
        return jax.vmap(flip)(x)
