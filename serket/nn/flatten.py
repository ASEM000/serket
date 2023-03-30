from __future__ import annotations

import jax
import jax.numpy as jnp
import pytreeclass as pytc


@pytc.treeclass
class Flatten:
    """
    Args:
        start_dim: the first dim to flatten
        end_dim: the last dim to flatten (inclusive)
    Returns:
        a function that flattens a jnp.ndarray

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> Flatten(0,1)(jnp.ones([1,2,3,4,5])).shape
        (2, 3, 4, 5)
        >>> Flatten(0,2)(jnp.ones([1,2,3,4,5])).shape
        (6, 4, 5)
        >>> Flatten(1,2)(jnp.ones([1,2,3,4,5])).shape
        (1, 6, 4, 5)
        >>> Flatten(-1,-1)(jnp.ones([1,2,3,4,5])).shape
        (1, 2, 3, 4, 5)
        >>> Flatten(-2,-1)(jnp.ones([1,2,3,4,5])).shape
        (1, 2, 3, 20)
        >>> Flatten(-3,-1)(jnp.ones([1,2,3,4,5])).shape
        (1, 2, 60)

    Note:
        https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html?highlight=flatten#torch.nn.Flatten
    """

    start_dim: int = pytc.field(default=0, callbacks=[pytc.freeze])
    end_dim: int = pytc.field(default=-1, callbacks=[pytc.freeze])

    def __call__(self, x: jax.Array) -> jax.Array:
        start_dim = self.start_dim + (
            0 if self.start_dim >= 0 else x.ndim
        )  # normalize start_dim
        end_dim = (
            self.end_dim + 1 + (0 if self.end_dim >= 0 else x.ndim)
        )  # normalize end_dim
        return jax.lax.collapse(x, start_dim, end_dim)


@pytc.treeclass
class Unflatten:
    dim: int = pytc.field(default=0, callbacks=[pytc.freeze])
    shape: tuple = pytc.field(default=None, callbacks=[pytc.freeze])

    """
    
    Example:
        >>> Unflatten(0, (1,2,3,4,5))(jnp.ones([120])).shape
        (1, 2, 3, 4, 5)
        >>> Unflatten(2,(2,3))(jnp.ones([1,2,6])).shape
        (1, 2, 2, 3)

    Note:
        https://pytorch.org/docs/stable/generated/torch.nn.Unflatten.html?highlight=unflatten
    """

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        shape = list(x.shape)
        shape = [*shape[: self.dim], *self.shape, *shape[self.dim + 1 :]]
        return jnp.reshape(x, shape)
