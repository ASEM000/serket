from __future__ import annotations

import functools as ft
import operator as op

import jax.numpy as jnp
import pytreeclass as pytc


@pytc.treeclass
class Flatten:
    start_dim: int = pytc.nondiff_field(default=0)
    end_dim: int = pytc.nondiff_field(default=-1)

    """
    See https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html?highlight=flatten#torch.nn.Flatten

    Args:
        start_dim: the first dim to flatten
        end_dim: the last dim to flatten (inclusive)

    Returns:
        a function that flattens a jnp.ndarray
    
    Example:
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
    """

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        # normalize start_dim and end_dim for negative indices
        start = self.start_dim + (0 if self.start_dim >= 0 else len(x.shape))
        end = self.end_dim + (0 if self.end_dim >= 0 else len(x.shape))

        shape = list(x.shape[:start])
        shape += [ft.reduce(op.mul, x.shape[start : end + 1])]
        shape += list(x.shape[end + 1 :])
        return jnp.reshape(x, shape)


@pytc.treeclass
class Unflatten:
    dim: int = pytc.nondiff_field(default=0)
    shape: tuple = pytc.nondiff_field(default=None)

    """
    See https://pytorch.org/docs/stable/generated/torch.nn.Unflatten.html?highlight=unflatten
    Example:
        >>> Unflatten(0, (1,2,3,4,5))(jnp.ones([120])).shape
        (1, 2, 3, 4, 5)

        >>> Unflatten(2,(2,3))(jnp.ones([1,2,6])).shape
        (1, 2, 2, 3)
    """

    def __call__(self, x: jnp.ndaray, **kwargs) -> jnp.ndarray:
        shape = list(x.shape)
        shape = [*shape[: self.dim], *self.shape, *shape[self.dim + 1 :]]
        return jnp.reshape(x, shape)
