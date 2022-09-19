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

    def __call__(self, x):
        # normalize start_dim and end_dim
        start = self.start_dim if self.start_dim >= 0 else len(x.shape) + self.start_dim
        end = self.end_dim if self.end_dim >= 0 else len(x.shape) + self.end_dim

        shape = list(x.shape[:start])
        shape += [ft.reduce(op.mul, x.shape[start : end + 1])]
        shape += list(x.shape[end + 1 :])
        return jnp.reshape(x, shape)
