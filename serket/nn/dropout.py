from __future__ import annotations

import functools as ft

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc
from jax.lax import stop_gradient

from serket.nn.callbacks import range_cb_factory, validate_spatial_in_shape


def dropout(x, *, p: float = 0.5, key: jr.KeyArray = jr.PRNGKey(0)):
    return jnp.where(jr.bernoulli(key, (1 - p), x.shape), x / (1 - p), 0)


def dropout_nd(x, *, p: float = 0.5, key: jr.KeyArray = jr.PRNGKey(0)):
    mask = jr.bernoulli(key, 1 - p, shape=(x.shape[0],))
    return jnp.where(mask, x / (1 - p), 0)


class Dropout(pytc.TreeClass):
    r"""Randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.

    Args:
        p: probability of an element to be zeroed. Default: 0.5

    Example:
        >>> import serket as sk
        >>> import pytreeclass as pytc
        >>> layer = sk.nn.Dropout(0.5)
        >>> # change `p` to 0.0 to turn off dropout
        >>> layer = layer.at["p"].set(0.0, is_leaf=pytc.is_frozen)
    Note:
        Use `p`= 0.0 to turn off dropout.
    """

    p: float = pytc.field(default=0.5, callbacks=[range_cb_factory(0, 1)])

    def __call__(self, x, *, key: jr.KeyArray = jr.PRNGKey(0)):
        return dropout(x, p=stop_gradient(self.p), key=key)


class DropoutND(pytc.TreeClass):
    """Drops full feature maps along the channel axis.

    Args:
        p: fraction of an elements to be zeroed out

    Note:
        https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
        https://arxiv.org/abs/1411.4280

    Example:
        >>> layer = DropoutND(0.5, spatial_ndim=1)
        >>> layer(jnp.ones((1, 10)))
        [[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]]
    """

    spatial_ndim: int = pytc.field(callbacks=[pytc.freeze])
    p: float = pytc.field(default=0.5, callbacks=[range_cb_factory(0, 1)])

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x, *, key=jr.PRNGKey(0)):
        return dropout_nd(x, p=stop_gradient(self.p), key=key)


class Dropout1D(DropoutND):
    def __init__(self, p: float = 0.5):
        """Drops full feature maps along the channel axis.

        Args:

            p: fraction of an elements to be zeroed out

        Note:
            https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
            https://arxiv.org/abs/1411.4280

        Example:
            >>> layer = DropoutND(0.5, spatial_ndim=1)
            >>> layer(jnp.ones((1, 10)))
            [[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]]
        """
        super().__init__(p=p, spatial_ndim=1)


class Dropout2D(DropoutND):
    def __init__(self, p: float = 0.5):
        """Drops full feature maps along the channel axis.

        Args:

            p: fraction of an elements to be zeroed out

        Note:
            https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
            https://arxiv.org/abs/1411.4280

        Example:
            >>> layer = DropoutND(0.5, spatial_ndim=1)
            >>> layer(jnp.ones((1, 10)))
            [[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]]
        """
        super().__init__(p=p, spatial_ndim=2)


class Dropout3D(DropoutND):
    def __init__(self, p: float = 0.5):
        """Drops full feature maps along the channel axis.

        Args:

            p: fraction of an elements to be zeroed out

        Note:
            https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
            https://arxiv.org/abs/1411.4280

        Example:
            >>> layer = DropoutND(0.5, spatial_ndim=1)
            >>> layer(jnp.ones((1, 10)))
            [[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]]
        """
        super().__init__(p=p, spatial_ndim=3)
