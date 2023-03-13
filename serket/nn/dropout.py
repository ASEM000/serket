from __future__ import annotations

import functools as ft

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from serket.nn.callbacks import (
    instance_cb_factory,
    range_cb_factory,
    validate_spatial_in_shape,
)

bool_or_none_cb = instance_cb_factory((bool, type(None)))
frozen_in_zero_one_cbs = [range_cb_factory(0, 1), pytc.freeze]


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class Dropout:
    p: float = pytc.field(default=0.5, callbacks=[*frozen_in_zero_one_cbs])
    eval: bool = pytc.field(default=None, callbacks=[bool_or_none_cb])

    def __call__(self, x, *, key: jr.PRNGKey = jr.PRNGKey(0)):
        if self.eval is True:
            return x

        return jnp.where(jr.bernoulli(key, (1 - self.p), x.shape), x / (1 - self.p), 0)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class DropoutND:
    """Drops full feature maps along the channel axis.

    Args:
        p: fraction of an elements to be zeroed out

    Note:
        See:
            https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
            https://arxiv.org/abs/1411.4280

    Example:
        >>> layer = DropoutND(0.5, spatial_ndim=1)
        >>> layer(jnp.ones((1, 10)))
        [[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]]
    """

    spatial_ndim: int = pytc.field(callbacks=[pytc.freeze])
    p: float = pytc.field(default=0.5, callbacks=[*frozen_in_zero_one_cbs])
    eval: bool = pytc.field(default=None, callbacks=[bool_or_none_cb])

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x, *, key=jr.PRNGKey(0)):
        if self.eval is True:
            return x

        mask = jr.bernoulli(key, 1 - self.p, shape=(x.shape[0],))
        return jnp.where(mask, x / (1 - self.p), 0)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class Dropout1D(DropoutND):
    def __init__(self, p: float = 0.5, *, eval: bool = None):
        """Drops full feature maps along the channel axis.

        Args:
            p: fraction of an elements to be zeroed out

        Note:
            See:
                https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
                https://arxiv.org/abs/1411.4280

        Example:
            >>> layer = DropoutND(0.5, spatial_ndim=1)
            >>> layer(jnp.ones((1, 10)))
            [[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]]
        """
        super().__init__(p=p, eval=eval, spatial_ndim=1)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class Dropout2D(DropoutND):
    def __init__(self, p: float = 0.5, *, eval: bool = None):
        """Drops full feature maps along the channel axis.

        Args:
            p: fraction of an elements to be zeroed out

        Note:
            See:
                https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
                https://arxiv.org/abs/1411.4280

        Example:
            >>> layer = DropoutND(0.5, spatial_ndim=1)
            >>> layer(jnp.ones((1, 10)))
            [[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]]
        """
        super().__init__(p=p, eval=eval, spatial_ndim=2)


@ft.partial(pytc.treeclass, leafwise=True, indexing=True)
class Dropout3D(DropoutND):
    def __init__(self, p: float = 0.5, *, eval: bool = None):
        """Drops full feature maps along the channel axis.

        Args:
            p: fraction of an elements to be zeroed out

        Note:
            See:
                https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
                https://arxiv.org/abs/1411.4280

        Example:
            >>> layer = DropoutND(0.5, spatial_ndim=1)
            >>> layer(jnp.ones((1, 10)))
            [[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]]
        """
        super().__init__(p=p, eval=eval, spatial_ndim=3)
