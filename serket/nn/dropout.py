from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from serket.nn.utils import _check_spatial_in_shape


@pytc.treeclass
class Dropout:
    p: float = pytc.nondiff_field()
    eval: bool = None

    def __init__(self, p=0.5, eval=None):
        """
        Args:
            p: dropout probability. Defaults to 0.5.
            eval : if True, dropout is disabled. Defaults to None.

        Note:
            to disable dropout during testing, set eval to True
            using model.at[(model == "eval") & (model == Dropout)].set(True, is_leaf = lambda x:x is None)
        """
        if not (isinstance(eval, bool) or eval is None):
            raise ValueError("eval must be None, True or False")

        if p < 0.0 or p > 1.0:
            raise ValueError("`p` must be in [0, 1]")

        self.p = p
        self.eval = eval

    def __call__(self, x, *, key=jr.PRNGKey(0)):
        if self.eval is True:
            return x

        return jnp.where(jr.bernoulli(key, (1 - self.p), x.shape), x / (1 - self.p), 0)


@pytc.treeclass
class DropoutND:
    p: float = pytc.nondiff_field()
    eval: bool = None
    ndim: int = pytc.nondiff_field()

    def __init__(self, p=0.5, eval=None, ndim=1):
        """Drops full feature maps along the channel axis.

        Args:
            p: fraction of an elements to be zeroed out

        Note:
            See:
                https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
                https://arxiv.org/abs/1411.4280

        Example:
            >>> layer = DropoutND(0.5, ndim=1)
            >>> layer(jnp.ones((1, 10)))
            [[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]]
        """

        if p < 0 or p > 1:
            raise ValueError(f"p must be in [0, 1]. Found {p}")

        if not (isinstance(eval, bool) or eval is None):
            raise ValueError(f"eval must be True, False, or None. Found {eval}")

        self.p = p
        self.eval = eval
        self.ndim = ndim

    @_check_spatial_in_shape
    def __call__(self, x, *, key=jr.PRNGKey(0)):
        if self.eval is True:
            return x

        mask = jr.bernoulli(key, 1 - self.p, shape=(x.shape[0],))
        return jnp.where(mask, x / (1 - self.p), 0)


@pytc.treeclass
class Dropout1D(DropoutND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=1)


@pytc.treeclass
class Dropout2D(DropoutND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=2)


@pytc.treeclass
class Dropout3D(DropoutND):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, ndim=3)
