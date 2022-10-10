from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc


@pytc.treeclass
class Dropout:
    p: float = pytc.nondiff_field(default=0.5)
    eval: bool = None

    def __post_init__(self):
        """
        Args:
            p: dropout probability. Defaults to 0.5.
            eval : if True, dropout is disabled. Defaults to None.

        Note:
            to disable dropout during testing, set eval to True
            using model.at[(model == "eval") & (model == Dropout)].set(True, is_leaf = lambda x:x is None)
        """
        if self.eval not in (None, True, False):
            raise ValueError("eval must be None, True or False")

        if self.p < 0.0 or self.p > 1.0:
            raise ValueError("`p` must be in [0, 1]")

    def __call__(self, x, *, key=jr.PRNGKey(0)):
        if self.eval is True:
            return x

        return jnp.where(jr.bernoulli(key, (1 - self.p), x.shape), x / (1 - self.p), 0)


@pytc.treeclass
class DropoutND:
    p: float = pytc.nondiff_field(default=0.5)
    eval: bool = None
    ndim: int = pytc.nondiff_field(default=1)

    def __post_init__(self):
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

        if self.p < 0 or self.p > 1:
            raise ValueError(f"p must be in [0, 1]. Found {self.p}")

        if self.eval not in (True, False, None):
            raise ValueError(f"eval must be True, False, or None. Found {self.eval}")

    def __call__(self, x, *, key=jr.PRNGKey(0)):
        assert x.ndim == (self.ndim + 1), f"input must be {self.ndim+1}D, got {x.ndim}D"

        if self.eval is True:
            return x

        mask = jr.bernoulli(key, 1 - self.p, shape=(x.shape[0],))
        return jnp.where(mask, x / (1 - self.p), 0)


@pytc.treeclass
class Dropout1D(DropoutND):
    def __init__(self, p: float = 0.5, eval: bool = None):
        super().__init__(p=p, ndim=1, eval=eval)


@pytc.treeclass
class Dropout2D(DropoutND):
    def __init__(self, p: float = 0.5, eval: bool = None):
        super().__init__(p=p, ndim=2, eval=eval)


@pytc.treeclass
class Dropout3D(DropoutND):
    def __init__(self, p: float = 0.5, eval: bool = None):
        super().__init__(p=p, ndim=3, eval=eval)
