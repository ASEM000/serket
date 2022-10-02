from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc
from pytreeclass._src.tree_util import is_treeclass


@pytc.treeclass
class Dropout:
    p: float
    eval: bool | None

    def __init__(self, p: float = 0.5, eval: bool | None = None):
        """p : probability of an element to be zeroed out"""

        # to disable dropout during testing, set eval to True
        # using model.at[(model == "eval") & (model == Dropout)].set(True, is_leaf = lambda x:x is None)
        # during training, set eval to any non True value
        self.p = p
        self.eval = eval

    def __call__(self, x, *, key=jr.PRNGKey(0)):
        return (
            x
            if (self.eval is True)
            else jnp.where(
                jr.bernoulli(key, (1 - self.p), x.shape), x / (1 - self.p), 0
            )
        )


@pytc.treeclass
class DropoutND:
    p: float = pytc.nondiff_field()
    eval: bool | None

    def __init__(self, p: float = 0.5, ndim: int = 1, eval: bool | None = None):
        """
        Drops full feature maps along the channel axis.

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
            raise ValueError(f"p must be between 0 and 1, got {p}")

        if isinstance(eval, bool) or eval is None:
            self.eval = eval
        else:
            raise ValueError(f"eval must be a boolean or None, got {eval}")

        self.p = p
        self.ndim = ndim

    def __call__(self, x, *, key=jr.PRNGKey(0)):
        assert x.ndim == (self.ndim + 1), f"input must be {self.ndim+1}D, got {x.ndim}D"

        if self.eval is True:
            return x

        mask = jr.bernoulli(key, 1 - self.p, shape=(x.shape[0],))
        return jnp.where(mask, x / (1 - self.p), 0)


@pytc.treeclass
class Dropout1D(DropoutND):
    def __init__(self, p: float = 0.5):
        super().__init__(p=p, ndim=1)


@pytc.treeclass
class Dropout2D(DropoutND):
    def __init__(self, p: float = 0.5):
        super().__init__(p=p, ndim=2)


@pytc.treeclass
class Dropout3D(DropoutND):
    def __init__(self, p: float = 0.5):
        super().__init__(p=p, ndim=3)


@pytc.treeclass
class RandomApply:
    layer: int
    p: float = pytc.nondiff_field(default=1.0)
    eval: bool | None

    def __init__(self, layer, p: float = 0.5, ndim: int = 1, eval: bool | None = None):
        """
        Randomly applies a layer with probability p.

        Args:
            p: probability of applying the layer

        Example:
            >>> layer = RandomApply(sk.nn.MaxPool2D(kernel_size=2, strides=2), p=0.0)
            >>> layer(jnp.ones((1, 10, 10))).shape
            (1, 10, 10)

            >>> layer = RandomApply(sk.nn.MaxPool2D(kernel_size=2, strides=2), p=1.0)
            >>> layer(jnp.ones((1, 10, 10))).shape
            (1, 5, 5)
        """

        if p < 0 or p > 1:
            raise ValueError(f"p must be between 0 and 1, got {p}")

        if isinstance(eval, bool) or eval is None:
            self.eval = eval
        else:
            raise ValueError(f"eval must be a boolean or None, got {eval}")

        self.p = p

        if not is_treeclass(layer):
            raise ValueError("Layer must be a `treeclass`.")
        self.layer = layer

    def __call__(self, x: jnp.ndarray, key: jr.PRNGKey = jr.PRNGKey(0)):

        if self.eval is True or not jr.bernoulli(key, (self.p)):
            return x

        return self.layer(x)
