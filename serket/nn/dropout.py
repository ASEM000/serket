from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc


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
