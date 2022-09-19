from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from serket.nn import Linear


@pytc.treeclass
class RNN:
    in_features: int = pytc.nondiff_field()
    hidden_features: int = pytc.nondiff_field()
    hidden_to_hidden: Linear
    input_to_hidden: Linear

    def __init__(
        self, in_features: int, hidden_features: int, *, key: jr.PRNGKey = jr.PRNGKey(0)
    ):
        """Vanilla RNN
        see https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

        Args:
            in_features (int): input features
            hidden_features (int): hidden features
            key (jr.PRNGKey, optional): random key. Defaults to jr.PRNGKey(0).
        """

        k1, k2 = jr.split(key, 2)

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_to_hidden = Linear(
            in_features=hidden_features, out_features=hidden_features, key=k1
        )
        self.input_to_hidden = Linear(
            in_features=in_features, out_features=hidden_features, key=k2
        )

    def __call__(
        self, x: jnp.ndarray, h: jnp.ndarray, **kwargs
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = jnp.tanh(self.hidden_to_hidden(h) + self.input_to_hidden(x))
        return h, h
