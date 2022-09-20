from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from serket.nn import Linear


@pytc.treeclass
class RNNCell:
    in_features: int = pytc.nondiff_field()
    hidden_features: int = pytc.nondiff_field()

    in_and_hidden_to_hidden: Linear

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
        self.in_and_hidden_to_hidden = Linear(
            in_features=in_features + hidden_features,
            out_features=hidden_features,
            key=key,
        )

    def __call__(
        self, x: jnp.ndarray, h: jnp.ndarray, **kwargs
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = jnp.tanh(self.in_and_hidden_to_hidden(jnp.concatenate([x, h], axis=-1)))
        return h, h


@pytc.treeclass
class LSTMCell:
    in_features: int = pytc.nondiff_field()
    hidden_features: int = pytc.nondiff_field()
    forget_bias: float
    in_and_hidden_to_hidden: Linear

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        layer_norm: bool,
        forget_bias: float = 0.0,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):

        self.hidden_features = hidden_features
        self.in_features = in_features
        self.forget_bias = forget_bias

        self.in_and_hidden_to_hidden = Linear(
            in_features=in_features + hidden_features,
            out_features=hidden_features,
            bias_init_func=None,
            key=key,
        )

    def __call__(
        self, x: jnp.ndarray, h_c: jnp.ndarray, **kwargs
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:

        h, c = h_c
        ifgo = self.in_and_hidden_to_hidden(jnp.concatenate([x, h], axis=-1))
        i, f, g, o = jnp.split(ifgo, 4, axis=-1)
        f = jax.nn.sigmoid(f + self.forget_bias)
        c = f * c + jax.nn.sigmoid(i) + jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return (h, c), h
