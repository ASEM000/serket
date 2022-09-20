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
        self.hidden_to_hidden = Linear(hidden_features, hidden_features, key=k1)
        self.input_to_hidden = Linear(in_features, hidden_features, key=k2)

    def __call__(
        self, x: jnp.ndarray, h: jnp.ndarray, **kwargs
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = jnp.tanh(self.hidden_to_hidden(h) + self.input_to_hidden(x))
        return h, h


@pytc.treeclass
class LSTMCell:
    in_features: int = pytc.nondiff_field()
    hidden_features: int = pytc.nondiff_field()

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        layer_norm: bool,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        key_hh, key_ih = jr.split(key, 2)

        self.hidden_features = hidden_features
        self.in_features = in_features

        self.hidden_to_hidden = Linear(hidden_features, 4 * hidden_features, key=key_hh)
        self.input_to_hidden = Linear(
            in_features,
            hidden_features,
            bias_init_func=None,
            key=key_ih,
        )

    def __call__(
        self, x: jnp.ndarray, h_c: jnp.ndarray, **kwargs
    ) -> tuple[jnp.ndarray, jnp.ndarray]:

        h, c = h_c
        ifgo = self.hidden_to_hidden(h) + self.input_to_hidden(x)
        i, f, g, o = jnp.split(ifgo, 4, axis=-1)
        c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) + jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh((c))
        return h, c
