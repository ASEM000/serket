from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc
from jax.nn import sigmoid as σ

from serket.nn import Linear


@pytc.treeclass
class RNNState:
    hidden_state: jnp.ndarray


@pytc.treeclass
class RNNCell:
    in_features: int = pytc.nondiff_field()
    hidden_features: int = pytc.nondiff_field()

    in_and_hidden_to_hidden: Linear

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        weight_init_func: str | Callable = "he_norma",
        bias_init_func: str | Callable | None = "zeros",
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """Vanilla RNN cell that defines the update rule for the hidden state
        See:
            https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

        Args:
            in_features: input features
            hidden_features: hidden features
            weight_init_func: weight initialization function . Defaults to "he_normal".
            bias_init_func: bias initialization function . Defaults to zeros.
            key: Random key for weight and bias initialization. Defaults to jr.PRNGKey(0).
        """

        k1, k2 = jr.split(key, 2)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.in_and_hidden_to_hidden = Linear(
            in_features=in_features + hidden_features,
            out_features=hidden_features,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=key,
        )

    def __call__(
        self, x: jnp.ndarray, state: RNNState, **kwargs
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = state.hidden_state
        h = jnp.tanh(self.in_and_hidden_to_hidden(jnp.concatenate([x, h], axis=-1)))
        return h, h


@pytc.treeclass
class LSTMState:
    hidden_state: jnp.ndarray
    cell_state: jnp.ndarray


@pytc.treeclass
class LSTMCell:
    in_features: int = pytc.nondiff_field()
    hidden_features: int = pytc.nondiff_field()
    in_and_hidden_to_hidden: Linear

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        weight_init_func: str | Callable = "he_normal",
        bias_init_func: str | Callable | None = "zeros",
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """Defines the update rule for the hidden state and the cell state

        See:
            https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
            https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py

        Args:
            in_features: input features
            hidden_features: hidden features
            weight_init_func: weight initialization function . Defaults to "he_normal".
            bias_init_func: bias initialization function . Defaults to zeros.
            key: Random key for weight and bias initialization. Defaults to jr.PRNGKey(0).
        """
        self.hidden_features = hidden_features
        self.in_features = in_features

        self.in_and_hidden_to_hidden = Linear(
            in_features=in_features + hidden_features,
            out_features=4 * hidden_features,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=key,
        )

    def __call__(
        self, x: jnp.ndarray, state: LSTMState, **kwargs
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:

        h, c = state.hidden_state, state.cell_state
        ifgo = self.in_and_hidden_to_hidden(jnp.concatenate([x, h], axis=-1))
        i, f, g, o = jnp.split(ifgo, 4, axis=-1)
        i, f, g, o = σ(i), σ(f), jnp.tanh(g), σ(o)
        c = f * c + i * g
        h = o * jnp.tanh(c)
        return LSTMState(h, c)
