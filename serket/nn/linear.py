from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc


@pytc.treeclass(field_only=True)
class Linear:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int = pytc.static_field()
    out_features: int = pytc.static_field()
    weight_init_func: Callable = pytc.static_field()
    bias_init_func: Callable = pytc.static_field()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        weight_init_func: Callable = jax.nn.initializers.he_normal(),
        bias_init_func: Callable = lambda key, shape: jnp.ones(shape),
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):

        self.in_features = in_features
        self.out_features = out_features
        self.weight_init_func = weight_init_func
        self.bias_init_func = bias_init_func

        self.weight = weight_init_func(key, (in_features, out_features))
        self.bias = (
            bias_init_func(key, (out_features,)) if (bias_init_func is not None) else 0
        )

    def __call__(self, x, **kwargs):
        return x @ self.weight + self.bias
