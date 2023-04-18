from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.random as jr
import pytreeclass as pytc

from serket.nn.linear import Linear
from serket.nn.utils import ActivationType, InitFuncType


class FNN(pytc.TreeClass):
    layers: Sequence[Linear]
    act_func: Callable

    def __init__(
        self,
        layers: Sequence[int],
        *,
        act_func: ActivationType = jax.nn.relu,
        weight_init_func: InitFuncType = "he_normal",
        bias_init_func: InitFuncType = "ones",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """Fully connected neural network
        Args:
            layers: Sequence of layer sizes
            act_func: Activation function to use. Defaults to jax.nn.relu.
            weight_init_func: Weight initializer function. Defaults to jax.nn.initializers.he_normal().
            bias_init_func: Bias initializer function. Defaults to lambda key, shape: jnp.ones(shape).
            key: Random key for weight and bias initialization. Defaults to jr.PRNGKey(0).

        Example:
            >>> fnn = FNN([10, 5, 2])
            >>> fnn(jnp.ones((3, 10))).shape
            (3, 2)
        """

        keys = jr.split(key, len(layers) - 1)
        self.act_func = act_func

        self.layers = tuple(
            Linear(
                in_features=in_dim,
                out_features=out_dim,
                key=ki,
                weight_init_func=weight_init_func,
                bias_init_func=bias_init_func,
            )
            for (ki, in_dim, out_dim) in (zip(keys, layers[:-1], layers[1:]))
        )

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        *layers, last = self.layers
        for layer in layers:
            x = layer(x)
            x = self.act_func(x)
        return last(x)
