from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

__all__ = ["Linear", "Bilinear"]


@pytc.treeclass
class Linear:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int = pytc.nondiff_field()
    out_features: int = pytc.nondiff_field()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        weight_init_func: Callable = jax.nn.initializers.he_normal(),
        bias_init_func: Callable = lambda key, shape: jnp.ones(shape),
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """Fully connected layer

        Args:
            in_features (int): number of input features
            out_features (int): number of output features
            weight_init_func (Callable, optional): . Defaults to jax.nn.initializers.he_normal().
            bias_init_func (Callable, optional): . Defaults to lambdakey.
            key (jr.PRNGKey, optional):  . Defaults to jr.PRNGKey(0).

        Examples:
            >>> l1 = Linear(10, 5)
            >>> l1(jnp.ones((3, 10))).shape
            (3, 5)
        """

        self.in_features = in_features
        self.out_features = out_features

        self.weight = weight_init_func(key, (in_features, out_features))
        self.bias = (
            bias_init_func(key, (out_features,)) if (bias_init_func is not None) else 0
        )

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return x @ self.weight + self.bias


@pytc.treeclass
class Bilinear:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in1_features: int = pytc.nondiff_field()
    in2_features: int = pytc.nondiff_field()
    out_features: int = pytc.nondiff_field()

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        *,
        weight_init_func: Callable = jax.nn.initializers.he_normal(),
        bias_init_func: Callable = lambda key, shape: jnp.ones(shape),
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """Bilinear layer
        see: https://pytorch.org/docs/stable/generated/torch.nn.Bilinear.html

        Args:
            in_features (int): number of input features
            out_features (int): number of output features
            weight_init_func (Callable, optional): . Defaults to jax.nn.initializers.he_normal().
            bias_init_func (Callable, optional): . Defaults to lambdakey.
            key (jr.PRNGKey, optional):  . Defaults to jr.PRNGKey(0).

        Examples:
            >>> b1 = Bilinear(10, 5, 3)
            >>> b1(jnp.ones((3, 10)), jnp.ones((3, 5))).shape
            (3, 3)
        """

        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features

        self.weight = weight_init_func(key, (in1_features, in2_features, out_features))
        self.bias = (
            bias_init_func(key, (out_features,)) if (bias_init_func is not None) else 0
        )

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = x1 @ self.weight.reshape(-1, self.in2_features * self.out_features)
        x = x2 @ x.reshape(-1, self.out_features)
        return x + self.bias
