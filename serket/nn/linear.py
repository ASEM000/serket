from __future__ import annotations

import dataclasses
import functools as ft
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from serket.nn.utils import (
    _TRACER_ERROR_MSG,
    _check_and_return_init_func,
    _multilinear_einsum_string,
)


@pytc.treeclass
class Linear:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int | None = pytc.nondiff_field()
    out_features: int = pytc.nondiff_field()

    def __init__(
        self,
        in_features: int | None,
        out_features: int,
        *,
        weight_init_func: str | Callable = "he_normal",
        bias_init_func: str | Callable = "ones",
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """Linear layer transformation applied to the last dimension of the input.

        Args:
            in_features: number of input features
            out_features: number of output features
            weight_init_func: function to initialize the weight matrix with . Defaults to "he_normal".
            bias_init_func: bias initializer function . Defaults to ones.
            key: Random key for weight and bias initialization. Defaults to jr.PRNGKey(0).

        Examples:
            >>> l1 = Linear(10, 5)
            >>> l1(jnp.ones((3, 10))).shape
            (3, 5)
        """
        if in_features is None:
            for field_item in dataclasses.fields(self):
                setattr(self, field_item.name, None)

            self._partial_init = ft.partial(
                Linear.__init__,
                self,
                out_features=out_features,
                weight_init_func=weight_init_func,
                bias_init_func=bias_init_func,
                key=key,
            )
            return

        if hasattr(self, "_partial_init"):
            delattr(self, "_partial_init")

        self.weight_init_func = _check_and_return_init_func(
            weight_init_func, "weight_init_func"
        )
        self.bias_init_func = _check_and_return_init_func(
            bias_init_func, "bias_init_func"
        )

        self.in_features = in_features
        self.out_features = out_features

        self.weight = self.weight_init_func(key, (in_features, out_features))

        if bias_init_func is None:
            self.bias = None
        else:
            self.bias = self.bias_init_func(key, (out_features,))

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        if hasattr(self, "_partial_init"):
            if isinstance(x, jax.core.Tracer):
                raise ValueError(_TRACER_ERROR_MSG(self.__class__.__name__))
            self._partial_init(in_features=x.shape[-1])

        if self.bias is None:
            return x @ self.weight
        return x @ self.weight + self.bias


@pytc.treeclass
class Multilinear:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int | tuple[int, ...] | None = pytc.nondiff_field()
    out_features: int = pytc.nondiff_field()

    def __init__(
        self,
        in_features: int | tuple[int, ...] | None,
        out_features: int,
        *,
        weight_init_func: str | Callable = "he_normal",
        bias_init_func: str | Callable = "ones",
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """Linear layer with arbitrary number of inputs applied to last axis of each input

        Args:
            in_features: number of input features for each input
            out_features: number of output features
            weight_init_func: function to initialize the weights
            bias_init_func: function to initialize the bias
            key: key for the random number generator

        Example:
            # Bilinear layer
            >>> layer = Multilinear((5,6), 7)
            >>> layer(jnp.ones((1,5)), jnp.ones((1,6))).shape
            (1, 7)

            # Trilinear layer
            >>> layer = Multilinear((5,6,7), 8)
            >>> layer(jnp.ones((1,5)), jnp.ones((1,6)), jnp.ones((1,7))).shape
            (1, 8)

            * Use with lazy initialization
            >>> x = jnp.linspace(0, 1, 100)[:, None]
            >>> lhs = Multilinear(None, 10)
            >>> assert lhs(x, x, x).shape == (100, 10)
            # here a trilinear layer is created with in_features=(1, 1, 1)
            # with weight shape (1, 1, 1, 10) and bias shape (10,)
        """
        if (
            any([i is None for i in in_features])
            if isinstance(in_features, Sequence)
            else (in_features is None)
        ):
            for field_item in dataclasses.fields(self):
                setattr(self, field_item.name, None)

            self._partial_init = ft.partial(
                Multilinear.__init__,
                self,
                out_features=out_features,
                weight_init_func=weight_init_func,
                bias_init_func=bias_init_func,
                key=key,
            )
            return

        if hasattr(self, "_partial_init"):
            delattr(self, "_partial_init")

        if not isinstance(in_features, (tuple, int)):
            raise TypeError(
                f"Expected tuple or int for in_features, got {type(in_features)}"
            )

        self.in_features = (
            (in_features,) if isinstance(in_features, int) else in_features
        )

        self.out_features = out_features

        self.weight_init_func = _check_and_return_init_func(
            weight_init_func, "weight_init_func"
        )
        self.bias_init_func = _check_and_return_init_func(
            bias_init_func, "bias_init_func"
        )

        weight_shape = (*self.in_features, out_features)
        self.weight = self.weight_init_func(key, weight_shape)

        if self.bias_init_func is None:
            self.bias = None
        else:
            self.bias = self.bias_init_func(key, (out_features,))

    def __call__(self, *x, **kwargs) -> jnp.ndarray:
        if hasattr(self, "_partial_init"):
            if any(isinstance(xi, jax.core.Tracer) for xi in x):
                raise ValueError(_TRACER_ERROR_MSG(self.__class__.__name__))
            self._partial_init(in_features=tuple(xi.shape[-1] for xi in x))

        einsum_string = _multilinear_einsum_string(len(self.in_features))
        x = jnp.einsum(einsum_string, *x, self.weight)

        if self.bias is None:
            return x
        return x + self.bias


@pytc.treeclass
class Bilinear(Multilinear):
    def __init__(
        self,
        in1_features: int | None,
        in2_features: int | None,
        out_features: int,
        *,
        weight_init_func: str | Callable = "he_normal",
        bias_init_func: str | Callable = "ones",
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """Bilinear layer

        Args:
            in1_features: number of input features for the first input
            in2_features: number of input features for the second input
            out_features: number of output features
            weight_init_func: function to initialize the weights
            bias_init_func: function to initialize the bias
            key: key for the random number generator

        Example:
            >>> layer = Bilinear(5, 6, 7)
            >>> layer(jnp.ones((1,5)), jnp.ones((1,6))).shape
            (1, 7)
        """
        super().__init__(
            (in1_features, in2_features),
            out_features,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=key,
        )


@pytc.treeclass
class Identity:
    """Identity layer"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return x
