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
    _general_linear_einsum_string,
    _multilinear_einsum_string,
)


@pytc.treeclass
class Multilinear:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: tuple[int, ...] | None = pytc.nondiff_field()
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
            msg = f"Expected tuple or int for in_features, got {type(in_features)}"
            raise ValueError(msg)

        self.in_features = in_features
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
class Linear(Multilinear):
    """Linear layer with 1 input applied to last axis of input

    Args:
        in_features: number of input features
        out_features: number of output features
        weight_init_func: function to initialize the weights
        bias_init_func: function to initialize the bias
        key: key for the random number generator

    Example:
        >>> layer = Linear(5, 6)
        >>> layer(jnp.ones((1,5))).shape
        (1, 6)

        * Use with lazy initialization
        >>> x = jnp.linspace(0, 1, 100)[:, None]
        >>> lhs = Linear(None, 10)
        >>> assert lhs(x).shape == (100, 10)
        # here a linear layer is created with in_features=1
        # with weight shape (1, 10) and bias shape (10,)
    """

    def __init__(
        self,
        in_features: int | None,
        out_features: int,
        *,
        weight_init_func: str | Callable = "he_normal",
        bias_init_func: str | Callable = "ones",
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        super().__init__(
            (in_features,),
            out_features,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=key,
        )

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return super().__call__(x, **kwargs)


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
class GeneralLinear:
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: tuple[int, ...] | None = pytc.nondiff_field()
    out_features: tuple[int, ...] | None = pytc.nondiff_field()
    in_axes: tuple[int, ...] | None = pytc.nondiff_field()

    def __init__(
        self,
        in_features: tuple[int, ...] | None,
        out_features: int,
        *,
        in_axes: tuple[int, ...],
        weight_init_func: str | Callable = "he_normal",
        bias_init_func: str | Callable = "ones",
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """Apply a Linear Layer to input at in_axes

        Args:
            in_features: number of input features corresponding to in_axes
            out_features: number of output features
            in_axes: axes to apply the linear layer to
            weight_init_func: weight initialization function
            bias_init_func: bias initialization function
            key: random key

        Example:
            >>> x = jnp.ones([1, 2, 3, 4])
            >>> layer = GeneralLinear(in_features=(1, 2), in_axes=(0, 1), out_features=5)
            >>> assert layer(x).shape == (3, 4, 5)

        Note:
            This layer is similar to to flax linen's DenseGeneral, the difference is that
            this layer uses einsum to apply the linear layer to the specified axes.
        """
        if in_axes is None:
            raise ValueError("in_axes must be specified for GeneralLinear")

        if (
            any([i is None for i in in_features])
            if isinstance(in_features, Sequence)
            else (in_features is None)
        ):
            for field_item in dataclasses.fields(self):
                setattr(self, field_item.name, None)
            self.in_axes = in_axes
            self._partial_init = ft.partial(
                GeneralLinear.__init__,
                self,
                in_axes=in_axes,
                out_features=out_features,
                weight_init_func=weight_init_func,
                bias_init_func=bias_init_func,
                key=key,
            )
            return

        if hasattr(self, "_partial_init"):
            delattr(self, "_partial_init")

        if not isinstance(in_features, tuple):
            raise ValueError(
                f"Expected in_features to be tuple, got {type(in_features)}"
            )

        if not isinstance(in_axes, tuple):
            raise ValueError(f"Expected in_axes to be tuple, got {type(in_axes)}")

        if len(in_axes) != len(in_features):
            raise ValueError(
                f"Expected in_axes and in_features to have the same length, got {len(in_axes)} and {len(in_features)}"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.in_axes = in_axes
        self.weight_init_func = _check_and_return_init_func(
            weight_init_func, "weight_init_func"
        )
        self.bias_init_func = _check_and_return_init_func(
            bias_init_func, "bias_init_func"
        )

        self.weight = self.weight_init_func(key, (*self.in_features, self.out_features))

        if self.bias_init_func is None:
            self.bias = None
        else:
            self.bias = self.bias_init_func(key, (self.out_features,))

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        if hasattr(self, "_partial_init"):
            if isinstance(x, jax.core.Tracer):
                raise ValueError(_TRACER_ERROR_MSG(self.__class__.__name__))

            in_features = tuple(x.shape[i] for i in self.in_axes)
            self._partial_init(in_features=in_features)

        # ensure negative axes
        axes = map(lambda i: i if i < 0 else i - x.ndim, self.in_axes)
        einsum_string = _general_linear_einsum_string(*axes)
        x = jnp.einsum(einsum_string, x, self.weight)
        return x


@pytc.treeclass
class Identity:
    """Identity layer"""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return x
