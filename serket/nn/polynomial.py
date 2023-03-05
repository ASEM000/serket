from __future__ import annotations

import functools as ft
from typing import Callable

import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

import serket as sk
from serket.nn.utils import (
    _canonicalize_init_func,
    _canonicalize_positive_int,
    _check_in_features,
    _check_non_tracer,
)


@pytc.treeclass
class Polynomial:
    degree: int = pytc.field(callbacks=[pytc.freeze])
    linears: tuple[sk.nn.Linear, ...]
    bias: jnp.ndarray

    def __init__(
        self,
        in_features: int | tuple[int, ...] | None,
        out_features: int,
        *,
        degree: int = 1,
        weight_init_func: str | Callable = "he_normal",
        bias_init_func: str | Callable = "ones",
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """Polynomial layer
        This layer is a generalization of Quadratic Residual layer https://arxiv.org/pdf/2101.08366.pdf

        Example:
            Degree 1 : y = (w1@x) + b  # Linear layer
            Degree 2 : y = (w1@x) + (w1@x)*(w2@x) + b # Quadratic Residual layer
            Degree n : y = (w1@x) + (w1@x)*(w2@x) + ... + (w1@x)*(w2@x)*...*(wn@x) + b

        Args:
            in_features: number of input features for each input
            out_features: number of output features
            degree: degree of the polynomial
            weight_init_func: function to initialize the weights
            bias_init_func: function to initialize the bias
            key: key for the random number generator
        """
        if in_features is None:
            for field_item in pytc.fields(self):
                setattr(self, field_item.name, None)

            self._init = ft.partial(
                Polynomial.__init__,
                self,
                out_features=out_features,
                degree=degree,
                weight_init_func=weight_init_func,
                bias_init_func=bias_init_func,
                key=key,
            )
            return

        if hasattr(self, "_init"):
            delattr(self, "_init")

        self.degree = _canonicalize_positive_int(degree, "degree")
        self.out_features = _canonicalize_positive_int(out_features, "out_features")

        keys = jr.split(key, self.degree + 1)

        self.linears = [
            sk.nn.Linear(
                in_features=in_features,
                out_features=out_features,
                weight_init_func=weight_init_func,
                bias_init_func=None,
                key=keys[i],
            )
            for i in range(self.degree)
        ]

        self.bias_init_func = _canonicalize_init_func(bias_init_func, "bias_init_func")
        self.bias = self.bias_init_func(keys[-1], (self.out_features,))

    def __call__(self, x, **k):
        if hasattr(self, "_init"):
            _check_non_tracer(*x, self.__class__.__name__)
            getattr(self, "_init")(in_features=x.shape[-1])

        _check_in_features(x, self.linears[0].in_features[0], axis=-1)
        y = jnp.array([li(x) for li in self.linears])
        return jnp.cumproduct(y, axis=0).sum(axis=0) + self.bias
