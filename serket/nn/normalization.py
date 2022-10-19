from __future__ import annotations

import dataclasses
import functools as ft

import jax
import jax.numpy as jnp
import pytreeclass as pytc

from serket.nn.utils import _TRACER_ERROR_MSG


@pytc.treeclass
class LayerNorm:
    γ: jnp.ndarray = None
    β: jnp.ndarray = None

    ε: float = pytc.nondiff_field()
    affine: bool = pytc.nondiff_field()
    normalized_shape: int | tuple[int] = pytc.nondiff_field()

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        *,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """Layer Normalization
        See: https://nn.labml.ai/normalization/layer_norm/index.html
        transform the input by scaling and shifting to have zero mean and unit variance.

        Args:
            normalized_shape: the shape of the input to be normalized.
            eps: a value added to the denominator for numerical stability.
            affine: a boolean value that when set to True, this module has learnable affine parameters.
        """
        self.normalized_shape = (
            normalized_shape
            if isinstance(normalized_shape, tuple)
            else (normalized_shape,)
        )
        self.ε = eps
        self.affine = affine

        if self.affine:
            # make γ and β trainable
            self.γ = jnp.ones(normalized_shape)
            self.β = jnp.zeros(normalized_shape)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:

        dims = tuple(range(len(x.shape) - len(self.normalized_shape), len(x.shape)))

        μ = jnp.mean(x, axis=dims, keepdims=True)
        σ_2 = jnp.var(x, axis=dims, keepdims=True)
        x̂ = (x - μ) * jax.lax.rsqrt((σ_2 + self.ε))

        x̂ = (x̂ * self.γ + self.β) if self.affine else x̂

        return x̂


@pytc.treeclass
class GroupNorm:
    γ: jnp.ndarray = None
    β: jnp.ndarray = None

    in_features: int = pytc.nondiff_field()
    groups: int = pytc.nondiff_field()
    ε: float = pytc.nondiff_field()
    affine: bool = pytc.nondiff_field()

    def __init__(
        self,
        in_features,
        *,
        groups: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """Group Normalization
        See: https://nn.labml.ai/normalization/group_norm/index.html
        transform the input by scaling and shifting to have zero mean and unit variance.

        Args:
            in_features : the shape of the input to be normalized.
            groups : number of groups to separate the channels into.
            eps : a value added to the denominator for numerical stability.
            affine : a boolean value that when set to True, this module has learnable affine parameters.
        """
        if in_features is None:
            for field_item in dataclasses.fields(self):
                setattr(self, field_item.name, None)
            self._partial_init = ft.partial(
                GroupNorm.__init__,
                self=self,
                groups=groups,
                eps=eps,
                affine=affine,
            )
            return

        if hasattr(self, "_partial_init"):
            delattr(self, "_partial_init")

        if in_features <= 0 or not isinstance(in_features, int):
            raise ValueError("in_features must be a positive integer")

        if groups <= 0 or not isinstance(groups, int):
            raise ValueError("groups must be a positive integer")

        if in_features % groups != 0:
            raise ValueError(
                f"in_features must be divisible by groups. Got {in_features} and {groups}"
            )

        self.ε = eps
        self.affine = affine
        self.in_features = in_features
        self.groups = groups

        if self.affine:
            # make γ and β trainable
            self.γ = jnp.ones(self.in_features)
            self.β = jnp.zeros(self.in_features)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        if hasattr(self, "_partial_init"):
            if isinstance(x, jax.core.Tracer):
                raise ValueError(_TRACER_ERROR_MSG(self.__class__.__name__))
            self._partial_init(in_features=x.shape[0])

        assert len(x.shape) > 1, "Input must have at least 2 dimensions"
        # split channels into groups
        xx = x.reshape(self.groups, self.in_features // self.groups, *x.shape[1:])
        dims = tuple(range(1, x.ndim + 1))

        μ = jnp.mean(xx, axis=dims, keepdims=True)
        σ_2 = jnp.var(xx, axis=dims, keepdims=True)
        x̂ = (xx - μ) * jax.lax.rsqrt((σ_2 + self.ε))
        x̂ = x̂.reshape(*x.shape)

        if self.affine:
            γ = jnp.expand_dims(self.γ, axis=(dims[:-1]))
            β = jnp.expand_dims(self.β, axis=(dims[:-1]))
            x̂ = x̂ * γ + β
        return x̂


@pytc.treeclass
class InstanceNorm(GroupNorm):
    def __init__(
        self,
        in_features: int,
        *,
        groups: int = 1,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """Instance Normalization
        See: https://nn.labml.ai/normalization/instance_norm/index.html
        transform the input by scaling and shifting to have zero mean and unit variance.

        Args:
            in_features : the shape of the input to be normalized.
            eps : a value added to the denominator for numerical stability.
            affine : a boolean value that when set to True, this module has learnable affine parameters.
        """
        super().__init__(
            in_features=in_features, groups=in_features, eps=eps, affine=affine
        )
