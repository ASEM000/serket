from __future__ import annotations

import jax
import jax.numpy as jnp
import pytreeclass as pytc

from serket.nn.callbacks import non_negative_scalar_cbs, positive_int_cb


def layer_norm(
    x: jax.Array,
    *,
    γ: jax.Array,
    β: jax.Array,
    ε: float,
    normalized_shape: int | tuple[int],
) -> jax.Array:
    """Layer Normalization
    See: https://nn.labml.ai/normalization/layer_norm/index.html
    transform the input by scaling and shifting to have zero mean and unit variance.

    Args:
        x: input array
        γ: scale
        β: shift
        ε: a value added to the denominator for numerical stability.
        normalized_shape: the shape of the input to be normalized.
    """
    dims = tuple(range(len(x.shape) - len(normalized_shape), len(x.shape)))

    μ = jnp.mean(x, axis=dims, keepdims=True)
    σ_2 = jnp.var(x, axis=dims, keepdims=True)
    x̂ = (x - μ) * jax.lax.rsqrt((σ_2 + ε))

    if γ is not None and β is not None:
        return x̂ * γ + β
    return x̂


def group_norm(
    x: jax.Array,
    *,
    γ: jax.Array,
    β: jax.Array,
    ε: float,
    in_features: int,
    groups: int,
) -> jax.Array:
    """Group Normalization
    See: https://nn.labml.ai/normalization/group_norm/index.html
    transform the input by scaling and shifting to have zero mean and unit variance.

    Args:
        x: input array
        γ: scale
        β: shift
        ε: a value added to the denominator for numerical stability.
        in_features: number of input features
        groups: number of groups to separate the channels into
    """
    if len(x.shape) <= 1:
        raise ValueError("Input must have at least 2 dimensions")

    # split channels into groups
    xx = x.reshape(groups, in_features // groups, *x.shape[1:])
    dims = tuple(range(1, x.ndim + 1))

    μ = jnp.mean(xx, axis=dims, keepdims=True)
    σ_2 = jnp.var(xx, axis=dims, keepdims=True)
    x̂ = (xx - μ) * jax.lax.rsqrt((σ_2 + ε))
    x̂ = x̂.reshape(*x.shape)

    if γ is not None and β is not None:
        γ = jnp.expand_dims(γ, axis=(dims[:-1]))
        β = jnp.expand_dims(β, axis=(dims[:-1]))
        x̂ = x̂ * γ + β
    return x̂


class LayerNorm(pytc.TreeClass):
    γ: jax.Array = None
    β: jax.Array = None
    ε: float = pytc.field(callbacks=[*non_negative_scalar_cbs])

    affine: bool
    normalized_shape: int | tuple[int]

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

        # make γ and β trainable
        self.γ = jnp.ones(normalized_shape) if self.affine else None
        self.β = jnp.zeros(normalized_shape) if self.affine else None

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        return layer_norm(
            x,
            γ=self.γ,
            β=self.β,
            ε=self.ε,
            normalized_shape=self.normalized_shape,
        )


class GroupNorm(pytc.TreeClass):
    γ: jax.Array = None
    β: jax.Array = None
    ε: float = pytc.field(callbacks=[*non_negative_scalar_cbs])

    in_features: int = pytc.field(callbacks=[positive_int_cb])
    groups: int = pytc.field(callbacks=[positive_int_cb])
    affine: bool

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
        # checked by callbacks
        self.in_features = in_features
        self.groups = groups
        self.ε = eps
        self.affine = affine

        # needs more info for checking
        if in_features % groups != 0:
            msg = f"in_features must be divisible by groups. Got {in_features} and {groups}"
            raise ValueError(msg)

        # make γ and β trainable
        self.γ = jnp.ones(self.in_features) if self.affine else None
        self.β = jnp.zeros(self.in_features) if self.affine else None

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return group_norm(
            x=x,
            γ=self.γ,
            β=self.β,
            ε=self.ε,
            in_features=self.in_features,
            groups=self.groups,
        )


class InstanceNorm(GroupNorm):
    def __init__(
        self,
        in_features: int,
        *,
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
            in_features=in_features,
            groups=in_features,
            eps=eps,
            affine=affine,
        )
